"""Progress observers must report activity without changing model results."""
import contextlib
import io
import json
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

from notebook_helpers import load_tag


class FakeBar:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.completed = 0
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def set_postfix_str(self, message, **kwargs):
        self.messages.append(message)

    def update(self, amount):
        self.completed += amount


class NotebookProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_tag("model_helpers", {
            "json": json, "Counter": Counter, "MAX_NEW_TOKENS": 128,
            "MAX_LENGTH": 1536, "RESULTS": None,
        })

    def test_token_observer_never_requests_a_stop(self):
        torch = self.namespace["torch"]
        bar = FakeBar()
        observer = self.namespace["TokenProgress"](bar, prompt_length=2, example_number=1)
        self.assertIs(observer(torch.tensor([[1, 2, 3]]), None), False)
        self.assertIs(observer(torch.tensor([[1, 2, 3, 4]]), None), False)
        self.assertTrue(any("1/128 new tokens" in message for message in bar.messages))

    def test_ten_gib_device_is_not_rejected_by_old_twelve_gib_guard(self):
        namespace = self.namespace
        torch = namespace["torch"]
        function = namespace["load_base"]
        loader = Mock(return_value="quantized model")
        with patch.dict(function.__globals__, {"MIN_FREE_VRAM_GIB": 7.5, "MODEL_ID": "test", "SEED": 4820,
                                             "set_seed": Mock(), "BitsAndBytesConfig": Mock()}), \
             patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "mem_get_info", return_value=(9 * 1024**3, 10 * 1024**3)), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=False), \
             patch.object(namespace["AutoModelForCausalLM"], "from_pretrained", loader):
            self.assertEqual(function(), "quantized model")
            loader.assert_called_once()
            with patch.object(torch.cuda, "mem_get_info", return_value=(2 * 1024**3, 10 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, "availability guard"):
                    function()

    def test_prediction_feedback_preserves_order_and_generation_settings(self):
        namespace = dict(self.namespace)
        torch = namespace["torch"]
        bars = []

        def make_bar(**kwargs):
            bar = FakeBar(**kwargs)
            bars.append(bar)
            return bar

        class Model:
            config = SimpleNamespace(use_cache=False)

            def __init__(self):
                self.prompts = []

            def eval(self):
                pass

            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.tensor([0]))

            def generate(self, input_ids, **kwargs):
                self.prompts.append(input_ids[0].tolist())
                self.last_kwargs = kwargs
                result = torch.cat([input_ids, torch.tensor([[40, 41]])], dim=1)
                for observer in kwargs["stopping_criteria"]:
                    assert observer(result, None) is False
                return result

        tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=2,
                                    decode=lambda ids, **kwargs: str(ids.tolist()))
        function = namespace["predict"]
        output = io.StringIO()
        with patch.dict(function.__globals__, {"tqdm": make_bar, "prompt_tokens": lambda t, r: [r["number"], 2]}), contextlib.redirect_stdout(output):
            model = Model()
            predictions = function(model, tokenizer, [{"number": 1}, {"number": 9}], phase="base / validation")
        self.assertEqual(predictions, ["[40, 41]", "[40, 41]"])
        self.assertEqual(model.prompts, [[1, 2], [9, 2]])
        self.assertEqual(bars[0].completed, 2)
        self.assertEqual(model.last_kwargs["max_new_tokens"], 128)
        self.assertIs(model.last_kwargs["do_sample"], False)
        self.assertIn("base / validation: 2/2 complete", output.getvalue())
        self.assertIn("ETA 00:00:00", output.getvalue())

    def test_training_callback_reports_without_changing_control(self):
        namespace = dict(self.namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            load_tag("training_helpers", namespace)
        callback = namespace["TrainingFeedback"](16)
        args = SimpleNamespace(gradient_accumulation_steps=16)
        state = SimpleNamespace(global_step=0, max_steps=625)
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        with patch.object(namespace["torch"].cuda, "is_available", return_value=False), contextlib.redirect_stdout(output):
            self.assertIsNone(callback.on_train_begin(args, state, control))
            self.assertIsNone(callback.on_substep_end(args, state, control))
            state.global_step = 1
            self.assertIsNone(callback.on_step_end(args, state, control))
            self.assertIsNone(callback.on_save(args, state, control))
            self.assertIsNone(callback.on_train_end(args, state, control))
        self.assertEqual(vars(control), {"should_training_stop": False, "should_save": False})
        self.assertIn("accumulating 1/16", output.getvalue())
        self.assertIn("optimizer step 1/625", output.getvalue())
        self.assertIn("checkpoint saved", output.getvalue())


if __name__ == "__main__":
    unittest.main()
