"""Offline labels must reach SFT without a full response-generation pass."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from candleprep.datasets import load_jsonl
from notebook_helpers import load_tag


def records(count=3):
    return [{"id": str(index), "input": "corrupted " + str(index),
             "target": {"recoverable": False, "operations": []}} for index in range(count)]


def demo_namespace(destination):
    return {
        "json": json, "RESULTS": destination, "MODEL_ID": "test-base", "SEED": 4820,
        "MAX_NEW_TOKENS": 128, "demo_records": records(), "load_base": Mock(),
        "predict": Mock(return_value=['{"recoverable":false,"operations":[]}'] * 3),
        "show_model_examples": Mock(), "save_records": Mock(), "save_json": Mock(),
        "render_prompt": lambda record: record["input"], "report_phase": Mock(),
        "score_predictions": Mock(return_value={}), "run_environment": Mock(return_value={}),
        "release_gpu": Mock(), "tokenizer": object(),
        "FULL_EVALUATION": True,  # An old kernel value must not restore full generation.
        "evaluate_and_save": Mock(side_effect=AssertionError("Full inference must be separate")),
    }


class SFTWorkflowTests(unittest.TestCase):
    def test_fresh_kernel_explains_setup_without_loading_a_model(self):
        for tag in ('after_run', 'rank_study_run', 'full_comparison_run'):
            with self.subTest(tag=tag), self.assertRaisesRegex(RuntimeError, 'restart checklist'):
                load_tag(tag, {'load_base': Mock(side_effect=AssertionError('No model should load'))})

    def test_before_section_only_generates_three_demonstrations(self):
        namespace = demo_namespace(Path("unused"))
        load_tag("baseline_run", namespace)
        namespace["predict"].assert_called_once()
        self.assertEqual(len(namespace["predict"].call_args.args[2]), 3)
        namespace["evaluate_and_save"].assert_not_called()
        namespace["release_gpu"].assert_called_once()

    def test_primary_sft_starts_once_without_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "demo_before.jsonl").write_text("{}\n", encoding="utf-8")
            namespace = {
                "RESULTS": destination, "LORA_RANK": 16, "RANKS": [8, 16, 32],
                "tokenized_train": range(10000), "report_phase": Mock(), "save_json": Mock(),
                "train_adapter": Mock(return_value=destination / "rank_16/adapter"),
                "predict": Mock(side_effect=AssertionError("SFT uses cached labels")),
                "evaluate_and_save": Mock(side_effect=AssertionError("Evaluation must not block SFT")),
            }
            load_tag("training_run", namespace)
            namespace["train_adapter"].assert_called_once_with(16, destination / "rank_16")
            namespace["predict"].assert_not_called()
            namespace["evaluate_and_save"].assert_not_called()
            self.assertEqual(namespace["save_json"].call_args.args[1]["rank"], 16)

    def test_after_section_keeps_three_comparisons_and_adds_two_inspections(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            examples = [{"id": record["id"], "prompt": record["input"], "prediction": "before"}
                        for record in records()]
            (destination / "demo_before.jsonl").write_text(
                "\n".join(json.dumps(example) for example in examples), encoding="utf-8")
            (destination / "demo_generation.json").write_text(json.dumps({
                "model": "test-base", "seed": 4820, "max_new_tokens": 128, "do_sample": False}), encoding="utf-8")
            (destination / "selection.json").write_text('{"rank":16}', encoding="utf-8")
            namespace = demo_namespace(destination)
            namespace.update({"load_jsonl": load_jsonl, "data": {"validation": records(5)},
                              "choose_inspection_records": Mock(return_value=records(5)),
                              "PeftModel": SimpleNamespace(from_pretrained=Mock())})
            namespace["predict"].return_value = ['{"recoverable":false,"operations":[]}'] * 5
            load_tag("after_run", namespace)
            namespace["predict"].assert_called_once()
            self.assertEqual([record["id"] for record in namespace["predict"].call_args.args[2]], ["0", "1", "2", "3", "4"])
            self.assertEqual([len(call.args[1]) for call in namespace["score_predictions"].call_args_list], [3, 5])
            shown = namespace["show_model_examples"].call_args.args
            self.assertEqual([example["id"] for example in shown[2]], ["0", "1", "2"])
            namespace["evaluate_and_save"].assert_not_called()

    def test_full_evaluation_and_rank_study_are_explicitly_opt_in(self):
        for tag, flag in (("rank_study_run", "RUN_RANK_STUDY"),
                          ("full_comparison_run", "RUN_FULL_TEST_COMPARISON")):
            with self.subTest(tag=tag):
                namespace = {flag: False, "report_phase": Mock(),
                             "load_base": Mock(side_effect=AssertionError("No model should load")),
                             "train_adapter": Mock(side_effect=AssertionError("No training should start"))}
                load_tag(tag, namespace)
                namespace["load_base"].assert_not_called()
                namespace["train_adapter"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
