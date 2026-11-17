"""The simple inference loop keeps ordering, generation settings, and progress."""
import contextlib
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from notebook_helpers import load_definitions


class NotebookProgressTests(unittest.TestCase):
    def test_prediction_preserves_order_and_decodes_only_new_tokens(self):
        import torch
        progress = []
        def tqdm(records, **kwargs):
            progress.append(kwargs)
            return records
        class Model:
            config = SimpleNamespace(use_cache=False)
            def __init__(self): self.prompts = []
            def eval(self): pass
            def get_input_embeddings(self): return SimpleNamespace(weight=torch.tensor([0]))
            def generate(self, input_ids, **kwargs):
                self.prompts.append(input_ids[0].tolist())
                self.kwargs = kwargs
                return torch.cat([input_ids,torch.tensor([[40,41]])],dim=1)
        tokenizer = SimpleNamespace(pad_token_id=2,eos_token_id=2,decode=lambda ids,**kwargs: str(ids.tolist()))
        namespace = load_definitions('model_helpers', {'json':json,'torch':torch,'tokenizer':tokenizer,
            'tqdm':tqdm,'MAX_NEW_TOKENS':128,'MAX_LENGTH':1536})
        namespace['prompt_tokens'] = lambda t,r: [r['number'],2]
        model = Model()
        result = namespace['predict'](model,[{'number':1},{'number':9}], 'base / validation')
        self.assertEqual(result,['[40, 41]','[40, 41]'])
        self.assertEqual(model.prompts,[[1,2],[9,2]])
        self.assertEqual(model.kwargs['max_new_tokens'],128)
        self.assertIs(model.kwargs['do_sample'],False)
        self.assertEqual(progress,[{'desc':'base / validation','unit':'example'}])

    def test_ten_gib_gpu_is_not_rejected_by_an_arbitrary_capacity_guard(self):
        import torch
        loader = Mock(return_value='quantized model')
        namespace = load_definitions('model_helpers', {'torch':torch,'MODEL_ID':'test','SEED':4820,
            'set_seed':Mock(),'BitsAndBytesConfig':Mock(),
            'AutoModelForCausalLM':SimpleNamespace(from_pretrained=loader)})
        with patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.cuda,'mem_get_info',return_value=(9*1024**3,10*1024**3)), \
             patch.object(torch.cuda,'is_bf16_supported',return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(namespace['load_base'](),'quantized model')
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs['device_map'],{'':0})


if __name__ == '__main__':
    unittest.main()
