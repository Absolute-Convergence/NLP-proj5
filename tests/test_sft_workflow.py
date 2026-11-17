"""The short notebook trains cached labels and compares five fixed examples."""
import json
import math
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from notebook_helpers import load_tag, load_definitions


def records(count=5):
    return [{'id':str(i),'input':'corrupted '+str(i),
             'target':{'recoverable':False,'operations':[]}} for i in range(count)]


def save_json(path,value):
    Path(path).write_text(json.dumps(value),encoding='utf-8')


def demo_namespace(destination):
    return {'json':json,'RESULTS':destination,'dataset':SimpleNamespace(
        examples=lambda count: records(),display_predictions=Mock()),
        'load_base':Mock(), 'predict':Mock(return_value=['after']*5),
        'save_json':save_json,'user_text':lambda r:r['input'],'release_gpu':Mock(),
        'score_predictions':Mock(return_value={}),
        'evaluate':Mock(side_effect=AssertionError('Full inference must be separate'))}


class SFTWorkflowTests(unittest.TestCase):
    def test_baseline_only_generates_five_then_reuses_frozen_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace=demo_namespace(Path(temporary))
            load_tag('baseline_run',namespace)
            self.assertEqual(len(namespace['predict'].call_args.args[1]),5)
            load_tag('baseline_run',namespace)
            namespace['predict'].assert_called_once()
            namespace['load_base'].assert_called_once()
            namespace['evaluate'].assert_not_called()

    def test_primary_sft_executes_once_without_response_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            before=Path(temporary)/'before.json';before.write_text('{}')
            namespace={'before_path':before,'LORA_RANK':16,'train_adapter':Mock(return_value='adapter'),
                       'predict':Mock(side_effect=AssertionError('SFT must use cached labels'))}
            load_tag('training_run',namespace)
            namespace['train_adapter'].assert_called_once_with(16)
            namespace['predict'].assert_not_called()

    def test_after_sft_compares_the_same_five_ids_prompts_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)
            save_json(destination/'before.json',{'ids':[r['id'] for r in records()],
                'prompts':[r['input'] for r in records()],'predictions':['base']*5})
            namespace=demo_namespace(destination)
            (destination/'rank_16').mkdir()
            namespace.update({'LORA_RANK':16,'PeftModel':SimpleNamespace(from_pretrained=Mock())})
            load_tag('after_run',namespace)
            self.assertEqual([r['id'] for r in namespace['predict'].call_args.args[1]],[str(i) for i in range(5)])
            self.assertEqual(namespace['dataset'].display_predictions.call_args.args[2],['base']*5)
            namespace['evaluate'].assert_not_called()

    def test_disabled_extra_jobs_do_not_load_models_or_train(self):
        namespace={'COMPARISON_EXAMPLES':0,'load_base':Mock(),'train_adapter':Mock()}
        load_tag('sample_comparison',namespace)
        namespace['load_base'].assert_not_called()
        namespace['train_adapter'].assert_not_called()

    def test_optional_generation_uses_only_the_requested_same_sample_for_both_models(self):
        import pandas as pd
        from candleprep.evaluation import score_predictions
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary);(destination/'rank_16').mkdir()
            namespace={'COMPARISON_EXAMPLES':3,'RESULTS':destination,'SEED':4820,'LORA_RANK':16,
                'random':random,'pd':pd,'display':Mock(),'dataset':SimpleNamespace(records=Mock(return_value=records(100))),
                'load_base':Mock(),'PeftModel':SimpleNamespace(from_pretrained=Mock()),'release_gpu':Mock(),
                'predict':Mock(return_value=['{"recoverable":false,"operations":[]}']*3),
                'score_predictions':score_predictions,'save_json':save_json,'train_adapter':Mock()}
            load_tag('sample_comparison',namespace)
            calls=namespace['predict'].call_args_list
            self.assertEqual(len(calls),2)
            self.assertEqual(len(calls[0].args[1]),3)
            self.assertEqual(calls[0].args[1],calls[1].args[1])
            namespace['dataset'].records.assert_called_once_with('test')
            namespace['train_adapter'].assert_not_called()
            saved=json.loads((destination/'rank_16/sample_3.json').read_text())
            self.assertEqual(saved['base']['ids'],saved['selected']['ids'])
            self.assertEqual(saved['selected']['metrics']['examples'],3)

    def test_real_training_function_uses_all_cached_pairs_saves_and_reuses_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)
            model=Mock();model.generate.side_effect=AssertionError('Training is not generation')
            def save_adapter(path):
                path.mkdir(parents=True,exist_ok=True)
                (path/'adapter_model.safetensors').write_bytes(b'test weights')
            model.save_pretrained.side_effect=save_adapter
            trainer=Mock(model=model,state=SimpleNamespace(global_step=625,epoch=1,save_to_json=Mock()))
            trainer.train.return_value=SimpleNamespace(training_loss=0.028)
            trainer.evaluate.return_value={'eval_loss':0.05,'eval_target_token_accuracy':0.9}
            cuda=SimpleNamespace(is_bf16_supported=lambda:True,reset_peak_memory_stats=Mock(),
                max_memory_allocated=lambda:5*1024**3,max_memory_reserved=lambda:6*1024**3)
            namespace={'RESULTS':destination,'settings':{},'json':json,'math':math,'time':time,
                'BATCH_SIZE':1,'ACCUMULATION':16,'EPOCHS':1,'LEARNING_RATE':2e-4,'SEED':4820,
                'save_json':save_json,'get_last_checkpoint':lambda path:None,'load_base':Mock(return_value=model),
                'prepare_model_for_kbit_training':lambda m:m,'get_peft_model':lambda m,c:m,
                'LoraConfig':Mock(),'TrainingArguments':Mock(),'Trainer':Mock(return_value=trainer),
                'DataCollatorForSeq2Seq':Mock(),'tokenizer':Mock(),'train_dataset':range(10000),
                'torch':SimpleNamespace(cuda=cuda),'release_gpu':Mock(),'VALIDATION_EVERY':100,
                'validation_dataset':range(64),'validation_metrics':Mock(),'validation_token_ids':Mock()}
            load_definitions('training_helpers',namespace)
            adapter=namespace['train_adapter'](16)
            namespace['train_adapter'](16)
            namespace['load_base'].assert_called_once()
            trainer.train.assert_called_once_with(resume_from_checkpoint=None)
            trainer.evaluate.assert_called_once_with()
            self.assertEqual(len(namespace['Trainer'].call_args.kwargs['train_dataset']),10000)
            args=namespace['TrainingArguments'].call_args.kwargs
            self.assertTrue(args['gradient_checkpointing'])
            self.assertFalse(args['disable_tqdm'])
            self.assertEqual(args['gradient_accumulation_steps'],16)
            self.assertEqual(args['eval_strategy'],'steps')
            self.assertEqual(args['per_device_eval_batch_size'],1)
            self.assertEqual(len(namespace['Trainer'].call_args.kwargs['eval_dataset']),64)
            report=json.loads((adapter.parent/'training.json').read_text())
            self.assertEqual(report['steps'],625)
            self.assertEqual(report['peak_reserved_gib'],6)
            self.assertEqual(report['validation']['eval_target_token_accuracy'],0.9)
            model.generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
