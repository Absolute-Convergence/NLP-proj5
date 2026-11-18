"""Training feedback is teacher-forced and charts do not require generation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from notebook_helpers import load_definitions, load_tag


class TrainingFeedbackTests(unittest.TestCase):
    def test_validation_accuracy_shifts_causal_predictions_and_ignores_prompt_padding(self):
        namespace=load_definitions('model_helpers',{})
        labels=np.array([[-100,-100,2,3,-100],[-100,4,5,-100,-100]])
        predictions=np.array([[99,2,3,99,99],[4,0,99,99,99]])
        self.assertEqual(namespace['validation_metrics']((predictions,labels)),{'target_token_accuracy':0.75})

    def test_real_cpu_trainer_records_validation_metrics_without_generate(self):
        from datasets import Dataset
        from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, default_data_collator
        namespace=load_definitions('model_helpers',{})
        model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_positions=8,n_embd=8,n_layer=1,n_head=1,use_cache=False))
        dataset=Dataset.from_list([
            {'input_ids':[1,2,3,4],'attention_mask':[1]*4,'labels':[-100,-100,3,4]},
            {'input_ids':[1,5,6,7],'attention_mask':[1]*4,'labels':[-100,-100,6,7]}])
        with tempfile.TemporaryDirectory() as temporary, patch.object(model,'generate',side_effect=AssertionError('No generation')) as generate:
            arguments=TrainingArguments(output_dir=temporary,use_cpu=True,max_steps=1,
                per_device_train_batch_size=1,per_device_eval_batch_size=1,eval_strategy='steps',eval_steps=1,
                eval_accumulation_steps=1,save_strategy='no',logging_steps=1,report_to='none',
                disable_tqdm=True,dataloader_pin_memory=False)
            trainer=Trainer(model=model,args=arguments,train_dataset=dataset,eval_dataset=dataset,
                data_collator=default_data_collator,compute_metrics=namespace['validation_metrics'],
                preprocess_logits_for_metrics=namespace['validation_token_ids'])
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train()
            metrics=next(row for row in trainer.state.log_history if 'eval_loss' in row)
            self.assertTrue(np.isfinite(metrics['eval_loss']))
            self.assertTrue(0 <= metrics['eval_target_token_accuracy'] <= 1)
            generate.assert_not_called()

    def test_charts_read_saved_history_with_no_inference_or_optional_results(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        from candleprep.evaluation import score_predictions
        for with_validation,count in ((True,3),(False,7)):
            records=[{'id':str(i),'target':{'recoverable':False,'operations':[]}} for i in range(count)]
            with self.subTest(with_validation=with_validation,count=count),tempfile.TemporaryDirectory() as temporary:
                destination=Path(temporary);rank=destination/'rank_16';rank.mkdir()
                history=[{'step':1,'loss':1.2},{'step':100,'loss':0.5}]
                if with_validation:
                    history.append({'step':100,'eval_loss':0.6,'eval_target_token_accuracy':0.75})
                (rank/'trainer_state.json').write_text(json.dumps({'log_history':history}))
                (destination/f'before_{count}_metrics.json').write_text(json.dumps(score_predictions(['bad']*count,records)))
                (rank/f'after_{count}_metrics.json').write_text(json.dumps(score_predictions(
                    ['{"recoverable":false,"operations":[]}']*count,records)))
                figures=[]
                namespace={'RESULTS':destination,'LORA_RANK':16,'DEMO_EXAMPLES':count,'pd':pd,'plt':plt,'json':json,
                    'display':Mock(),'load_base':Mock(),'predict':Mock(),'train_adapter':Mock()}
                with patch.object(plt,'show',side_effect=lambda:figures.append(plt.gcf())):
                    load_tag('results_display',namespace)
                self.assertEqual(len(figures),2)
                self.assertEqual(len(figures[0].axes[0].lines),2 if with_validation else 1)
                self.assertEqual(len(figures[0].axes[1].lines),1 if with_validation else 0)
                self.assertEqual(namespace['display'].call_args.args[0].shape,(4,2))
                self.assertIn(f'({count} cases)',namespace['display'].call_args.args[0].columns[0])
                self.assertEqual(figures[1].axes[0].get_title(),f'{count}-case generated-answer comparison')
                namespace['load_base'].assert_not_called()
                namespace['predict'].assert_not_called()
                namespace['train_adapter'].assert_not_called()
                plt.close('all')


if __name__=='__main__':
    unittest.main()
