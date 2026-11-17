import ast
import json
import unittest
from candleprep import CandleDataset
from candleprep.task import TASK, user_text
from notebook_helpers import load_definitions, NOTEBOOK


class NotebookTests(unittest.TestCase):
    def test_simple_structure_explanations_and_syntax(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertLessEqual(sum(len(c['source']) for c in code_cells), 300)
        for index, cell in enumerate(notebook['cells']):
            self.assertIn('id', cell)
            if cell['cell_type'] == 'code':
                self.assertEqual(notebook['cells'][index-1]['cell_type'], 'markdown')
                ast.parse(''.join(s for s in cell['source'] if not s.startswith('%')))
        preparation = ''.join(next(c['source'] for c in code_cells if 'preparation' in c['metadata']['tags']))
        self.assertIn('CandleDataset.prepare(', preparation)
        self.assertNotIn('TVC_GOLD', preparation)

    def test_actual_cached_tokenizer_masks_targets_and_preserves_budget(self):
        from transformers import AutoTokenizer, DataCollatorForSeq2Seq
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3', local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        namespace = load_definitions('tokenization', {'json': json, 'user_text': user_text, 'MAX_LENGTH': 1536})
        dataset = CandleDataset.prepare(source=NOTEBOOK.parent/'TVC_GOLD, 360.csv',
                                       output_dir=NOTEBOOK.parent/'generated_data/v1')
        features = []
        longest = 0
        for record in dataset.records():
            feature = namespace['tokenize_record'](record, tokenizer)
            prompt = namespace['prompt_tokens'](tokenizer, record)
            self.assertEqual(feature['labels'][:len(prompt)], [-100]*len(prompt))
            self.assertEqual(feature['labels'][len(prompt):], feature['input_ids'][len(prompt):])
            self.assertGreater(len(feature['input_ids']), len(prompt))
            self.assertLessEqual(len(prompt)+128, 1536)
            longest = max(longest, len(feature['input_ids']))
            if len(features) < 2:
                features.append(feature)
        collated = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100)(features)
        for row, feature in enumerate(features):
            self.assertTrue(all(int(v) == -100 for v in collated['labels'][row,len(feature['input_ids']):]))
        print('Longest complete training sequence:', longest)

    def test_real_data_previews_and_five_comparisons_are_displayed_without_leakage(self):
        from unittest.mock import patch
        dataset = CandleDataset.prepare(source=NOTEBOOK.parent/'TVC_GOLD, 360.csv',
                                       output_dir=NOTEBOOK.parent/'generated_data/v1')
        shown = []
        with patch('IPython.display.display', side_effect=shown.append):
            dataset.display_samples()
            dataset.display_labels()
            records = dataset.examples()
            predictions = [json.dumps(r['target']) for r in records]
            dataset.display_predictions(records, predictions, ['base']*5)
        tables = [value.data for value in shown if hasattr(value, 'data') and hasattr(value.data, 'columns')]
        self.assertEqual([len(table) for table in tables], [5]*8)
        self.assertTrue(all(v is True for v in tables[-2].iloc[:3]['matches_original']))
        self.assertEqual(len({r['id'] for r in records}), 5)
        self.assertEqual({r['target']['recoverable'] for r in records}, {True, False})
        for record, received in zip(records, tables[-1]['model_received_input']):
            self.assertEqual(received, TASK+record['input'])
            self.assertEqual(received, user_text({**record,'target':'SENTINEL','original':'SENTINEL'}))


if __name__ == '__main__':
    unittest.main()
