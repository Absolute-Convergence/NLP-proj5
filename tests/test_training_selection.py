"""Sampling shortens SFT while preserving complete pairs and checkpoint identity."""
import ast
import contextlib
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from notebook_helpers import NOTEBOOK, load_tag


def configure(destination, count=1000, seed=4820, demos=5):
    """Run the actual settings cell, substituting inputs without ML imports."""
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'configuration' in c.get('metadata', {}).get('tags', []))
    tree = ast.parse(source)
    overrides = {'RESULTS': destination, 'TRAINING_EXAMPLES': count, 'SEED': seed, 'DEMO_EXAMPLES': demos}
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
                 and not (isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id in overrides for t in node.targets))]
    splits = {split: [{'id': f'{split}-{i}', 'input': f'window-{i}',
                       'target': {'recoverable': bool(i % 2), 'operations': []}}
                      for i in range(size)]
              for split, size in [('train', 2000), ('validation', 80), ('test', 80)]}
    dataset = SimpleNamespace(records=Mock(side_effect=lambda split: splits[split]),
                              manifest={'artifact_sha256': {'train.jsonl': 'verified-training-hash'}})
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib, 'random': random,
                 'TASK': 'fixed task', 'dataset': dataset, **overrides}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree, str(NOTEBOOK) + ':configuration', 'exec'), namespace)
    return namespace, splits


class TrainingSelectionTests(unittest.TestCase):
    def test_seeded_sample_is_unique_repeatable_and_preserves_complete_training_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            first, splits = configure(destination)
            repeated, _ = configure(destination)
            changed_seed, _ = configure(destination / 'another-seed', seed=4821)
            selected = first['training_records']
            self.assertEqual(len(selected), 1000)
            self.assertEqual(len({r['id'] for r in selected}), 1000)
            self.assertEqual(selected, repeated['training_records'])
            self.assertNotEqual(selected, changed_seed['training_records'])
            self.assertNotEqual(selected, splits['train'][:1000])
            first['dataset'].records.assert_called_once_with('train')
            originals = {r['id']: r for r in splits['train']}
            for record in selected:
                self.assertIs(record, originals[record['id']])
            saved = json.loads((destination / 'settings.json').read_text())
            self.assertEqual(saved['training_ids'], [r['id'] for r in selected])

    def test_none_uses_all_training_pairs_and_invalid_counts_fail_before_saving_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            namespace, splits = configure(destination / 'all', count=None)
            self.assertEqual(namespace['training_records'], splits['train'])
            for count in (0, -1, True, 2.5, 2001):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'TRAINING_EXAMPLES'):
                    configure(destination / str(count), count=count)
                self.assertFalse((destination / str(count) / 'settings.json').exists())

    def test_changed_sample_requires_new_training_run_but_demo_count_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            original, _ = configure(destination, count=17)
            changed_demos, _ = configure(destination, count=17, demos=8)
            self.assertEqual(original['settings'], changed_demos['settings'])
            saved = (destination / 'settings.json').read_bytes()
            for values in ({'count': 18}, {'count': 17, 'seed': 4821}):
                with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'new RESULTS directory'):
                    configure(destination, **values)
                self.assertEqual((destination / 'settings.json').read_bytes(), saved)

    def test_tokenization_consumes_only_the_selected_training_pairs_and_separate_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            namespace, _ = configure(Path(temporary), count=17)
            tokenizer = SimpleNamespace(eos_token='EOS', apply_chat_template=Mock(
                side_effect=lambda messages, **kwargs: [11, 12] + ([13, 14] if len(messages) == 2 else [])))
            received_ids = []
            def user_text(record):
                received_ids.append(record['id'])
                return record['input']
            namespace.update({'AutoTokenizer': SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                              'Dataset': SimpleNamespace(from_list=lambda rows: rows), 'user_text': user_text})
            with contextlib.redirect_stdout(io.StringIO()):
                load_tag('tokenization', namespace)
            self.assertEqual(len(namespace['train_dataset']), 17)
            self.assertEqual(len(namespace['validation_dataset']), namespace['VALIDATION_EXAMPLES'])
            self.assertEqual({i for i in received_ids if i.startswith('train-')},
                             {r['id'] for r in namespace['training_records']})
            self.assertFalse(any(i.startswith('test-') for i in received_ids))
            self.assertEqual(namespace['train_dataset'][0]['labels'], [-100, -100, 13, 14])
            self.assertEqual(namespace['tokenized_training_ids'], namespace['settings']['training_ids'])


if __name__ == '__main__':
    unittest.main()
