import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from candleprep import CandleDataset
from candleprep.source import download_source
from candleprep.task import user_text

SOURCE = Path(__file__).resolve().parents[1] / "TVC_GOLD, 360.csv"


class SourceTests(unittest.TestCase):
    def test_download_is_verified_cached_and_does_not_need_a_root_csv(self):
        content = SOURCE.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "cache/source.csv"
            with patch('candleprep.source.urlopen', return_value=io.BytesIO(content)) as request:
                download_source(target)
                self.assertEqual(target.read_bytes(), content)
                download_source(target)
                request.assert_called_once()
            target.write_bytes(b'broken')
            with patch('candleprep.source.urlopen', side_effect=AssertionError('Do not replace cache')):
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    download_source(target)
            self.assertEqual(target.read_bytes(), b'broken')

    def test_failed_or_invalid_response_does_not_cache_a_partial_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'source.csv'
            with patch('candleprep.source.urlopen', side_effect=URLError('offline')):
                with self.assertRaisesRegex(RuntimeError, 'internet'):
                    download_source(target)
            self.assertFalse(target.exists())
            content = b'<html>not a CSV</html>'
            with patch('candleprep.source.urlopen', return_value=io.BytesIO(content)):
                with self.assertRaises(ValueError):
                    download_source(target, sha256=hashlib.sha256(content).hexdigest())
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temporary).iterdir()), [])


class DatasetAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / 'data'
        cls.kwargs = dict(source=SOURCE, output_dir=cls.output, train_size=90, validation_size=65, test_size=70, seed=12)
        cls.dataset = CandleDataset.prepare(**cls.kwargs)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_reuse_requires_matching_settings_and_unmodified_artifacts(self):
        with patch('candleprep.dataset.generate_dataset', side_effect=AssertionError('Must reuse')):
            self.assertEqual(CandleDataset.prepare(**self.kwargs).summary(), {'train': 90, 'validation': 65, 'test': 70})
        with self.assertRaisesRegex(ValueError, 'settings/source'):
            CandleDataset.prepare(**{**self.kwargs, 'seed': 13})
        path = self.output / 'train.jsonl'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                CandleDataset.prepare(**self.kwargs)
        finally:
            path.write_bytes(original)

    def test_public_pairs_previews_and_frozen_examples_keep_the_oracle_separate(self):
        records = self.dataset.records()
        self.assertTrue(all(set(r) == {'id', 'input', 'target'} for r in records))
        for preview in (self.dataset.original_samples(), self.dataset.synthetic_samples()):
            self.assertEqual(len(preview), 15)
            self.assertEqual([sum(r['sample'] == label for r in preview) for label in ('First', 'Middle', 'Last')], [5, 5, 5])
        labels = self.dataset.label_samples()
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(r['matches_original'] is True for r in labels[:3]))
        self.assertTrue(all('OMIT complete window' in r['deterministic_reconstruction'] for r in labels[3:]))
        chosen = self.dataset.examples()
        self.assertEqual(len({r['id'] for r in chosen}), 5)
        self.assertEqual({r['target']['recoverable'] for r in chosen}, {True, False})
        for record in chosen:
            self.assertEqual(user_text(record), user_text({**record, 'id': 'changed',
                'target': {'recoverable': 'SENTINEL'}, 'original': 'SENTINEL', 'history': 'SENTINEL'}))

    def test_scoring_uses_saved_labels_without_executing_tools(self):
        records = self.dataset.records('test')
        predictions = [json.dumps(r['target']) for r in records]
        with patch('candleprep.oracle.executor.execute', side_effect=AssertionError('No execution in scoring')):
            report = self.dataset.evaluate(predictions)
        self.assertEqual(report['full_answer_accuracy'], 1)
        self.assertEqual(report['format_validity'], 1)
        self.assertTrue(report['breakdowns'])

    def test_display_remains_optional_for_the_core_import(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys, candleprep; assert not any(n in sys.modules for n in ("torch","pandas","transformers","IPython")); print(candleprep.__version__)'],
            check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '0.2.0')


if __name__ == '__main__':
    unittest.main()
