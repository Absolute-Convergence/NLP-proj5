import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from candleprep.datasets import generate_dataset, load_jsonl
from candleprep.oracle.executor import execute
from candleprep.oracle.verifier import equivalent

SOURCE = Path(__file__).resolve().parents[1] / "TVC_GOLD, 360.csv"


class DatasetTests(unittest.TestCase):
    def test_reproducibility_leakage_and_positive_reconstruction(self):
        original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            a = generate_dataset(SOURCE, first, 90, 65, 70, seed=12)
            b = generate_dataset(SOURCE, second, 90, 65, 70, seed=12)
            self.assertEqual(a, b)
            self.assertEqual(a, json.loads((first / "manifest.json").read_text(encoding="utf-8")))
            seen_sources = {}
            seen_inputs = set()
            all_categories = set()
            for split in ("train", "validation", "test"):
                self.assertEqual((first / (split+".jsonl")).read_bytes(), (second / (split+".jsonl")).read_bytes())
                public = load_jsonl(first / (split+".jsonl"))
                refs = load_jsonl(first / (split+".references.jsonl"))
                self.assertEqual(len(public), len(refs))
                classes = set()
                for record, reference in zip(public, refs):
                    self.assertEqual(set(record), {"id", "input", "target"})
                    self.assertEqual(record["id"], reference["id"])
                    self.assertNotIn(record["input"], seen_inputs)
                    seen_inputs.add(record["input"])
                    classes.add(record["target"]["recoverable"])
                    all_categories.update(reference["categories"])
                    for index in reference["source_indices"]:
                        self.assertIn(seen_sources.get(index, split), (split,))
                        seen_sources[index] = split
                    if record["target"]["recoverable"]:
                        self.assertTrue(equivalent(execute(record["input"], record["target"]["operations"]), reference["original"]))
                self.assertEqual(classes, {True, False})
            self.assertEqual(len(all_categories), 7)
            with self.assertRaises(FileExistsError):
                generate_dataset(SOURCE, first, 1, 1, 1)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), original_hash)


if __name__ == "__main__":
    unittest.main()
