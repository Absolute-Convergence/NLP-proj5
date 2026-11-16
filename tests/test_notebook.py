import ast
import json
import unittest
from collections import Counter
from pathlib import Path
from notebook_helpers import load_tag, NOTEBOOK
from candleprep.datasets import load_jsonl


class NotebookTests(unittest.TestCase):
    def test_explanations_and_syntax(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["nbformat_minor"], 5)
        for index, cell in enumerate(notebook["cells"]):
            self.assertIn("id", cell)
            if cell["cell_type"] == "code":
                self.assertEqual(notebook["cells"][index-1]["cell_type"], "markdown")
                ast.parse("".join(cell["source"]))

    def test_actual_cached_tokenizer_masks_targets_and_preserves_budget(self):
        from transformers import AutoTokenizer, DataCollatorForSeq2Seq
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        namespace = load_tag("scoring", {"json": json, "Counter": Counter})
        namespace["MAX_LENGTH"] = 1536
        namespace = load_tag("tokenization", namespace)
        records = load_jsonl(NOTEBOOK.parent / "generated_data/v1/train.jsonl")
        longest = 0
        features = []
        # Include every operator and composition in the first catalog cycles,
        # plus all source-column shapes from the full corpus.
        for record in records:
            feature = namespace["tokenize_record"](record, tokenizer)
            prompt = namespace["prompt_tokens"](tokenizer, record)
            self.assertEqual(feature["labels"][:len(prompt)], [-100] * len(prompt))
            self.assertEqual(feature["labels"][len(prompt):], feature["input_ids"][len(prompt):])
            self.assertGreater(len(feature["input_ids"]), len(prompt))
            self.assertLessEqual(len(prompt) + 128, 1536)
            longest = max(longest, len(feature["input_ids"]))
            if len(features) < 2:
                features.append(feature)
        collated = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100)(features)
        for row, feature in enumerate(features):
            self.assertTrue(all(int(value) == -100 for value in collated["labels"][row, len(feature["input_ids"]):]))
        print("Longest training sequence including full JSON:", longest)


if __name__ == "__main__":
    unittest.main()
