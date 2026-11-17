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

    def test_sft_labels_correspond_to_verified_reconstruction(self):
        import pandas as pd
        records = load_jsonl(NOTEBOOK.parent / "generated_data/v1/train.jsonl")
        references = load_jsonl(NOTEBOOK.parent / "generated_data/v1/train.references.jsonl")
        namespace = load_tag("sft_construction", {
            "json": json, "pd": pd, "data": {"train": records},
            "metadata": {"train": references}, "display": lambda value: None})
        dataset = namespace["sft_dataset"]
        self.assertEqual(len(dataset), 10000)
        self.assertEqual(set(dataset.column_names), {"id", "input", "target"})
        preview = namespace["verified_preview"]
        self.assertEqual(len(preview), 5)
        targets = [json.loads(value) for value in preview["sft_json_label"]]
        self.assertEqual(sum(target["recoverable"] for target in targets), 3)
        self.assertTrue(all(value is True for value in preview.iloc[:3]["matches_original"]))
        self.assertTrue(all("OMIT complete window" in value for value in preview.iloc[3:]["deterministic_reconstruction"]))

    def test_fixed_demo_examples_never_use_targets_as_model_input(self):
        records = load_jsonl(NOTEBOOK.parent / "generated_data/v1/validation.jsonl")
        references = load_jsonl(NOTEBOOK.parent / "generated_data/v1/validation.references.jsonl")
        namespace = load_tag("scoring", {"json": json, "Counter": Counter})
        load_tag("tokenization", namespace)
        namespace.update({"data": {"validation": records}, "metadata": {"validation": references}})
        load_tag("demo_helpers", namespace)
        chosen = namespace["demo_records"]
        self.assertEqual(len({record["id"] for record in chosen}), 3)
        self.assertEqual({record["target"]["recoverable"] for record in chosen}, {True, False})
        inspection = namespace["choose_inspection_records"](records, chosen)
        self.assertEqual(inspection[:3], chosen)
        self.assertEqual(len({record["id"] for record in inspection}), 5)
        self.assertIn("normalize_numeric_format", inspection[3]["target"]["operations"])
        self.assertIn("deduplicate_records", inspection[4]["target"]["operations"])
        for record in chosen:
            changed = {**record, "id": "different-id", "target": {"recoverable": "SENTINEL", "operations": ["SENTINEL"]}}
            self.assertEqual(namespace["user_text"](record), namespace["user_text"](changed))
        with self.assertRaises(ValueError):
            namespace["show_model_examples"](chosen, ["only one prediction"])

        import pandas as pd
        displayed = []
        namespace.update({"pd": pd, "display": displayed.append, "render_prompt": lambda r: r["input"]})
        before = [{"id": record["id"], "prediction": "baseline " + record["id"]} for record in chosen]
        namespace["show_model_examples"](inspection, [json.dumps(r["target"]) for r in inspection], before)
        frame = displayed[0].data
        self.assertEqual(len(frame), 5)
        self.assertEqual(frame.iloc[0]["before_sft_output"], before[0]["prediction"])
        self.assertIn("additional after-SFT", frame.iloc[4]["before_sft_output"])


if __name__ == "__main__":
    unittest.main()
