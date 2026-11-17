"""Convenient, CPU-only preparation and inspection of verified training pairs."""
import csv
import hashlib
import json
from pathlib import Path

from .datasets import generate_dataset, load_jsonl
from .oracle.executor import execute
from .oracle.verifier import equivalent
from .source import SOURCE_SHA256, SOURCE_URL, download_source


class CandleDataset:
    """Keep model records separate from the oracle's originals and history."""

    @classmethod
    def prepare(cls, source=None, output_dir="generated_data/v1", *, seed=4820,
                train_size=10000, validation_size=1000, test_size=2000, window_size=4,
                source_url=SOURCE_URL, source_sha256=SOURCE_SHA256):
        """Download/load the source, then generate or verify the cached dataset.

        No model is loaded and no responses are generated. A local source may
        still be supplied explicitly; it never depends on the working directory's CSV.
        """
        output = Path(output_dir)
        source = (Path(source) if source is not None else
                  download_source(output / "source.csv", source_url, source_sha256))
        sizes = {"train": train_size, "validation": validation_size, "test": test_size}
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            generate_dataset(source, output, train_size, validation_size, test_size, seed, window_size)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["source"]["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
                or manifest["seed"] != seed or manifest["window_size"] != window_size
                or any(manifest["splits"][split]["examples"] != size for split, size in sizes.items())):
            raise ValueError("Cached dataset settings/source differ; choose a new output_dir")
        for filename, expected in manifest["artifact_sha256"].items():
            if Path(filename).name != filename:
                raise ValueError("Invalid artifact filename in manifest")
            if hashlib.sha256((output / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Cached dataset checksum mismatch: " + filename)
        return cls(source, output, manifest)

    def __init__(self, source, output_dir, manifest):
        self.source = Path(source)
        self.output_dir = Path(output_dir)
        self.manifest = manifest
        self._records = {}
        self._references = {}
        for split in ("train", "validation", "test"):
            self._records[split] = load_jsonl(self.output_dir / (split + ".jsonl"))
            references = load_jsonl(self.output_dir / (split + ".references.jsonl"))
            if len(references) != len(self._records[split]) or any(
                    a["id"] != b["id"] for a, b in zip(self._records[split], references)):
                raise ValueError("Oracle references do not match records: " + split)
            self._references[split] = references

    def records(self, split="train"):
        """Return only public id/input/target records; no oracle metadata."""
        return self._records[split]

    def summary(self):
        return {split: details["examples"] for split, details in self.manifest["splits"].items()}

    @staticmethod
    def _samples(records, rows):
        if type(rows) is not int or rows < 1:
            raise ValueError("rows must be a positive integer")
        result = []
        for label, start in (("First", 0), ("Middle", max(0, len(records) // 2 - rows // 2)),
                             ("Last", max(0, len(records) - rows))):
            result.extend({"sample": label, "row": index + 1, **records[index]}
                          for index in range(start, min(start + rows, len(records))))
        return result

    def original_samples(self, rows=5):
        with self.source.open(encoding="utf-8-sig", newline="") as stream:
            original = list(csv.DictReader(stream))
        return self._samples(original, rows)

    def synthetic_samples(self, rows=5):
        records = []
        for split, values in self._records.items():
            for record in values:
                records.append({"split": split, "id": record["id"], "input": record["input"],
                                "sft_json_label": json.dumps(record["target"])})
        return self._samples(records, rows)

    def examples(self, count=5, split="validation"):
        """Freeze representative cases without supplying labels to the model."""
        records = self.records(split)
        if type(count) is not int or not 1 <= count <= len(records):
            raise ValueError("count must fit the selected split")
        chosen = []
        references = {r["id"]: r for r in self._references[split]}
        selectors = (
            lambda r: r["target"]["recoverable"] and "repair_number_spelling" in r["target"]["operations"],
            lambda r: r["target"]["recoverable"] and len(r["target"]["operations"]) > 1
                      and len(references[r["id"]]["categories"]) > 1,
            lambda r: not r["target"]["recoverable"] and any(
                event["operator"] == "remove_required_column" for event in references[r["id"]]["history"]),
            lambda r: "normalize_numeric_format" in r["target"]["operations"],
            lambda r: "deduplicate_records" in r["target"]["operations"],
        )
        for select in selectors:
            record = next((r for r in records if r not in chosen and select(r)), None)
            if record is not None and len(chosen) < count:
                chosen.append(record)
        for record in records:
            if len(chosen) == count:
                break
            if record not in chosen:
                chosen.append(record)
        return chosen

    def label_samples(self, rows=5, split="train"):
        """Execute verified labels for display; these are not model features."""
        records = self.records(split)
        references = {r["id"]: r for r in self._references[split]}
        positives = [r for r in records if r["target"]["recoverable"]]
        negatives = [r for r in records if not r["target"]["recoverable"]]
        if type(rows) is not int or rows < 1:
            raise ValueError("rows must be a positive integer")
        selected = positives[:(rows + 1) // 2] + negatives[:rows // 2]
        for record in records:
            if len(selected) >= min(rows, len(records)):
                break
            if record not in selected:
                selected.append(record)
        preview = []
        for record in selected:
            target = record["target"]
            original = references[record["id"]]["original"]
            restored = execute(record["input"], target["operations"]) if target["recoverable"] else None
            verified = equivalent(restored, original) if restored is not None else None
            if target["recoverable"] and not verified:
                raise AssertionError("Label failed reconstruction: " + record["id"])
            preview.append({"id": record["id"], "corrupted_input": record["input"],
                            "sft_json_label": json.dumps(target),
                            "deterministic_reconstruction": restored if restored is not None else "OMIT complete window; no reconstruction",
                            "oracle_original": original, "matches_original": verified})
        return preview

    def evaluate(self, predictions, split="test"):
        from .evaluation import score_breakdowns, score_predictions
        records = self.records(split)
        report = score_predictions(predictions, records)
        report["breakdowns"] = score_breakdowns(predictions, records, self._references[split])
        return report

    def execute_predictions(self, predictions, destination, split="test"):
        from .evaluation import final_execution_demo
        saved = [{"id": record["id"], "prediction": prediction}
                 for record, prediction in zip(self.records(split), predictions)]
        if len(predictions) != len(self.records(split)):
            raise ValueError("Need one prediction per record")
        return final_execution_demo(self.records(split), self._references[split], saved, Path(destination))

    def display_samples(self, rows=5):
        from .display import display_samples
        display_samples(self, rows)

    def display_labels(self, rows=5):
        from .display import display_table
        display_table(self.label_samples(rows))

    def display_coverage(self):
        from .display import display_coverage
        display_coverage(self)

    def display_predictions(self, records, predictions, before=None):
        from .display import display_predictions
        return display_predictions(records, predictions, before)
