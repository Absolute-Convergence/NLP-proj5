"""CPU generation, source-first splitting, verified labels, and private sidecars."""

import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from candleprep import __version__
from candleprep.corruption.engine import OPERATORS, corrupt
from candleprep.oracle.labeling import label
from candleprep.oracle.provenance import reference_record
from candleprep.recovery.registry import REGISTRY
from candleprep.schema import COLUMNS, RecoveryError, canonical, valid_label, write_csv
from .serialization import write_json, write_jsonl
from .splitting import source_windows, split_source


def recipes():
    result = []
    for name, (_, _, _, _, _) in OPERATORS.items():
        category = OPERATORS[name][0]
        if category != "lexical" or name in {
            "number_to_words", "number_to_currency_words", "change_number_expression",
            "mix_digits_and_words", "change_decimal_expression", "destroy_fraction",
        }:
            result.append((name,))
    for name in ("substitute_letter", "insert_letter", "delete_letter", "transpose_letters",
                 "repeat_letter", "common_misspelling", "change_conjunction", "change_hyphenation",
                 "change_case", "change_spacing", "change_currency_phrasing", "change_decimal_phrasing"):
        result.append(("number_to_words", name))
    result.extend([
        ("number_to_words", "common_misspelling", "add_whitespace"),
        ("insert_invalid_symbol", "add_currency_symbol"),
        ("duplicate_row", "shuffle_rows"),
        ("delete_character", "insert_invalid_symbol"),
        ("change_date_format", "add_currency_symbol"),
        ("swap_columns", "number_to_words"),
    ])
    return result


def audit_source(path):
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    text = write_csv(rows)
    records = canonical(text)
    if len(records) < 40:
        raise ValueError("Source is too small for three window partitions")
    missing = Counter()
    for row in rows[1:]:
        for name, value in zip(COLUMNS, row):
            if not value.strip():
                missing[name] += 1
    violations = 0
    for row in records:
        _, opening, high, low, close, volume, _ = row
        if not (low <= opening <= high and low <= close <= high) or volume < 0:
            violations += 1
    if violations:
        raise ValueError("Baseline contains domain violations; review it before corruption")
    intervals = Counter(str(b[0]-a[0]) for a, b in zip(records, records[1:]))
    audit = {
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(records), "columns": list(COLUMNS), "missing": dict(missing),
        "duplicate_rows": len(records)-len(set(records)),
        "duplicate_timestamps": len(records)-len(set(row[0] for row in records)),
        "domain_violations": violations,
        "first_timestamp_utc": records[0][0].isoformat(),
        "last_timestamp_utc": records[-1][0].isoformat(),
        "common_intervals": [list(item) for item in intervals.most_common(8)],
    }
    return rows[1:], audit


def generate_dataset(source, output_dir="generated_data", train_size=10000,
                     validation_size=1000, test_size=2000, seed=4820, window_size=4):
    """Generate model JSONL and oracle-only sidecars with a fixed seed.

    Source windows never overlap. Variants of a window stay in its partition.
    Existing files in the target directory are not overwritten.
    """
    sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    if any(type(size) is not int or size < 1 for size in sizes.values()):
        raise ValueError("Every requested split size must be a positive integer")
    output = Path(output_dir)
    paths = [output / "manifest.json"]
    for split in sizes:
        paths.extend([output / (split + ".jsonl"), output / (split + ".references.jsonl")])
    if any(path.exists() for path in paths):
        raise FileExistsError("Dataset output already exists; choose a new output directory")
    rows, audit = audit_source(source)
    partitions = split_source(rows)
    catalog = recipes()
    rng = random.Random(seed)
    seen_inputs = {}
    all_artifacts = {}
    manifest = {
        "candleprep_version": __version__, "seed": seed, "window_size": window_size,
        "source": audit, "required_columns": list(COLUMNS[:-1]), "optional_columns": ["ROC"],
        "canonical_policy": "ordered complete rows; exact Decimal values; UTC instants; preserved ROC blanks",
        "source_split_fractions": [0.7, 0.1, 0.2], "splits": {},
        "operators": list(OPERATORS), "recovery_operations": list(REGISTRY),
        "recipes": [list(recipe) for recipe in catalog], "maximum_composition": max(len(recipe) for recipe in catalog),
    }
    for split, size in sizes.items():
        offset, source_rows = partitions[split]
        windows = source_windows(source_rows, offset, window_size)
        if not windows:
            raise ValueError("A source partition has no complete windows")
        public = []
        references = []
        categories = Counter()
        classes = Counter()
        operators = Counter()
        plans = Counter()
        compositions = Counter()
        rejected = Counter()
        attempts = 0
        # Cycling recipes makes operator coverage intentional. A failed/no-op
        # recipe is retried on a different window, never silently relabeled.
        while len(public) < size:
            if attempts >= size * 40:
                raise RuntimeError("Generation exhausted its retry budget")
            recipe = catalog[attempts % len(catalog)]
            source_indices, records = windows[rng.randrange(len(windows))]
            original = write_csv([list(COLUMNS)] + records)
            corruption_seed = rng.randrange(2**32)
            attempts += 1
            try:
                example = corrupt(original, recipe, seed=corruption_seed)
            except RecoveryError:
                rejected["unsupported_or_no_change"] += 1
                continue
            target = label(example)
            if not valid_label(target, REGISTRY):
                raise AssertionError("Oracle emitted an invalid target")
            fingerprint = hashlib.sha256(example.corrupted.encode("utf-8")).hexdigest()
            if fingerprint in seen_inputs:
                if seen_inputs[fingerprint] != target:
                    rejected["conflicting_visible_input"] += 1
                else:
                    rejected["duplicate_visible_input"] += 1
                continue
            seen_inputs[fingerprint] = target
            identifier = f"{split}-{len(public)+1:06d}"
            public.append({"id": identifier, "input": example.corrupted, "target": target})
            reference = reference_record(example)
            reference.update({"id": identifier, "source_indices": source_indices,
                              "categories": sorted(set(event.category for event in example.history)),
                              "composition_size": len(example.history)})
            references.append(reference)
            classes[str(target["recoverable"]).lower()] += 1
            categories.update(reference["categories"])
            operators.update(event.operator for event in example.history)
            plans[" -> ".join(target["operations"]) if target["recoverable"] else "OMIT"] += 1
            compositions[str(len(example.history))] += 1
        all_artifacts[split] = (public, references)
        manifest["splits"][split] = {
            "examples": size, "source_start": offset, "source_end_exclusive": offset+len(source_rows),
            "available_windows": len(windows), "used_windows": len(set(tuple(ref["source_indices"]) for ref in references)),
            "classes": dict(classes), "categories": dict(categories), "operators": dict(operators),
            "plans": dict(plans), "composition_sizes": dict(compositions),
            "attempts": attempts, "rejections": dict(rejected),
        }
    # Generation and verification finish before any final artifact is written.
    output.mkdir(parents=True, exist_ok=True)
    for split, (public, references) in all_artifacts.items():
        write_jsonl(output / (split + ".jsonl"), public)
        write_jsonl(output / (split + ".references.jsonl"), references)
    manifest["artifact_sha256"] = {}
    for path in paths[1:]:
        manifest["artifact_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)
    return manifest
