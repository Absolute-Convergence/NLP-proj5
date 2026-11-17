import json
from collections import Counter
from pathlib import Path
from .datasets.serialization import write_json as save_json, write_jsonl as save_records

from candleprep.recovery.registry import REGISTRY

KNOWN_OPERATIONS = frozenset(REGISTRY)

def strict_json(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("Non-JSON constant: " + value)
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None

def score_prediction(prediction, reference):
    value = strict_json(prediction)
    decision = value.get("recoverable") if value is not None else None
    decision_valid = type(decision) is bool
    operations = value.get("operations") if value is not None else None
    operations_valid = type(operations) is list and all(type(name) is str and name in KNOWN_OPERATIONS for name in operations)
    valid = (value is not None and set(value) == {"recoverable", "operations"}
             and decision_valid and operations_valid and (decision or not operations))
    return {
        "recoverability_correct": decision_valid and decision == reference["recoverable"],
        "operation_sequence_correct": operations_valid and operations == reference["operations"],
        "full_answer_correct": bool(valid and value == reference),
        "format_valid": bool(valid),
        "predicted_class": str(decision).lower() if decision_valid else "invalid",
    }

def score_predictions(predictions, records):
    if len(predictions) != len(records) or not records:
        raise ValueError("Need one prediction per reference and a nonempty dataset")
    counts = Counter()
    confusion = {"false": {"false": 0, "true": 0, "invalid": 0},
                 "true": {"false": 0, "true": 0, "invalid": 0}}
    for prediction, record in zip(predictions, records):
        reference = record["target"]
        result = score_prediction(prediction, reference)
        counts["recoverability"] += int(result["recoverability_correct"])
        counts["full_answer"] += int(result["full_answer_correct"])
        counts["format"] += int(result["format_valid"])
        if reference["recoverable"]:
            counts["positive"] += 1
            counts["sequence"] += int(result["operation_sequence_correct"])
        confusion[str(reference["recoverable"]).lower()][result["predicted_class"]] += 1
    return {
        "examples": len(records), "oracle_positive_examples": counts["positive"],
        "recoverability_accuracy": counts["recoverability"] / len(records),
        "operation_sequence_accuracy": counts["sequence"] / counts["positive"] if counts["positive"] else None,
        "full_answer_accuracy": counts["full_answer"] / len(records),
        "format_validity": counts["format"] / len(records), "confusion_matrix": confusion,
    }

def score_breakdowns(predictions, records, references):
    if len(references) != len(records) or any(a["id"] != b["id"] for a, b in zip(records, references)):
        raise ValueError("Evaluation metadata must match record IDs")
    groups = {}
    for index, reference in enumerate(references):
        names = list(reference["categories"])
        names.append("single" if reference["composition_size"] == 1 else "composed")
        for name in names:
            groups.setdefault(name, []).append(index)
    output = {}
    for name, indices in groups.items():
        output[name] = score_predictions([predictions[i] for i in indices], [records[i] for i in indices])
    return output


def final_execution_demo(records, references, saved_predictions, destination):
    from candleprep.oracle.executor import execute
    from candleprep.oracle.verifier import equivalent
    from candleprep.schema import RecoveryError
    if destination.exists():
        raise FileExistsError("Demonstration output already exists")
    if not (len(records) == len(references) == len(saved_predictions)):
        raise ValueError("Demonstration records must align")
    counts = Counter()
    repaired = []
    omissions = []
    outcomes = []
    for record, reference, saved in zip(records, references, saved_predictions):
        if not (record["id"] == reference["id"] == saved["id"]):
            raise ValueError("Demonstration IDs do not match")
        expected_positive = record["target"]["recoverable"]
        counts["oracle_recoverable"] += int(expected_positive)
        result = score_prediction(saved["prediction"], record["target"])
        plan = strict_json(saved["prediction"])
        status = "invalid_plan"
        reconstructed = False
        if result["format_valid"]:
            if not plan["recoverable"]:
                status = "omitted"
                omissions.append(record["id"])
                counts["false_omissions"] += int(expected_positive)
            else:
                counts["attempted_repairs_of_oracle_negative"] += int(not expected_positive)
                try:
                    restored = execute(record["input"], plan["operations"])
                    reconstructed = equivalent(restored, reference["original"])
                    status = "reconstructed" if reconstructed else "incorrect_reconstruction"
                    repaired.append({"id": record["id"], "csv": restored})
                    counts["retained"] += 1
                    counts["exact_on_oracle_recoverable"] += int(reconstructed and expected_positive)
                except RecoveryError:
                    status = "execution_failure"
        counts[status] += 1
        outcomes.append({"id": record["id"], "status": status, "exact_reconstruction": reconstructed})
    summary = dict(counts)
    summary["examples"] = len(records)
    summary["retained_coverage"] = counts["retained"] / len(records)
    summary["exact_reconstruction_on_oracle_recoverable"] = (counts["exact_on_oracle_recoverable"] / counts["oracle_recoverable"] if counts["oracle_recoverable"] else None)
    destination.mkdir(parents=True)
    save_records(destination / "repaired_windows.jsonl", repaired)
    save_json(destination / "omitted_example_ids.json", omissions)
    save_records(destination / "outcomes.jsonl", outcomes)
    save_json(destination / "summary.json", summary)
    return summary
