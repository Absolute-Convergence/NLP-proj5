import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from notebook_helpers import load_tag
from candleprep.schema import RecoveryError
from test_corruption import window


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_tag("scoring", {"json": json, "Counter": Counter})

    def score(self, prediction, target=None):
        if target is None:
            target = {"recoverable": True, "operations": ["repair_number_spelling", "parse_number_words"]}
        return self.namespace["score_prediction"](prediction, target)

    def test_classification_credit_is_independent_of_plan(self):
        result = self.score('{"recoverable":true,"operations":["unknown"]}')
        self.assertTrue(result["recoverability_correct"])
        self.assertFalse(result["operation_sequence_correct"])
        self.assertFalse(result["format_valid"])

    def test_order_and_strict_contract(self):
        cases = [
            '{"recoverable":true,"operations":["parse_number_words","repair_number_spelling"]}',
            '{"recoverable":true,"operations":[],"extra":1}',
            '{"recoverable":"true","operations":[]}',
            '{"recoverable":1,"operations":[]}',
            '{"recoverable":true,"operations":[12]}',
            '{"recoverable":true,"operations":[],"recoverable":false}',
            '{"recoverable":true,"operations":NaN}',
            '```json\n{"recoverable":true,"operations":[]}\n```',
            'not JSON',
        ]
        for prediction in cases:
            with self.subTest(prediction=prediction):
                self.assertFalse(self.score(prediction)["full_answer_correct"])

    def test_negative_contract_and_key_order(self):
        target = {"recoverable": False, "operations": []}
        self.assertTrue(self.score('{"operations": [], "recoverable": false}', target)["full_answer_correct"])
        self.assertFalse(self.score('{"recoverable":false,"operations":["parse_number_words"]}', target)["format_valid"])

    def test_sequence_denominator_does_not_count_negatives(self):
        records = [{"target": {"recoverable": True, "operations": ["parse_number_words"]}},
                   {"target": {"recoverable": False, "operations": []}}]
        predictions = ['{"recoverable":true,"operations":[]}', '{"recoverable":false,"operations":[]}']
        with patch("candleprep.oracle.executor.execute", side_effect=AssertionError("No tools in scoring")):
            metrics = self.namespace["score_predictions"](predictions, records)
        self.assertEqual(metrics["recoverability_accuracy"], 1)
        self.assertEqual(metrics["operation_sequence_accuracy"], 0)
        self.assertEqual(metrics["full_answer_accuracy"], 0.5)

    def test_invalid_predictions_stay_in_denominator(self):
        records = [{"target": {"recoverable": True, "operations": []}}]
        metrics = self.namespace["score_predictions"](["bad"], records)
        self.assertEqual(metrics["recoverability_accuracy"], 0)
        self.assertEqual(metrics["confusion_matrix"]["true"]["invalid"], 1)

    def test_final_demo_measures_false_omissions_and_failures(self):
        namespace = dict(self.namespace)
        namespace.update({"RUN_FINAL_DEMO": False})
        def save_json(path, value):
            Path(path).write_text(json.dumps(value), encoding="utf-8")
        def save_records(path, value):
            Path(path).write_text("\n".join(json.dumps(item) for item in value), encoding="utf-8")
        namespace.update({"save_json": save_json, "save_records": save_records})
        namespace = load_tag("final_demo", namespace)
        records = []
        references = []
        predictions = []
        plans = ["bad", '{"recoverable":false,"operations":[]}',
                 '{"recoverable":true,"operations":[]}',
                 '{"recoverable":true,"operations":["parse_number_words"]}']
        for index, prediction in enumerate(plans):
            text = window() if index != 3 else window().replace("653.54", "unsupported text", 1)
            records.append({"id": str(index), "input": text, "target": {"recoverable": True, "operations": []}})
            references.append({"id": str(index), "original": window()})
            predictions.append({"id": str(index), "prediction": prediction})
        with tempfile.TemporaryDirectory() as temporary:
            summary = namespace["final_execution_demo"](records, references, predictions, Path(temporary) / "demo")
        self.assertEqual(summary["invalid_plan"], 1)
        self.assertEqual(summary["false_omissions"], 1)
        self.assertEqual(summary["execution_failure"], 1)
        self.assertEqual(summary["exact_reconstruction_on_oracle_recoverable"], 0.25)
        self.assertEqual(summary["retained_coverage"], 0.25)


if __name__ == "__main__":
    unittest.main()
