import unittest
from candleprep.schema import CorruptedExample, CorruptionEvent, write_csv
from candleprep.oracle.labeling import label
from candleprep.oracle.executor import execute
from candleprep.oracle.verifier import equivalent


def sample(value="653.54"):
    return write_csv([
        ["time", "open", "high", "low", "close", "Volume", "ROC"],
        ["2026-01-01T00:00:00-06:00", value, "660", "640", "650", "100", ""],
    ])


class OracleTests(unittest.TestCase):
    def test_spelling_then_parsing(self):
        original = sample()
        corrupted = sample("six hundrad fify-three point five four")
        event = CorruptionEvent("lexical", "misspell", 0, "open", "653.54", "words",
                                recovery=("repair_number_spelling", "parse_number_words"))
        example = CorruptedExample(original, corrupted, [event])
        result = label(example)
        self.assertEqual(result, {"recoverable": True, "operations": ["repair_number_spelling", "parse_number_words"]})
        self.assertTrue(equivalent(execute(corrupted, result["operations"]), original))

    def test_lost_decimal_omits_whole_example(self):
        event = CorruptionEvent("syntactic", "delete_character", 0, "open", "653.54", "65354", information_lost=True)
        self.assertEqual(label(CorruptedExample(sample(), sample("65354"), [event])),
                         {"recoverable": False, "operations": []})

    def test_no_unnecessary_operations(self):
        self.assertEqual(label(CorruptedExample(sample(), sample())), {"recoverable": True, "operations": []})

    def test_programming_errors_are_not_negative_labels(self):
        from unittest.mock import patch
        with patch("candleprep.oracle.planner.execute", side_effect=TypeError("bug")):
            with self.assertRaises(TypeError):
                label(CorruptedExample(sample(), sample("bad")))

    def test_invalid_reference_is_not_an_omission_label(self):
        from candleprep.schema import RecoveryError
        event = CorruptionEvent("completeness", "delete_cell_value", 0, "open", "bad", "", information_lost=True)
        with self.assertRaises(RecoveryError):
            label(CorruptedExample(sample("bad"), sample(""), [event]))


if __name__ == "__main__":
    unittest.main()
