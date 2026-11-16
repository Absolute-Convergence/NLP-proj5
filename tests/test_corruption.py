import random
import unittest
from candleprep.corruption.engine import corrupt
from candleprep.corruption.lexical import spelling
from candleprep.oracle.labeling import label
from candleprep.schema import COLUMNS, write_csv


def window():
    rows = [list(COLUMNS)]
    for index in range(4):
        rows.append([f"2026-01-0{index+1}T00:00:00-06:00", "653.54", "660", "640", "650", "100", ""])
    return write_csv(rows)


class CorruptionTests(unittest.TestCase):
    def test_reproducible_copied_provenance(self):
        original = window()
        a = corrupt(original, ["number_to_words", "common_misspelling"], seed=13)
        b = corrupt(original, ["number_to_words", "common_misspelling"], seed=13)
        self.assertEqual(a, b)
        self.assertEqual(a.original, original)
        self.assertNotEqual(a.corrupted, original)
        self.assertEqual(len(a.history), 2)
        self.assertEqual(label(a)["operations"], ["repair_number_spelling", "parse_number_words"])

    def test_all_categories_have_real_positive_or_negative_cases(self):
        cases = {
            "syntactic": ("insert_invalid_symbol", True),
            "formatting": ("add_currency_symbol", True),
            "structural": ("swap_columns", True),
            "semantic": ("change_unit_scale", True),
            "completeness": ("delete_cell_value", False),
            "integrity": ("duplicate_row", True),
            "lexical": ("number_to_words", True),
        }
        for category, (operator, recoverable) in cases.items():
            with self.subTest(category=category):
                example = corrupt(window(), [operator], seed=5, column="open")
                self.assertEqual(example.history[0].category, category)
                self.assertEqual(label(example)["recoverable"], recoverable)

    def test_order_sensitive_compositions(self):
        for operators in (["insert_invalid_symbol", "add_currency_symbol"],
                          ["duplicate_row", "shuffle_rows"],
                          ["swap_columns", "number_to_words"],
                          ["number_to_words", "common_misspelling", "add_whitespace"]):
            with self.subTest(operators=operators):
                result = label(corrupt(window(), operators, seed=8, column="open"))
                self.assertTrue(result["recoverable"])

    def test_negative_component_omits_complete_window(self):
        example = corrupt(window(), ["delete_character", "insert_invalid_symbol"], column="open")
        self.assertEqual(label(example), {"recoverable": False, "operations": []})


if __name__ == "__main__":
    unittest.main()
