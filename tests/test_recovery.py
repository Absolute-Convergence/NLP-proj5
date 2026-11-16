import unittest
from candleprep.schema import COLUMNS, RecoveryError, canonical, write_csv
from candleprep.recovery.registry import REGISTRY
from candleprep.recovery.lexical import number_words, parse_words, repair_number_spelling
from candleprep.corruption.engine import corrupt
from candleprep.oracle.executor import execute
from candleprep.oracle.labeling import label
from test_corruption import window


class RecoveryTests(unittest.TestCase):
    def test_exact_number_round_trips(self):
        from decimal import Decimal
        for value in ("0", "-2.3147798020025516", "653.54", "637483", "1000001", "100.00000000000000001"):
            with self.subTest(value=value):
                self.assertEqual(Decimal(parse_words(number_words(value))), Decimal(value))

    def test_ambiguous_and_unsupported_words_fail(self):
        for words in ("one two", "hundred", "six point fifty four", "one and", "one million million"):
            with self.subTest(words=words), self.assertRaises(RecoveryError):
                parse_words(words)

    def test_public_tools_have_no_reference_argument(self):
        import inspect
        for name, function in REGISTRY.items():
            self.assertEqual(list(inspect.signature(function).parameters), ["text"], name)

    def test_supported_format_and_structure_repairs(self):
        for operator in ("duplicate_decimal", "change_decimal_separator", "change_date_format",
                         "change_numeric_notation", "add_whitespace", "shift_row_cells",
                         "break_temporal_sequence", "change_number_expression",
                         "number_to_currency_words", "mix_digits_and_words"):
            with self.subTest(operator=operator):
                example = corrupt(window(), [operator], seed=2, column="open")
                result = label(example)
                self.assertTrue(result["recoverable"], result)
                self.assertEqual(canonical(execute(example.corrupted, result["operations"])), canonical(example.original))

    def test_validators_preserve_payload(self):
        for name in ("validate_schema", "validate_numeric_context", "validate_domain_constraints",
                     "detect_outliers", "validate_temporal_sequence", "validate_timestamp_integrity"):
            self.assertEqual(REGISTRY[name](window()), window())

    def test_meaningful_signs_and_exponents_are_preserved(self):
        rows = [list(COLUMNS), ["2026-01-01T00:00:00+00:00", "1e2", "110", "90", "100", "10", "-2.5"]]
        original = write_csv(rows)
        for operation in ("normalize_numeric_format", "remove_extraneous_characters", "repair_numeric_punctuation"):
            self.assertEqual(canonical(REGISTRY[operation](original)), canonical(original))

    def test_conflicting_duplicates_are_not_arbitrarily_resolved(self):
        example = corrupt(window(), ["duplicate_with_conflict"], seed=4)
        with self.assertRaises(RecoveryError):
            REGISTRY["resolve_duplicate_conflicts"](example.corrupted)

    def test_irregular_market_intervals_are_valid(self):
        self.assertEqual(REGISTRY["validate_temporal_sequence"](window()), window())

    def test_optional_missingness_and_unknown_dispatch(self):
        original = window()
        marked = original.replace(',100,\n', ',100,N/A\n', 1)
        self.assertEqual(canonical(REGISTRY["normalize_missing_value"](marked)), canonical(original))
        with self.assertRaises(RecoveryError):
            execute(original, ["invented_operation"])

    def test_supported_csv_dialect_preserves_quoted_values(self):
        import csv
        import io
        from candleprep.schema import read_csv
        output = io.StringIO()
        csv.writer(output, delimiter=";", lineterminator="\n").writerows(read_csv(window()))
        self.assertEqual(canonical(REGISTRY["repair_csv_structure"](output.getvalue())), canonical(window()))


if __name__ == "__main__":
    unittest.main()
