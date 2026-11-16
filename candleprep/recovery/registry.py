"""Fixed public operation names; no dynamic evaluation or plugins."""

from .lexical import (repair_number_spelling, parse_number_words, parse_currency_words,
                      normalize_number_expression, parse_mixed_number_expression)
from .numeric import (normalize_numeric_format, normalize_whitespace,
                      repair_numeric_punctuation, remove_extraneous_characters,
                      normalize_unit_scale)
from .temporal import normalize_timestamp, sort_chronologically
from .structural import restore_column_layout, repair_cell_alignment, repair_csv_structure
from .integrity import deduplicate_records, resolve_duplicate_conflicts
from .validation import (validate_schema, validate_numeric_context, validate_value_completeness,
                         validate_domain_constraints, detect_outliers, validate_context_consistency,
                         validate_timestamp_integrity, validate_temporal_sequence,
                         normalize_missing_value)

REGISTRY = {function.__name__: function for function in (
    repair_number_spelling, parse_number_words, parse_currency_words,
    normalize_number_expression, parse_mixed_number_expression,
    normalize_numeric_format, normalize_whitespace, repair_numeric_punctuation,
    remove_extraneous_characters, normalize_unit_scale, normalize_timestamp,
    sort_chronologically, restore_column_layout, repair_cell_alignment, repair_csv_structure,
    deduplicate_records, resolve_duplicate_conflicts, validate_schema, validate_numeric_context,
    validate_value_completeness, validate_domain_constraints, detect_outliers,
    validate_context_consistency, validate_timestamp_integrity, validate_temporal_sequence,
    normalize_missing_value,
)}
