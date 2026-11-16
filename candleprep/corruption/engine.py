"""Seeded, copied inputs and an explicit catalog of supported operators."""

import random
from candleprep.schema import COLUMNS, CorruptedExample, CorruptionEvent, RecoveryError, read_csv, write_csv
from . import syntactic, formatting, structural, semantic, completeness, integrity
from .lexical import representation, spelling, linguistic

# name -> (category, function, scope, recovery recipe, information lost)
# Recovery recipes are hints for the privileged oracle, never model features.
OPERATORS = {}


def _register(category, module, names, scope="cell", recovery=(), lost=False):
    for name in names:
        OPERATORS[name] = (category, getattr(module, name), scope, recovery, lost)


_register("syntactic", syntactic, ("replace_character", "delete_character", "transpose_characters"), lost=True)
_register("syntactic", syntactic, ("insert_character", "insert_invalid_symbol"), recovery=("remove_extraneous_characters",))
_register("syntactic", syntactic, ("duplicate_decimal",), recovery=("repair_numeric_punctuation",))
_register("formatting", formatting, ("add_currency_symbol", "add_unit_suffix", "change_decimal_separator", "change_numeric_notation"), recovery=("normalize_numeric_format",))
_register("formatting", formatting, ("change_date_format",), scope="time", recovery=("normalize_timestamp",))
_register("formatting", formatting, ("add_whitespace",), recovery=("normalize_whitespace",))
_register("structural", structural, ("swap_cell_values", "insert_delimiter", "remove_delimiter", "split_field", "merge_fields"), scope="table", lost=True)
_register("structural", structural, ("swap_columns",), scope="table", recovery=("restore_column_layout",))
_register("structural", structural, ("shift_row_cells",), scope="table", recovery=("repair_cell_alignment",))
_register("semantic", semantic, ("violate_domain_constraint", "introduce_outlier", "change_sign", "create_context_conflict"), lost=True)
_register("semantic", semantic, ("change_unit_scale",), recovery=("normalize_unit_scale",))
_register("completeness", completeness, ("delete_cell_value", "replace_with_null", "replace_with_missing_token", "truncate_value"), lost=True)
_register("completeness", completeness, ("remove_required_column",), scope="table", lost=True)
_register("integrity", integrity, ("duplicate_row",), scope="table", recovery=("deduplicate_records",))
_register("integrity", integrity, ("duplicate_with_conflict", "create_timestamp_collision", "create_time_gap"), scope="table", lost=True)
_register("integrity", integrity, ("shuffle_rows", "break_temporal_sequence"), scope="table", recovery=("sort_chronologically",))
_register("lexical", representation, ("number_to_words",), recovery=("parse_number_words",))
_register("lexical", representation, ("number_to_currency_words",), recovery=("parse_currency_words",))
_register("lexical", representation, ("change_number_expression", "change_decimal_expression"), recovery=("normalize_number_expression", "parse_number_words"))
_register("lexical", representation, ("mix_digits_and_words",), recovery=("parse_mixed_number_expression",))
_register("lexical", representation, ("destroy_fraction",), lost=True)
_register("lexical", spelling, ("substitute_letter", "insert_letter", "delete_letter", "transpose_letters", "repeat_letter", "common_misspelling"), recovery=("repair_number_spelling",))
_register("lexical", linguistic, ("change_conjunction", "change_hyphenation", "change_case", "change_spacing"), recovery=("parse_number_words",))
_register("lexical", linguistic, ("change_currency_phrasing",), recovery=("parse_currency_words",))
_register("lexical", linguistic, ("change_decimal_phrasing",), recovery=("normalize_number_expression", "parse_number_words"))


def corrupt(text, operators, seed=0, row=None, column=None):
    """Compose operations at one sampled cell unless they act on the table.

    Row numbers are zero-based data rows, excluding the header. Provenance
    records the actual row/column and before/after value for every event.
    """
    rng = random.Random(seed)
    example = CorruptedExample(text, text)
    initial = read_csv(text)
    selected_row = row if row is not None else rng.randrange(len(initial)-1)
    available_columns = [name for index, name in enumerate(COLUMNS[1:], start=1)
                         if initial[selected_row+1][index].strip()]
    selected_column = column if column is not None else rng.choice(available_columns)
    for name in operators:
        if name not in OPERATORS:
            raise ValueError("Unknown corruption: " + name)
        category, function, scope, recovery, lost = OPERATORS[name]
        rows = read_csv(example.corrupted)
        if scope == "table":
            before = example.corrupted
            rows = function(rows, rng)
            after = write_csv(rows)
            event_row, event_column = None, None
        else:
            event_row = selected_row
            event_column = "time" if scope == "time" else selected_column
            if event_column not in rows[0] or event_row + 1 >= len(rows):
                raise RecoveryError("Composition removed the target cell")
            index = rows[0].index(event_column)
            before = rows[event_row+1][index]
            if not before:
                raise RecoveryError("Scalar corruption requires a present value")
            after = function(before, rng)
            rows[event_row+1][index] = after
        if before == after:
            raise RecoveryError("Operator made no observable change: " + name)
        example.corrupted = write_csv(rows)
        example.history.append(CorruptionEvent(category, name, event_row, event_column,
                                               before, after, {"seed": seed}, recovery, lost))
    return example
