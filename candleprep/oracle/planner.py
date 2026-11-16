"""Short, bounded candidate recipes. The oracle can inspect all provenance."""

from itertools import combinations
from candleprep.schema import RecoveryError
from .executor import execute
from .verifier import equivalent

# Payload/representation dependencies for compositions whose reverse history
# is not an executable repair order. Only present operations are considered.
DEPENDENCY_ORDER = (
    "repair_csv_structure", "restore_column_layout", "repair_cell_alignment",
    "normalize_whitespace", "normalize_timestamp", "normalize_numeric_format",
    "remove_extraneous_characters", "repair_numeric_punctuation",
    "normalize_number_expression", "repair_number_spelling", "normalize_unit_scale",
    "parse_currency_words", "parse_mixed_number_expression", "parse_number_words",
    "normalize_missing_value", "sort_chronologically", "deduplicate_records",
)


def find_plan(example):
    if any(event.information_lost for event in example.history):
        return None
    recipe = []
    for event in reversed(example.history):
        for name in event.recovery:
            if name not in recipe:
                recipe.append(name)
    if len(recipe) > 10:
        raise ValueError("Candidate recipe exceeds the bounded planner policy")
    dependency_recipe = sorted(recipe, key=lambda name: (DEPENDENCY_ORDER.index(name)
                               if name in DEPENDENCY_ORDER else len(DEPENDENCY_ORDER), name))
    # Two explicit orderings, with ordered subsequences checked shortest first.
    # Equal-length verified plans use lexical operation-name order as tie-break.
    for size in range(len(recipe) + 1):
        candidates = set(combinations(recipe, size)) | set(combinations(dependency_recipe, size))
        for candidate in sorted(candidates):
            try:
                repaired = execute(example.corrupted, candidate)
            except RecoveryError:
                continue
            if equivalent(repaired, example.original):
                return list(candidate)
    return None
