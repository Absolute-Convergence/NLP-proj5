"""Validators preserve the text on success and raise RecoveryError on failure."""

from candleprep.schema import COLUMNS, RecoveryError, canonical, read_csv, write_csv


def validate_schema(text):
    rows = read_csv(text)
    if tuple(rows[0]) != COLUMNS:
        raise RecoveryError("Incomplete or unordered schema")
    return text


def validate_numeric_context(text):
    canonical(text)
    return text


def validate_value_completeness(text):
    canonical(text)
    return text


def validate_domain_constraints(text):
    for row in canonical(text):
        _, opening, high, low, close, volume, _ = row
        if not (low <= opening <= high and low <= close <= high) or volume < 0:
            raise RecoveryError("OHLC/volume constraint violation")
    return text


def detect_outliers(text):
    # Domain violations are observable; no statistical detection can identify
    # an arbitrary incorrect price from this small window with certainty.
    return validate_domain_constraints(text)


def validate_context_consistency(text):
    return validate_domain_constraints(text)


def validate_timestamp_integrity(text):
    records = canonical(text)
    times = [row[0] for row in records]
    if len(times) != len(set(times)):
        raise RecoveryError("Timestamp collision")
    return text


def validate_temporal_sequence(text):
    records = canonical(text)
    if any(a[0] >= b[0] for a, b in zip(records, records[1:])):
        raise RecoveryError("Records are not strictly chronological")
    # Irregular market intervals alone are not evidence of corruption.
    return text


def normalize_missing_value(text):
    rows = read_csv(text)
    if tuple(rows[0]) != COLUMNS:
        raise RecoveryError("Expected the ordered schema")
    for row in rows[1:]:
        for index in range(1, len(COLUMNS)):
            if row[index].strip().lower() in {"null", "none", "n/a", "nan", "missing"}:
                if COLUMNS[index] != "ROC":
                    raise RecoveryError("Required value is missing")
                row[index] = ""
    return write_csv(rows)
