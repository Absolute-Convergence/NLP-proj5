"""Shared CSV and label contracts. This module has no ML dependencies."""

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

COLUMNS = ("time", "open", "high", "low", "close", "Volume", "ROC")
REQUIRED = COLUMNS[:-1]
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


class RecoveryError(ValueError):
    """The observed example is outside a tool's supported recovery rules."""


def read_csv(text):
    try:
        rows = list(csv.reader(io.StringIO(text), strict=True))
    except csv.Error as exc:
        raise RecoveryError("Malformed CSV") from exc
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise RecoveryError("CSV must contain a header and rectangular records")
    return rows


def write_csv(rows):
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def decimal_value(value):
    if not NUMBER.fullmatch(value.strip()):
        raise RecoveryError("Not a supported numeric literal: " + value)
    try:
        result = Decimal(value.strip())
    except InvalidOperation as exc:
        raise RecoveryError("Invalid decimal") from exc
    if not result.is_finite():
        raise RecoveryError("Non-finite number")
    return result


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise RecoveryError("Timestamp must be ISO 8601") from exc
    if result.tzinfo is None:
        raise RecoveryError("Timestamp requires an explicit UTC offset")
    return result.astimezone(timezone.utc)


def canonical(text):
    rows = read_csv(text)
    if tuple(rows[0]) != COLUMNS:
        raise RecoveryError("Expected the complete ordered candle schema")
    result = []
    for row in rows[1:]:
        values = [timestamp(row[0])]
        for column, value in zip(COLUMNS[1:], row[1:]):
            if column == "ROC" and not value.strip():
                values.append(None)
            else:
                values.append(decimal_value(value))
        result.append(tuple(values))
    return tuple(result)


def map_numeric(text, transform):
    rows = read_csv(text)
    if tuple(rows[0]) != COLUMNS:
        raise RecoveryError("Normalize the column layout first")
    for row in rows[1:]:
        for index in range(1, len(COLUMNS)):
            if row[index].strip():
                row[index] = transform(row[index])
    return write_csv(rows)


def valid_label(value, known_operations=None):
    if not isinstance(value, dict) or set(value) != {"recoverable", "operations"}:
        return False
    if type(value["recoverable"]) is not bool or type(value["operations"]) is not list:
        return False
    for name in value["operations"]:
        if type(name) is not str:
            return False
        if known_operations is not None and name not in known_operations:
            return False
    return value["recoverable"] or not value["operations"]


@dataclass
class CorruptionEvent:
    category: str
    operator: str
    row: int | None
    column: str | None
    before: str
    after: str
    parameters: dict = field(default_factory=dict)
    recovery: tuple = ()
    information_lost: bool = False


@dataclass
class CorruptedExample:
    original: str
    corrupted: str
    history: list = field(default_factory=list)
