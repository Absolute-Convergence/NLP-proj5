import csv
import io
from candleprep.schema import COLUMNS, RecoveryError, read_csv, write_csv, timestamp


def restore_column_layout(text):
    rows = read_csv(text)
    if len(rows[0]) != len(COLUMNS) or set(rows[0]) != set(COLUMNS):
        raise RecoveryError("Cannot identify the complete column layout")
    order = [rows[0].index(column) for column in COLUMNS]
    return write_csv([[row[index] for index in order] for row in rows])


def repair_cell_alignment(text):
    rows = read_csv(text)
    if tuple(rows[0]) != COLUMNS:
        raise RecoveryError("Expected the ordered schema")
    for row in rows[1:]:
        positions = []
        for index, value in enumerate(row):
            try:
                timestamp(value)
                positions.append(index)
            except RecoveryError:
                continue
        if len(positions) != 1:
            raise RecoveryError("Cannot identify a unique cyclic row rotation")
        index = positions[0]
        row[:] = row[index:] + row[:index]
    return write_csv(rows)


def repair_csv_structure(text):
    # Supported structural representation: a consistent semicolon or pipe
    # delimiter with the complete known header. Broken field boundaries fail.
    for delimiter in (",", ";", "|"):
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
        except csv.Error:
            continue
        if rows and tuple(rows[0]) == COLUMNS and all(len(row) == len(COLUMNS) for row in rows):
            return write_csv(rows)
    raise RecoveryError("No supported unambiguous CSV structure")
