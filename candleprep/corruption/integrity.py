from datetime import datetime, timedelta
from .syntactic import replace_character


def duplicate_row(rows, rng):
    index = rng.randrange(1, len(rows))
    rows.insert(index, list(rows[index]))
    return rows


def duplicate_with_conflict(rows, rng):
    index = rng.randrange(1, len(rows))
    duplicate = list(rows[index])
    duplicate[1] = replace_character(duplicate[1], rng)
    rows.insert(index, duplicate)
    return rows


def shuffle_rows(rows, rng):
    records = rows[1:]
    rng.shuffle(records)
    if records == rows[1:]:
        records.reverse()
    return [rows[0]] + records


def create_timestamp_collision(rows, rng):
    rows[2][0] = rows[1][0]
    return rows


def create_time_gap(rows, rng):
    del rows[rng.randrange(1, len(rows))]
    return rows


def break_temporal_sequence(rows, rng):
    return [rows[0]] + list(reversed(rows[1:]))
