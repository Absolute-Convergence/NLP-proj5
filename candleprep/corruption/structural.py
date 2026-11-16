"""Functions transform a copied CSV matrix, never the source records."""


def swap_cell_values(rows, rng):
    row = rng.choice(rows[1:])
    first, second = rng.sample(range(1, 5), 2)
    row[first], row[second] = row[second], row[first]
    return rows


def swap_columns(rows, rng):
    # Headers move with values, making the mapping observable.
    first, second = rng.sample(range(len(rows[0])), 2)
    for row in rows:
        row[first], row[second] = row[second], row[first]
    return rows


def shift_row_cells(rows, rng):
    index = rng.randrange(1, len(rows))
    shift = rng.randrange(1, len(rows[index]))
    rows[index] = rows[index][-shift:] + rows[index][:-shift]
    return rows


def insert_delimiter(rows, rng):
    index = rng.randrange(1, len(rows))
    rows[index].insert(2, "")
    return rows


def remove_delimiter(rows, rng):
    index = rng.randrange(1, len(rows))
    rows[index][1:3] = [rows[index][1] + rows[index][2]]
    return rows


def split_field(rows, rng):
    index = rng.randrange(1, len(rows))
    value = rows[index][1]
    split = max(1, len(value)//2)
    rows[index][1:2] = [value[:split], value[split:]]
    return rows


def merge_fields(rows, rng):
    return remove_delimiter(rows, rng)
