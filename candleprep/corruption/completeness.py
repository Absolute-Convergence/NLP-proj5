def delete_cell_value(value, rng):
    return ""


def replace_with_null(value, rng):
    return "null"


def replace_with_missing_token(value, rng):
    return rng.choice(("N/A", "MISSING", "None"))


def truncate_value(value, rng):
    return value[:-1]


def remove_required_column(rows, rng):
    index = rng.randrange(0, len(rows[0])-1)
    for row in rows:
        del row[index]
    return rows
