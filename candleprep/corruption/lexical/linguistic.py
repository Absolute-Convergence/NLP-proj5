def change_conjunction(value, rng):
    return value.replace("hundred ", "hundred and ", 1)


def change_hyphenation(value, rng):
    return value.replace("-", " ")


def change_case(value, rng):
    return value.upper()


def change_spacing(value, rng):
    return value.replace(" ", "  ")


def change_currency_phrasing(value, rng):
    return value + " US dollars"


def change_decimal_phrasing(value, rng):
    return value.replace("point", "decimal")
