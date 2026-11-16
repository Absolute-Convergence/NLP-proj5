from datetime import datetime
from decimal import Decimal


def add_currency_symbol(value, rng):
    return "$" + value


def add_unit_suffix(value, rng):
    return value + " USD"


def change_decimal_separator(value, rng):
    return value.replace(".", ",", 1)


def change_date_format(value, rng):
    parsed = datetime.fromisoformat(value)
    return parsed.strftime("%d/%m/%Y %H:%M:%S%z")


def change_numeric_notation(value, rng):
    return format(Decimal(value), "E")


def add_whitespace(value, rng):
    return "  " + value + " \t"
