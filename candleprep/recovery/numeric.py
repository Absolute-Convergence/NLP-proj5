import re
from decimal import Decimal
from candleprep.schema import NUMBER, RecoveryError, map_numeric


def normalize_numeric_format(text):
    def normalize(value):
        value = value.strip()
        if value.startswith("$"):
            value = value[1:]
        for suffix in (" USD", " contracts", " percent"):
            if value.endswith(suffix):
                value = value[:-len(suffix)]
                break
        # Declared locale: a single comma is always a decimal separator;
        # thousands-group separators are deliberately unsupported.
        if re.fullmatch(r"[+-]?\d+,\d+", value):
            value = value.replace(",", ".")
        return value
    return map_numeric(text, normalize)


def normalize_whitespace(text):
    from candleprep.schema import read_csv, write_csv
    rows = read_csv(text)
    return write_csv([[value.strip() for value in row] for row in rows])


def repair_numeric_punctuation(text):
    def repair(value):
        if re.fullmatch(r"[+-]?\d+\.\.\d+", value):
            return value.replace("..", ".")
        return value
    return map_numeric(text, repair)


def remove_extraneous_characters(text):
    # Only the explicitly declared noise alphabet is removable.
    return map_numeric(text, lambda value: value.translate(str.maketrans("", "", "?@~")))


def normalize_unit_scale(text):
    def normalize(value):
        if value.endswith(" cents"):
            literal = value[:-6]
            if not NUMBER.fullmatch(literal):
                raise RecoveryError("Unsupported cents literal")
            # Decimal.scaleb is exact within a sufficiently large local context.
            from decimal import localcontext
            with localcontext() as context:
                context.prec = max(50, len(literal) + 10)
                return format(Decimal(literal).scaleb(-2), "f")
        return value
    return map_numeric(text, normalize)
