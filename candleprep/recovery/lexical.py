"""Small explicit English number grammar with exact fractional digits."""

import re
from decimal import Decimal
from candleprep.schema import NUMBER, RecoveryError, decimal_value, map_numeric

SMALL = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
VALUES = dict(enumerate(SMALL))
WORD_VALUES = {word: index for index, word in enumerate(SMALL)}
for _index in range(2, 10):
    WORD_VALUES[TENS[_index]] = _index * 10
SCALES = {"thousand": 1000, "million": 1000000, "billion": 1000000000,
          "trillion": 1000000000000}
MISSPELLINGS = {"hundrad": "hundred", "hundrd": "hundred", "fify": "fifty",
                "fourty": "forty", "thousnad": "thousand", "ninty": "ninety",
                "zerro": "zero", "seveen": "seven", "eigth": "eight"}
VOCABULARY = set(WORD_VALUES) | set(SCALES) | {"hundred", "point", "minus", "and"}


def integer_words(number):
    if number < 20:
        return SMALL[number]
    if number < 100:
        tens, remainder = divmod(number, 10)
        return TENS[tens] + ("-" + SMALL[remainder] if remainder else "")
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        return SMALL[hundreds] + " hundred" + (" " + integer_words(remainder) if remainder else "")
    for word, scale in reversed(tuple(SCALES.items())):
        if number >= scale:
            head, remainder = divmod(number, scale)
            return integer_words(head) + " " + word + (" " + integer_words(remainder) if remainder else "")
    raise RecoveryError("Integer is outside the supported grammar")


def number_words(value):
    number = decimal_value(value)
    if number.copy_abs() >= Decimal("1e15"):
        raise RecoveryError("Number is outside the supported grammar")
    literal = format(number.copy_abs(), "f")
    integer, dot, fraction = literal.partition(".")
    result = ("minus " if number < 0 else "") + integer_words(int(integer))
    if dot:
        result += " point " + " ".join(SMALL[int(digit)] for digit in fraction)
    return result


def _integer(tokens, mixed=False):
    total = 0
    group = 0
    previous_scale = 10**18
    last_kind = None
    seen_hundred = False
    for token in tokens:
        if token == "and":
            if last_kind is None or last_kind == "and":
                raise RecoveryError("Misplaced conjunction")
            last_kind = "and"
            continue
        if mixed and token.isdigit():
            value = int(token)
        elif token in WORD_VALUES:
            value = WORD_VALUES[token]
        else:
            value = None
        if value is not None:
            if last_kind == "number" and not (20 <= group % 100 <= 90 and group % 10 == 0 and 0 < value < 10):
                raise RecoveryError("Ambiguous adjacent number words")
            group += value
            last_kind = "number"
        elif token == "hundred":
            if seen_hundred or last_kind != "number" or not 1 <= group <= 9:
                raise RecoveryError("Unsupported hundred expression")
            group *= 100
            seen_hundred = True
            last_kind = "hundred"
        elif token in SCALES:
            scale = SCALES[token]
            if not group or scale >= previous_scale:
                raise RecoveryError("Unsupported scale order")
            total += group * scale
            group = 0
            previous_scale = scale
            seen_hundred = False
            last_kind = "scale"
        else:
            raise RecoveryError("Unknown number word: " + token)
    if not tokens or last_kind == "and":
        raise RecoveryError("Incomplete number expression")
    return total + group


def parse_words(value, mixed=False):
    words = value.lower().replace("-", " ").split()
    negative = bool(words and words[0] == "minus")
    if negative:
        words = words[1:]
    if words.count("point") > 1:
        raise RecoveryError("Multiple decimal points")
    if "point" in words:
        index = words.index("point")
        integer = _integer(words[:index], mixed)
        fraction = ""
        for word in words[index+1:]:
            if mixed and word.isdigit():
                fraction += word
            elif word in SMALL[:10]:
                fraction += str(WORD_VALUES[word])
            else:
                raise RecoveryError("Fraction requires individual digits")
        if not fraction:
            raise RecoveryError("Missing fractional digits")
        literal = str(integer) + "." + fraction
    else:
        literal = str(_integer(words, mixed))
    return ("-" if negative else "") + literal


def _repair_word(word):
    lower = word.lower()
    if lower in VOCABULARY or lower.isdigit():
        return word
    if lower in MISSPELLINGS:
        return MISSPELLINGS[lower]
    # Accept only a unique one-edit match; ambiguity fails explicitly.
    candidates = []
    for target in sorted(VOCABULARY):
        if abs(len(lower) - len(target)) > 1:
            continue
        if len(lower) == len(target):
            differences = [i for i in range(len(lower)) if lower[i] != target[i]]
            match = len(differences) == 1
            if len(differences) == 2:
                a, b = differences
                match = b == a + 1 and lower[a] == target[b] and lower[b] == target[a]
        elif len(lower) < len(target):
            match = any(target[:i] + target[i+1:] == lower for i in range(len(target)))
        else:
            match = any(lower[:i] + lower[i+1:] == target for i in range(len(lower)))
        if match:
            candidates.append(target)
    if len(candidates) != 1:
        raise RecoveryError("Unsupported or ambiguous spelling: " + word)
    return candidates[0]


def repair_number_spelling(text):
    def repair(value):
        if NUMBER.fullmatch(value.strip()):
            return value
        return re.sub(r"[A-Za-z]+", lambda match: _repair_word(match.group()), value)
    return map_numeric(text, repair)


def parse_number_words(text):
    return map_numeric(text, lambda value: value if NUMBER.fullmatch(value.strip()) else parse_words(value))


def parse_currency_words(text):
    def parse(value):
        if value.lower().endswith(" us dollars"):
            return parse_words(value[:-11])
        return value
    return map_numeric(text, parse)


def normalize_number_expression(text):
    def normalize(value):
        if NUMBER.fullmatch(value.strip()):
            return value
        value = re.sub(r"\bdecimal\b", "point", value, flags=re.IGNORECASE)
        return " ".join(value.lower().split())
    return map_numeric(text, normalize)


def parse_mixed_number_expression(text):
    return map_numeric(text, lambda value: value if NUMBER.fullmatch(value.strip()) else parse_words(value, mixed=True))
