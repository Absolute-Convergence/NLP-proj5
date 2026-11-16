import re
from candleprep.recovery.lexical import MISSPELLINGS, VOCABULARY


def _mutate(value, rng, kind):
    matches = list(re.finditer(r"[A-Za-z]{3,}", value))
    if not matches:
        return value
    match = rng.choice(matches)
    word = match.group()
    index = rng.randrange(len(word))
    if kind == "substitute":
        changed = word[:index] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[index+1:]
    elif kind == "insert":
        changed = word[:index] + rng.choice("abcdefghijklmnopqrstuvwxyz") + word[index:]
    elif kind == "delete":
        changed = word[:index] + word[index+1:]
    elif kind == "repeat":
        changed = word[:index] + word[index] + word[index:]
    else:
        index = min(index, len(word)-2)
        changed = word[:index] + word[index+1] + word[index] + word[index+2:]
    if changed.lower() in VOCABULARY:
        return value
    return value[:match.start()] + changed + value[match.end():]


def substitute_letter(value, rng):
    return _mutate(value, rng, "substitute")


def insert_letter(value, rng):
    return _mutate(value, rng, "insert")


def delete_letter(value, rng):
    return _mutate(value, rng, "delete")


def transpose_letters(value, rng):
    return _mutate(value, rng, "transpose")


def repeat_letter(value, rng):
    return _mutate(value, rng, "repeat")


def common_misspelling(value, rng):
    choices = [(right, wrong) for wrong, right in MISSPELLINGS.items() if right in value]
    if not choices:
        return substitute_letter(value, rng)
    right, wrong = rng.choice(choices)
    return value.replace(right, wrong, 1)
