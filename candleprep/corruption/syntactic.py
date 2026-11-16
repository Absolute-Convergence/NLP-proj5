def replace_character(value, rng):
    positions = [i for i, char in enumerate(value) if char.isdigit()]
    index = rng.choice(positions)
    digit = str((int(value[index]) + rng.randint(1, 9)) % 10)
    return value[:index] + digit + value[index+1:]


def insert_character(value, rng):
    index = rng.randrange(len(value) + 1)
    return value[:index] + rng.choice("?@~") + value[index:]


def insert_invalid_symbol(value, rng):
    return insert_character(value, rng)


def delete_character(value, rng):
    index = value.index(".") if "." in value else rng.randrange(len(value))
    return value[:index] + value[index+1:]


def transpose_characters(value, rng):
    positions = [i for i in range(len(value)-1) if value[i] != value[i+1]]
    if not positions:
        return value
    index = rng.choice(positions)
    return value[:index] + value[index+1] + value[index] + value[index+2:]


def duplicate_decimal(value, rng):
    return value.replace(".", "..", 1) if "." in value else value
