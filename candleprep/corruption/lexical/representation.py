from candleprep.recovery.lexical import SMALL, number_words


def number_to_words(value, rng):
    return number_words(value)


def number_to_currency_words(value, rng):
    return number_words(value) + " US dollars"


def change_number_expression(value, rng):
    return number_words(value).replace("point", "decimal")


def mix_digits_and_words(value, rng):
    words = number_words(value)
    for index, word in enumerate(SMALL[:10]):
        if word in words.split():
            return " ".join(str(index) if token == word else token for token in words.split())
    return words


def change_decimal_expression(value, rng):
    return change_number_expression(value, rng)


def destroy_fraction(value, rng):
    # An intentional information-losing lexical example.
    return number_words(value).split(" point ")[0]
