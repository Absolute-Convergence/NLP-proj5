from decimal import Decimal, localcontext


def violate_domain_constraint(value, rng):
    return "-" + value.lstrip("-+")


def introduce_outlier(value, rng):
    with localcontext() as context:
        context.prec = max(50, len(value) + 10)
        return format(Decimal(value) * 1000, "f")


def change_sign(value, rng):
    return value[1:] if value.startswith("-") else "-" + value


def change_unit_scale(value, rng):
    with localcontext() as context:
        context.prec = max(50, len(value) + 10)
        return format(Decimal(value) * 100, "f") + " cents"


def create_context_conflict(value, rng):
    return introduce_outlier(value, rng)
