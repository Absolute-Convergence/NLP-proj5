from candleprep.schema import RecoveryError, canonical


def equivalent(repaired, original):
    expected = canonical(original)  # Invalid references are programming/data errors.
    try:
        return canonical(repaired) == expected
    except RecoveryError:
        return False
