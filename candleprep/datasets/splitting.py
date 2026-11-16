from candleprep.schema import RecoveryError, timestamp


def split_source(rows, train_fraction=0.7, validation_fraction=0.1):
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Invalid source split fractions")
    times = [timestamp(row[0]) for row in rows]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise RecoveryError("Source must have unique chronological timestamps")
    train_end = int(len(rows) * train_fraction)
    validation_end = int(len(rows) * (train_fraction + validation_fraction))
    return {
        "train": (0, rows[:train_end]),
        "validation": (train_end, rows[train_end:validation_end]),
        "test": (validation_end, rows[validation_end:]),
    }


def source_windows(rows, offset, window_size=4):
    if window_size < 2:
        raise ValueError("At least two records are needed for integrity operators")
    windows = []
    for start in range(0, len(rows)-window_size+1, window_size):
        windows.append((list(range(offset+start, offset+start+window_size)), rows[start:start+window_size]))
    return windows
