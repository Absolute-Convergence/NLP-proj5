from candleprep.schema import RecoveryError, read_csv, write_csv, timestamp


def deduplicate_records(text):
    rows = read_csv(text)
    index = rows[0].index("time") if "time" in rows[0] else -1
    if index < 0:
        raise RecoveryError("No timestamp column")
    seen = {}
    output = [rows[0]]
    for row in rows[1:]:
        identity = timestamp(row[index])
        if identity in seen:
            if row != seen[identity]:
                raise RecoveryError("Conflicting duplicates have no supported tie-break")
        else:
            seen[identity] = row
            output.append(row)
    return write_csv(output)


def resolve_duplicate_conflicts(text):
    # No majority-vote or answer-key shortcut: conflicts cannot be resolved.
    return deduplicate_records(text)
