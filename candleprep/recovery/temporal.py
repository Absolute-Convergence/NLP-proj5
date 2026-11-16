from datetime import datetime
from candleprep.schema import RecoveryError, read_csv, write_csv, timestamp


def normalize_timestamp(text):
    rows = read_csv(text)
    index = rows[0].index("time") if "time" in rows[0] else -1
    if index < 0:
        raise RecoveryError("No timestamp column")
    for row in rows[1:]:
        try:
            timestamp(row[index])
        except RecoveryError:
            try:
                parsed = datetime.strptime(row[index], "%d/%m/%Y %H:%M:%S%z")
            except ValueError as exc:
                raise RecoveryError("Unsupported timestamp representation") from exc
            row[index] = parsed.isoformat()
    return write_csv(rows)


def sort_chronologically(text):
    rows = read_csv(text)
    if "time" not in rows[0]:
        raise RecoveryError("No timestamp column")
    index = rows[0].index("time")
    return write_csv([rows[0]] + sorted(rows[1:], key=lambda row: timestamp(row[index])))
