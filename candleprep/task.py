from .recovery.registry import REGISTRY

KNOWN_OPERATIONS = frozenset(REGISTRY)

TASK = """Given a corrupted candle CSV window, predict the canonical shortest repair plan under the supported tools.
Return only JSON with exactly two keys: recoverable (a Boolean) and operations (an ordered list of tool names).
Use {"recoverable":false,"operations":[]} when exact recovery of the complete window is unsupported. The complete window is omitted in that case.
Schema: time,open,high,low,close,Volume,ROC. All fields are required except ROC, which may be blank. Negative ROC is valid.
Timestamps require an explicit UTC offset. Irregular market intervals are allowed. Preserve all rows and exact numeric meaning.
Supported conventions: English cardinal words and individual fractional digits; minus for negatives; point or normalized decimal;
USD currency words; currency prefix $; suffixes USD/contracts/percent; decimal comma without thousands grouping;
cents explicitly identify a scale of 100; declared removable noise is ? @ ~; doubled decimal punctuation;
alternate date DD/MM/YYYY HH:MM:SS followed by numeric UTC offset; column headers move with column values;
cyclic row rotations identifiable by time; exact timestamp-identical duplicates; chronological row sorting.
Arbitrary digit/sign changes, lost decimals/digits, missing required values, conflicting duplicates, and missing rows are unsupported.
Operation names: """ + ", ".join(sorted(KNOWN_OPERATIONS)) + """.
For equally short valid plans, choose the lexicographically first ordered operation list.
Do not include unnecessary validator calls. Valid data needing no transforms uses recoverable true and an empty operations list.
Corrupted CSV:
"""


def user_text(record):
    """Model-visible text contains only the static task and corrupted input."""
    return TASK + record["input"]
