from .planner import find_plan
from candleprep.schema import canonical


def label(example):
    canonical(example.original)  # A broken reference is never a negative example.
    plan = find_plan(example)
    return {"recoverable": plan is not None, "operations": plan if plan is not None else []}
