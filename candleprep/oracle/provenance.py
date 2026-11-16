from dataclasses import asdict


def reference_record(example):
    return {"original": example.original, "history": [asdict(event) for event in example.history]}
