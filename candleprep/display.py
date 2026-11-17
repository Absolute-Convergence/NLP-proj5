"""Optional notebook presentation; pandas/IPython are imported only when used."""
import json


def display_table(records):
    import pandas as pd
    from IPython.display import display
    frame = pd.DataFrame(records)
    display(frame.style.format(escape="html").hide(axis="index").set_properties(
        **{"white-space": "pre-wrap", "font-family": "monospace", "text-align": "left", "min-width": "160px"}
    ).set_table_attributes('style="display:block;overflow-x:auto"'))
    return frame


def display_samples(dataset, rows):
    from IPython.display import HTML, display
    for title, records in (("Original data", dataset.original_samples(rows)),
                           ("Corrupted synthetic dataset", dataset.synthetic_samples(rows))):
        for group in ("First", "Middle", "Last"):
            display(HTML(f"<h3>{title}: {group.lower()} {rows} rows</h3>"))
            display_table([r for r in records if r["sample"] == group])


def display_coverage(dataset):
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display
    coverage = pd.DataFrame({s: d["categories"] for s, d in dataset.manifest["splits"].items()}).fillna(0).astype(int)
    display(coverage)
    coverage.plot.bar(figsize=(10, 4), title="Corruption category coverage")
    plt.ylabel("Examples containing category")
    plt.tight_layout()
    plt.show()


def display_predictions(records, predictions, before=None):
    from .evaluation import score_prediction
    from .task import user_text
    if len(records) != len(predictions) or (before is not None and len(before) != len(records)):
        raise ValueError("Need one prediction per displayed record")
    rows = []
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        row = {"id": record["id"], "model_received_input": user_text(record)}
        if before is not None:
            row["before_sft_output"] = before[index]
        row["after_sft_output" if before is not None else "model_output"] = prediction
        row["expected_json"] = json.dumps(record["target"])
        row["full_answer_correct"] = score_prediction(prediction, record["target"])["full_answer_correct"]
        rows.append(row)
    return display_table(rows)
