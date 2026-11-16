import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "proj5.ipynb"


def load_tag(tag, namespace=None):
    namespace = namespace if namespace is not None else {"json": json}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if tag in cell.get("metadata", {}).get("tags", []):
            source = cell["source"]
            exec(compile("".join(source), str(NOTEBOOK) + ":" + tag, "exec"), namespace)
            return namespace
    raise ValueError("Missing notebook tag: " + tag)
