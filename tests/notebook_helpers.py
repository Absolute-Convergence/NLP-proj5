import ast
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


def load_definitions(tag, namespace):
    """Test notebook functions without running tokenization/model execution."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if tag in c.get("metadata", {}).get("tags", []))
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace
