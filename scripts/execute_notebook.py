"""Execute and validate notebook outputs in the selected Python environment."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/commerce-mpl")
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/tmp/commerce-jupyter")
path = ROOT / "notebooks/delivery_intelligence.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print("Notebook executed successfully, with no error outputs.")
