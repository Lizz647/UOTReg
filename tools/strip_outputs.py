#!/usr/bin/env python3
"""Clear every notebook's outputs + execution counts, leaving sources untouched.

    python tools/strip_outputs.py

Run before committing: the repository ships its notebooks output-free (outputs re-appear whenever
a notebook is run interactively, and they can embed machine-local absolute paths).
"""
import glob
import json
import os

HERE = os.path.abspath(os.path.dirname(__file__))
RELEASE = os.path.dirname(HERE)

n_files = n_cells = 0
for nb in (sorted(glob.glob(os.path.join(RELEASE, "notebooks", "*", "*.ipynb")))
           + sorted(glob.glob(os.path.join(RELEASE, "tutorials", "*.ipynb")))):
    d = json.load(open(nb, encoding="utf-8"))
    src_before = ["".join(c["source"]) for c in d["cells"]]
    changed = False
    for c in d["cells"]:
        if c["cell_type"] == "code" and (c.get("outputs") or c.get("execution_count") is not None):
            c["outputs"], c["execution_count"] = [], None
            changed = True
            n_cells += 1
    if not changed:
        continue
    assert ["".join(c["source"]) for c in d["cells"]] == src_before   # sources must never move
    with open(nb, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_files += 1
    print(f"  stripped {os.path.relpath(nb, RELEASE)}")
print(f"{n_files} notebooks stripped ({n_cells} cells); the rest were already clean")
