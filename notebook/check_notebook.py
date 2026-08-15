"""Static check: does any cell use a name before an earlier cell defines it?

The laboratory is executed top to bottom by people who cannot debug it, and several
cells are skippable by design. A name introduced in a skippable cell and used in a
required one is therefore a defect, not a style question. Deleting a section is how it
happens: this file exists because deleting one did.

Usage:  python3 check_notebook.py
"""

import ast
import builtins
import json
import sys
from pathlib import Path

NB = Path(__file__).with_name("AIDA_Risk_Lab.ipynb")
KNOWN = set(dir(builtins)) | {"get_ipython", "display", "files"}


def bound_by(tree):
    """Every name this cell binds: assignment, import, def, class, for, with, except."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.arg):
            out.add(n.arg)                      # parameters are bound inside the body
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.comprehension):
            for t in ast.walk(n.target):
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def used_by(tree):
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def main():
    nb = json.loads(NB.read_text())
    defined, problems = set(KNOWN), []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        # IPython magics are not Python; drop them before parsing
        src = "\n".join("" if l.lstrip().startswith(("!", "%")) else l
                        for l in src.splitlines())
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, f"does not parse: {e}"))
            continue
        bound = bound_by(tree)          # a name bound later in the same cell is fine
        for name in sorted(used_by(tree) - defined - bound):
            problems.append((i, f"uses {name!r} before any cell defines it"))
        defined |= bound

    for i, msg in problems:
        print(f"  cell {i:3d}  {msg}")
    print(f"\n{len(problems)} problems across {len(nb['cells'])} cells")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
