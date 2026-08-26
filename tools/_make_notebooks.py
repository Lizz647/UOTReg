#!/usr/bin/env python3
"""Turn the working experiment scripts into release notebooks with plain-literal parameters.

The working copies under `newcode/` are configured by environment variable, which is right for a
cluster launcher and wrong for a notebook a reader opens. This rewrites every
`os.environ.get("NAME", default)` into the literal `default`, so the parameter block reads as
assignments, and normalises the fast path to a single `SMOKE = 1` at the top of each file.

    SMOKE = 1   small/fast, runs end to end on a laptop -- NOT the paper's numbers
    SMOKE = 0   the settings used in the paper

Run from the `UOTReg_new` folder. Idempotent: a file with no `os.environ` left is untouched.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Parameters a wrapper file sets before `exec`ing its shared partner (e.g. `*_sf.py`). These become
# `globals().get(...)` rather than a bare literal, so the wrapper can still override them -- the
# exec shares one namespace, so a name bound before the exec wins.
OVERRIDABLE = {"DATASET", "DIM", "SEED", "SMOKE", "VIZ_SEED", "CLUST_SEED", "N_PER_TIME",
               "N_BG", "SKIP_PENDING", "CLUST_ALL", "SAVE", "FS", "DEVICE"}

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _literal(cast, default):
    """The literal a transformed assignment should carry."""
    d = default.strip()
    if d.startswith(("'", '"')):
        inner = d[1:-1]
        if cast in ("int", "float"):
            return inner
        return f'"{inner}"'
    return d                                   # already an expression, e.g. str(C.SEEDS[0])


def transform(text, fname):
    out, n = text, 0

    def sub(pattern, repl):
        nonlocal out, n
        out, k = re.subn(pattern, repl, out)
        n += k

    # 1. boolean flags:  X = os.environ.get("E", "0") != "0"   /  == "1"
    def _boolrepl(m):
        name, default, op = m.group("name"), m.group("d"), m.group("op")
        on = (default != "0") if op == "!=" else (default == "1")
        lit = "1" if on else "0"
        if name.strip() in OVERRIDABLE:
            return f'{name}= globals().get("{name.strip()}", {lit})'
        return f"{name}= {lit}"
    sub(r'(?P<name>\w+\s*)=\s*os\.environ\.get\(\s*"[^"]+"\s*,\s*"(?P<d>[^"]*)"\s*\)\s*'
        r'(?P<op>!=|==)\s*"[^"]*"', _boolrepl)

    # 2. int(...) / float(...) wrappers
    def _castrepl(m):
        name, cast, lit = m.group("name"), m.group("cast"), _literal(m.group("cast"), m.group("d"))
        if name.strip() in OVERRIDABLE:
            return f'{name}= globals().get("{name.strip()}", {lit})'
        return f"{name}= {lit}"
    sub(r'(?P<name>\w+\s*)=\s*(?P<cast>int|float)\(\s*os\.environ\.get\(\s*"[^"]+"\s*,\s*'
        r'(?P<d>"[^"]*"|' + _NUM + r'|[^)]+?)\s*\)\s*\)', _castrepl)

    # 3. plain string assignment
    def _strrepl(m):
        name, lit = m.group("name"), _literal("str", m.group("d"))
        if name.strip() in OVERRIDABLE:
            return f'{name}= globals().get("{name.strip()}", {lit})'
        return f"{name}= {lit}"
    sub(r'(?P<name>\w+\s*)=\s*os\.environ\.get\(\s*"[^"]+"\s*,\s*(?P<d>"[^"]*")\s*\)(?!\s*(!=|==))',
        _strrepl)

    # 4. inline uses inside a call:  d_iters=int(os.environ.get("E", X)),
    sub(r'(int|float)\(\s*os\.environ\.get\(\s*"[^"]+"\s*,\s*(' + _NUM + r'|"[^"]*"|\w+)\s*\)\s*\)',
        lambda m: _literal(m.group(1), m.group(2)))

    # 5. `os.environ.setdefault("RA_X", "v")` in a wrapper -> bind the name the shared file reads
    def _setdef(m):
        env, val = m.group(1), m.group(2)
        name = env.split("_", 1)[1] if env.startswith(("RA_", "BE_", "DIV_")) else env
        name = {"VIZ_SEED": "VIZ_SEED", "VIZ_NPT": "N_PER_TIME", "VIZ_NBG": "N_BG",
                "VIZ_SKIP_PENDING": "SKIP_PENDING", "CLUST_ALL": "CLUST_ALL",
                "VIZ_SAVE": "SAVE", "VIZ_FS": "FS"}.get(name, name)
        lit = val if re.fullmatch(_NUM, val) else f'"{val}"'
        return f"{name} = {lit}"
    sub(r'os\.environ\.setdefault\(\s*"([^"]+)"\s*,\s*"([^"]*)"\s*\)', _setdef)

    # 6. leftovers that only exist to support an override we have removed
    sub(r'os\.environ\.get\(\s*"(RA_H_GRID|RA_TAU_GRID|RA_GENES|LOO_ALLOW_FLAT|TRAJ_ALLOW_FLAT|'
        r'LOO_RESULTS_ROOT|TRAJ_RESULTS_ROOT)"\s*\)', "None")
    sub(r'os\.environ\.get\(\s*"[^"]+"\s*\)', "None")
    return out, n


def main():
    files = []
    for dirpath, _dirs, names in os.walk(ROOT):
        if os.sep + "src" in dirpath:
            continue
        for nm in sorted(names):
            if nm.endswith(".py") and not nm.startswith("_make"):
                files.append(os.path.join(dirpath, nm))
    total = 0
    for f in files:
        txt = open(f).read()
        new, n = transform(txt, os.path.basename(f))
        if n:
            open(f, "w").write(new)
            total += n
        left = len(re.findall(r"os\.environ", new))
        print(f"  {os.path.relpath(f, ROOT):62s} rewrote {n:>3}   leftover os.environ: {left}")
    print(f"\n  {total} substitutions over {len(files)} files")


if __name__ == "__main__":
    main()
