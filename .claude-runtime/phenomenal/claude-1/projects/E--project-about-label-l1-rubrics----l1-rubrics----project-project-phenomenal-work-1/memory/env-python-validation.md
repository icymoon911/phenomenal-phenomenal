---
name: env-python-validation
description: How to run Python/pytest validation in this project's local env (broken numpy, CJK path)
metadata:
  type: project
---

In the local dev env (anaconda base, Python 3.13.9, numpy 2.3.5), `import numpy` **hard-crashes** the interpreter with exit 127 — an uncatchable native abort (not an ImportError). This breaks any local run that imports numpy/pandas or `openalea.phenomenal` core. The package is NOT pip-installed locally either.

Also: the project root path contains non-ASCII chars (`副本`), and passing a **script-file path** to python (`python foo.py`) fails with exit 127. Module form works.

**Why:** Without knowing this, validation looks impossible / falsely "broken". It's an environment defect, not a code problem — do not "fix" by reinstalling numpy (unauthorized, risky).

**How to apply:**
- Run tests as a module with src on the path: `PYTHONPATH=src python -m pytest <path> -v` (works; `python <script>.py` does not).
- `python -c "..."` (even multiline) works for light, numpy-free code.
- Write code so the export/util modules are numpy/pandas-free at import time (lazy-import pandas only where needed) — then they import and unit-test cleanly here. See [[tracking-leaf-export]].
- For tests needing numpy/phenomenal (real-data integration), probe `subprocess.run([sys.executable,"-c","import numpy"])` and `pytest.skip` if returncode!=0, so it runs in CI but skips locally instead of crashing the session.
