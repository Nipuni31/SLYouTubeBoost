# Python version

**Recommended: Python 3.10 or 3.11** (widely used in ML / production).

## Why not 3.14?

**SHAP** depends on **Numba** / **llvmlite**. As of early 2025, those stacks are often **not fully supported** on the newest Python release. You may see:

- Very slow `import shap`
- Hangs or `KeyboardInterrupt` during import

## What to do

1. Install **Python 3.10** or **3.11** (e.g. from [python.org](https://www.python.org/downloads/) or `pyenv`).
2. Create a fresh venv:

   ```bash
   python3.11 -m venv yt_env
   yt_env\Scripts\activate   # Windows
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. A **`.python-version`** file in this repo (for `pyenv`) pins **3.11** when present.

Training and the Flask app will **still run without SHAP** if import fails (plots and per-request SHAP are skipped).
