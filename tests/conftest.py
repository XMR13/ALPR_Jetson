import os
import sys
from pathlib import Path


def pytest_sessionstart(session):  # type: ignore[no-redef]
    """
    Ensure child processes (e.g., subprocess.run(["python", "-m", "alpr_jetson"]))
    can import the package without requiring an editable install.

    We prepend the repository's `src/` to PYTHONPATH in the test environment,
    so subprocesses inherit it.
    """
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    if src.is_dir():
        existing = os.environ.get("PYTHONPATH", "")
        new_val = str(src)
        if existing:
            new_val = new_val + os.pathsep + existing
        os.environ["PYTHONPATH"] = new_val
        # Also add to current interpreter sys.path for direct imports
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

