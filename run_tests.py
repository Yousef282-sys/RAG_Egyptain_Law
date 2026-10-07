"""Tiny offline test runner.

This environment has no network access to install pytest, but the
tests in tests/ are plain pytest-style functions (no framework features
used beyond bare asserts), so they can run standalone. If/when pytest
is available, prefer: `pytest -q`. This script is a drop-in fallback.
"""
import importlib
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).parent / "tests"


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    passed = 0
    failed = 0
    skipped = 0
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f"tests.{path.stem}"
        module = importlib.import_module(module_name)
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            try:
                result = func()
                if result == "SKIPPED":
                    skipped += 1
                    print(f"SKIP: {module_name}.{name}")
                else:
                    passed += 1
            except Exception:
                failed += 1
                print(f"FAIL: {module_name}.{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
