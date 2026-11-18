"""Test runner entry point: python -I run_tests.py <module names>"""

import sys

sys.path.insert(0, "/workspace/work/pytest_shim")
sys.path.insert(0, "/workspace")

import pytest  # noqa: E402  (the local shim)

if __name__ == "__main__":
    total_failures = 0
    total_passed = 0
    total_skipped = 0
    for name in sys.argv[1:]:
        r = pytest.run_module(name)
        total_passed += r.passed
        total_failures += r.failed
        total_skipped += r.skipped
    print(
        f"\nGRAND TOTAL: {total_passed} passed, {total_failures} failed,"
        f" {total_skipped} skipped"
    )
    sys.exit(1 if total_failures else 0)
