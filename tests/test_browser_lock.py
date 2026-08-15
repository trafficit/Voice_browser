"""Assert-based self-check for BrowserController._clear_stale_lock.

Run inside the project's env (selenium installed): python tests/test_browser_lock.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from browser import BrowserController  # noqa: E402


def test_removes_existing_lock_files():
    with tempfile.TemporaryDirectory() as d:
        names = ("SingletonLock", "SingletonSocket", "SingletonCookie")
        for name in names:
            open(os.path.join(d, name), "w").close()

        BrowserController._clear_stale_lock(d)

        for name in names:
            assert not os.path.exists(os.path.join(d, name)), f"{name} should be removed"


def test_missing_lock_files_do_not_raise():
    with tempfile.TemporaryDirectory() as d:
        BrowserController._clear_stale_lock(d)  # must not raise


if __name__ == "__main__":
    test_removes_existing_lock_files()
    test_missing_lock_files_do_not_raise()
    print("OK")
