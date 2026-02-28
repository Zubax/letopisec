from __future__ import annotations

from pathlib import Path

import nox

PACKAGE_DIR = Path("letopisec")
EXCLUDED_TEST_FILE = "__main__.py"

nox.options.sessions = ("tests", "black", "mypy")


def _discover_test_modules() -> list[str]:
    modules = sorted(
        f"{PACKAGE_DIR.name}.{path.stem}" for path in PACKAGE_DIR.glob("*.py") if path.name != EXCLUDED_TEST_FILE
    )
    if not modules:
        raise RuntimeError(f"No Python files found under {PACKAGE_DIR}/ excluding {EXCLUDED_TEST_FILE}")
    return modules


@nox.session
def tests(session: nox.Session) -> None:
    session.install("coverage", "httpx", ".")
    modules = _discover_test_modules()
    session.run(
        "coverage",
        "run",
        "--branch",
        "--source",
        PACKAGE_DIR.name,
        "--omit",
        "*/letopisec/__main__.py",
        "-m",
        "unittest",
        *modules,
    )
    session.run(
        "coverage",
        "report",
        "-m",
        "--omit",
        "*/letopisec/__main__.py",
        "--fail-under=80",
    )


@nox.session
def black(session: nox.Session) -> None:
    session.install("black")
    session.run("black", "--check", ".")


@nox.session
def mypy(session: nox.Session) -> None:
    session.install("mypy", ".")
    session.run("mypy", "letopisec")
