"""Unit tests for static architecture guardrails."""

from __future__ import annotations

from pathlib import Path

from scripts.check_architecture import ExceptionEntry, check_source, check_tree


def test_architecture_checker_detects_namespace_mutation() -> None:
    violations, _ = check_source(
        'globals()["hidden"] = 1\n',
        Path("python/tencirpauli/example.py"),
        entries=(),
    )
    assert [violation.rule for violation in violations] == ["namespace-mutation"]


def test_architecture_checker_detects_namespace_mutation_calls() -> None:
    for source in (
        'globals().update({"hidden": 1})\n',
        'module.__dict__.setdefault("hidden", 1)\n',
    ):
        violations, _ = check_source(
            source,
            Path("python/tencirpauli/example.py"),
            entries=(),
        )
        assert [violation.rule for violation in violations] == ["namespace-mutation"]


def test_architecture_checker_detects_composite_any_annotations() -> None:
    violations, _ = check_source(
        "def native(value: tuple[int, Any]) -> list[Any]: ...\n",
        Path("python/tencirpauli/_native.pyi"),
        entries=(),
    )
    assert [violation.position for violation in violations] == [
        "parameter:value",
        "return",
    ]


def test_architecture_checker_accepts_a_stable_exception_entry() -> None:
    entry = ExceptionEntry(
        "TEST-NAMESPACE",
        "namespace-mutation",
        "python/tencirpauli/example.py",
        "<module>",
        "assignment",
        "test",
        "fixture",
        "test removal",
    )
    violations, used = check_source(
        'globals()["hidden"] = 1\n',
        Path("python/tencirpauli/example.py"),
        entries=(entry,),
    )
    assert violations == []
    assert used == {"TEST-NAMESPACE"}


def test_architecture_checker_rejects_stale_exceptions_and_syntax_errors() -> None:
    stale_entry = ExceptionEntry(
        "TEST-STALE",
        "namespace-mutation",
        "python/tencirpauli/example.py",
        "<module>",
        "assignment",
        "test",
        "fixture",
        "test removal",
    )
    violations, stale = check_tree(paths=(), entries=(stale_entry,))
    assert violations == []
    assert stale == [stale_entry]

    syntax_violations, _ = check_source(
        "def broken(:\n", Path("python/tencirpauli/example.py"), entries=()
    )
    assert [violation.rule for violation in syntax_violations] == ["syntax-error"]
