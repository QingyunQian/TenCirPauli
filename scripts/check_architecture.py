"""Enforce the repository's Python/native architecture contracts."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = ROOT / "python" / "tencirpauli"


@dataclass(frozen=True)
class ExceptionEntry:
    """One stable, reviewable architecture exception."""

    exception_id: str
    rule: str
    path: str
    symbol: str
    position: str
    owner: str
    reason: str
    remove_when: str


@dataclass(frozen=True)
class Violation:
    """One architecture violation with a stable source location."""

    rule: str
    path: str
    symbol: str
    position: str
    line: int
    message: str


# ``object`` remains valid only at genuine dynamic native boundaries. Keep this
# ledger explicit: fixed NumPy arrays, handles, and tuple shapes must use the
# aliases in _native.pyi instead of joining this exception list.
_DYNAMIC_NATIVE_INPUTS = (
    ("u1_circuit_plan", "parameter:angles"),
    ("u1_circuit_plan", "parameter:gates"),
    ("NativeU1CircuitPlan::run", "parameter:parameters"),
    ("NativeU1CircuitPlan::run_cached", "parameter:parameters"),
    ("NativeU1CircuitPlan::probability", "parameter:parameters"),
    ("NativeU1CircuitPlan::to_dense", "parameter:parameters"),
    ("NativeU1CircuitPlan::probability_full", "parameter:parameters"),
    ("NativeU1CircuitPlan::expectation_handle", "parameter:parameters"),
    ("NativeU1CircuitPlan::value_and_grad_handle", "parameter:parameters"),
    ("NativePropagationEngine::expectation", "parameter:parameters"),
    ("NativePropagationEngine::value_and_grad", "parameter:parameters"),
    ("NativePropagationEngine::propagate_operator", "parameter:parameters"),
    ("NativePropagationEngine::profile", "parameter:parameters"),
    ("NativePropagationBatch::expectations", "parameter:parameters"),
    ("NativePropagationBatch::values_and_gradients", "parameter:parameters"),
    ("NativeSPPSEngine::expectation", "parameter:parameters"),
    ("NativeSPPSEngine::value_and_grad", "parameter:parameters"),
    ("NativeSPPSEngine::value_and_grad_adaptive", "parameter:parameters"),
)

EXCEPTIONS: tuple[ExceptionEntry, ...] = (
    ExceptionEntry(
        "ARCH-GETATTR-STRUCTURED",
        "dynamic-getattr",
        "python/tencirpauli/structured.py",
        "__getattr__",
        "function",
        "maintainers",
        "Preserve the documented private compatibility forwarder.",
        "Remove only in an explicitly approved breaking release.",
    ),
    *tuple(
        ExceptionEntry(
            f"ARCH-ABI-DYNAMIC-{symbol.replace('::', '-')}-{position.replace(':', '-')}",
            "broad-native-annotation",
            "python/tencirpauli/_native.pyi",
            symbol,
            position,
            "native-maintainers",
            "Backend arrays and scalar containers are intentionally accepted dynamically.",
            "Replace with a backend protocol when the public backend contract is frozen.",
        )
        for symbol, position in _DYNAMIC_NATIVE_INPUTS
    ),
)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _contains_name(node: Optional[ast.AST], names: set[str]) -> bool:
    if node is None:
        return False
    return any(
        isinstance(item, ast.Name) and item.id in names for item in ast.walk(node)
    )


def _function_symbol(stack: Sequence[str], name: str) -> str:
    return "::".join((*stack, name)) if stack else name


def _entry_matches(
    entry: ExceptionEntry, rule: str, path: str, symbol: str, position: str
) -> bool:
    return (
        entry.rule == rule
        and entry.path == path
        and entry.symbol == symbol
        and entry.position == position
    )


def _violation(
    violations: list[Violation],
    used: set[str],
    entries: Sequence[ExceptionEntry],
    *,
    rule: str,
    path: str,
    symbol: str,
    position: str,
    line: int,
    message: str,
) -> None:
    for entry in entries:
        if _entry_matches(entry, rule, path, symbol, position):
            used.add(entry.exception_id)
            return
    violations.append(Violation(rule, path, symbol, position, line, message))


class _Visitor(ast.NodeVisitor):
    def __init__(self, *, path: str, entries: Sequence[ExceptionEntry]) -> None:
        self.path = path
        self.entries = entries
        self.violations: list[Violation] = []
        self.used: set[str] = set()
        self.class_stack: list[str] = []
        self.function_stack: list[str] = []

    @property
    def symbol(self) -> str:
        return "::".join((*self.class_stack, *self.function_stack)) or "<module>"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "*" for alias in node.names):
            _violation(
                self.violations,
                self.used,
                self.entries,
                rule="wildcard-import",
                path=self.path,
                symbol=self.symbol,
                position="import",
                line=node.lineno,
                message="wildcard imports hide the module dependency graph",
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if not self.class_stack and node.name == "__getattr__":
            _violation(
                self.violations,
                self.used,
                self.entries,
                rule="dynamic-getattr",
                path=self.path,
                symbol=node.name,
                position="function",
                line=node.lineno,
                message="module-level dynamic __getattr__ requires an explicit exception",
            )
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _check_assignment(self, node: ast.AST, value: Optional[ast.AST]) -> None:
        if _contains_name(node, {"globals"}) or _contains_name(node, {"__dict__"}):
            _violation(
                self.violations,
                self.used,
                self.entries,
                rule="namespace-mutation",
                path=self.path,
                symbol=self.symbol,
                position="assignment",
                line=getattr(node, "lineno", 0),
                message="module namespace mutation hides ownership and dependencies",
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment(node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment(node.target, node.value)
        self.generic_visit(node)


def _annotation_has_broad_type(annotation: Optional[ast.AST]) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in {"object", "Any"}
    return _contains_name(annotation, {"object"})


def _check_stub_annotations(
    tree: ast.AST,
    *,
    path: str,
    entries: Sequence[ExceptionEntry],
    violations: list[Violation],
    used: set[str],
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        symbol = _function_symbol(_enclosing_classes(tree, node), node.name)
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if node.args.vararg is not None:
            arguments += (node.args.vararg,)
        if node.args.kwarg is not None:
            arguments += (node.args.kwarg,)
        for argument in arguments:
            if _annotation_has_broad_type(argument.annotation):
                _violation(
                    violations,
                    used,
                    entries,
                    rule="broad-native-annotation",
                    path=path,
                    symbol=symbol,
                    position=f"parameter:{argument.arg}",
                    line=argument.lineno,
                    message="native ABI parameter needs a precise type or a ledger entry",
                )
        if _annotation_has_broad_type(node.returns):
            _violation(
                violations,
                used,
                entries,
                rule="broad-native-annotation",
                path=path,
                symbol=symbol,
                position="return",
                line=node.lineno,
                message="native ABI return needs a precise type or a ledger entry",
            )


def _enclosing_classes(tree: ast.AST, target: ast.AST) -> tuple[str, ...]:
    """Recover class nesting for a function in a stub without line-based keys."""
    result: list[str] = []

    def walk(node: ast.AST, classes: tuple[str, ...]) -> bool:
        if node is target:
            result.extend(classes)
            return True
        for child in ast.iter_child_nodes(node):
            next_classes = (
                (*classes, child.name) if isinstance(child, ast.ClassDef) else classes
            )
            if walk(child, next_classes):
                return True
        return False

    walk(tree, ())
    return tuple(result)


def check_source(
    source: str,
    path: Path,
    *,
    entries: Sequence[ExceptionEntry] = EXCEPTIONS,
) -> tuple[list[Violation], set[str]]:
    """Check one source string; used by the checker and its unit tests."""
    relative = _relative(path) if path.is_absolute() else path.as_posix()
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as error:
        return [
            Violation(
                "syntax-error",
                relative,
                "<module>",
                "syntax",
                error.lineno or 0,
                str(error),
            )
        ], set()
    visitor = _Visitor(path=relative, entries=entries)
    visitor.visit(tree)
    if relative.endswith("_native.pyi"):
        _check_stub_annotations(
            tree,
            path=relative,
            entries=entries,
            violations=visitor.violations,
            used=visitor.used,
        )
    if relative.startswith("python/") and relative.endswith(".py"):
        for line_number, line in enumerate(source.splitlines(), 1):
            if re.match(r"^\s*#\s*ruff:\s*noqa\b", line):
                _violation(
                    visitor.violations,
                    visitor.used,
                    entries,
                    rule="file-suppression",
                    path=relative,
                    symbol="<module>",
                    position="file-header",
                    line=line_number,
                    message="whole-file Ruff suppression disables dependency checks",
                )
    return visitor.violations, visitor.used


def check_tree(
    root: Path = ROOT,
    *,
    entries: Sequence[ExceptionEntry] = EXCEPTIONS,
    paths: Optional[Iterable[Path]] = None,
) -> tuple[list[Violation], list[ExceptionEntry]]:
    """Check production Python sources and return violations plus stale entries."""
    source_paths = (
        list(paths)
        if paths is not None
        else [
            *sorted((root / "python" / "tencirpauli").rglob("*.py")),
            root / "python" / "tencirpauli" / "_native.pyi",
        ]
    )
    all_violations: list[Violation] = []
    used: set[str] = set()
    for path in source_paths:
        if not path.exists() or path.suffix not in {".py", ".pyi"}:
            continue
        violations, source_used = check_source(
            path.read_text(encoding="utf-8"), path, entries=entries
        )
        all_violations.extend(violations)
        used.update(source_used)
    stale = [entry for entry in entries if entry.exception_id not in used]
    return all_violations, stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    violations, stale = check_tree(arguments.root)
    for violation in violations:
        print(
            f"{violation.path}:{violation.line}: {violation.rule} "
            f"[{violation.symbol} {violation.position}] {violation.message}"
        )
    for entry in stale:
        print(
            f"{entry.path}: stale architecture exception {entry.exception_id} "
            f"({entry.rule} {entry.symbol} {entry.position})"
        )
    return 1 if violations or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
