"""Private structured-operator implementation module."""

from __future__ import annotations

# This module owns the public Boson facade and its explicit compile contract.
from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from .. import _native
from ..hamiltonian import (
    DEFAULT_MAX_BYTES,
    CompileResult,
    _effective_max_bytes,
)
from .base import (
    _boson_from_native,
    _native_boson_raw,
    _nonnegative_int,
    _StructuredOperator,
    _Term,
)
from .words import (
    BosonWord,
)


if TYPE_CHECKING:
    pass

from .compile import _compile_operator
from .space import OperatorSpace


class BosonOperator(_StructuredOperator):
    """Immutable symbolic boson operator with infinite-Fock CCR semantics.

    Finite matrix and MVP targets require explicit per-mode occupation cutoffs;
    symbolic algebra itself remains cutoff-free.
    """

    _domain = "boson"

    @classmethod
    def _from_words(
        cls, n_modes: int, products: Mapping[BosonWord, int]
    ) -> "BosonOperator":
        return cls._from_terms(
            OperatorSpace(bosons=n_modes),
            (
                _Term(None, word, (), None, None, coefficient)
                for word, coefficient in products.items()
            ),
        )

    @classmethod
    def from_terms(
        cls,
        n_modes: int,
        terms: Iterable[Tuple[Sequence[Tuple[int, str]], complex]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        """Construct and CCR-canonicalize raw ordered boson factors."""
        return _boson_from_native(
            cls,
            _nonnegative_int(n_modes, "n_modes"),
            _native_boson_raw(n_modes, terms, max_bytes),
        )

    def multiply(
        self,
        other: "_StructuredOperator",
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "_StructuredOperator":
        """Multiply two boson operators while retaining CCR contractions."""
        if not isinstance(other, BosonOperator) or other.space != self.space:
            return cast(
                "_StructuredOperator", super().multiply(other, max_bytes=max_bytes)
            )
        left_handle = self._native_handle
        right_handle = other._native_handle
        if not isinstance(
            left_handle, _native.NativeBosonOperatorHandle
        ) or not isinstance(right_handle, _native.NativeBosonOperatorHandle):
            raise RuntimeError("boson operators must retain native handles")
        result = left_handle.multiply(right_handle, _effective_max_bytes(max_bytes))
        return _boson_from_native(
            BosonOperator,
            self.space.bosons,
            result,
        )

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "BosonOperator":
        raw_terms = []
        for term in terms:
            if (
                term.boson is None
                or any(
                    value is not None
                    for value in (term.fermion, term.qudit, term.mapped_fermion)
                )
                or term.qubit
            ):
                raise ValueError("boson terms must contain only boson factors")
            raw_terms.append(
                (
                    tuple(
                        [
                            (mode, "create")
                            for mode, create, _ in term.boson.blocks
                            for _ in range(create)
                        ]
                        + [
                            (mode, "annihilate")
                            for mode, _, annihilate in term.boson.blocks
                            for _ in range(annihilate)
                        ]
                    ),
                    term.coefficient,
                )
            )
        return _boson_from_native(
            cls,
            space.bosons,
            _native_boson_raw(space.bosons, raw_terms, max_bytes),
        )

    def compile(
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        boson_cutoffs: Mapping[object, object],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a boson operator with its required finite cutoffs."""
        return _compile_operator(
            self,
            target,
            storage=storage,
            boson_cutoffs=boson_cutoffs,
            max_bytes=max_bytes,
        )
