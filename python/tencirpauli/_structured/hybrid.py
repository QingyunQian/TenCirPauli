"""Private structured-operator implementation module."""

from __future__ import annotations

# This module owns the public Hybrid facade and its explicit compile contract.
import cmath
import math
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .. import _native
from .._validation import normalize_pauli_code
from ..hamiltonian import (
    DEFAULT_MAX_BYTES,
    CompileResult,
    _effective_max_bytes,
)
from .base import (
    _IDENTITY_CODES,
    _PAULI_PRODUCT,
    _factor,
    _finite_complex,
    _hybrid_from_native,
    _positive_mode,
    _StructuredOperator,
    _Term,
)
from .words import (
    QuditWeylWord,
)


if TYPE_CHECKING:
    from ..mapping import FermionQubitMapping

from .compile import _compile_operator
from .space import OperatorSpace


class HybridOperator(_StructuredOperator):
    """Immutable mixed-domain operator with canonical domain factors.

    Hybrid terms may combine fermion, boson, qubit, and qudit factors. Fermion
    mapping and finite boson cutoffs are explicit at compilation time.
    """

    _domain = "hybrid"

    @classmethod
    def _from_terms(
        cls,
        space: OperatorSpace,
        terms: Iterable[_Term],
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "HybridOperator":
        raw_terms = tuple(terms)
        fermion_present: List[bool] = []
        fermion_factors: List[List[Tuple[int, int]]] = []
        boson_factors: List[List[Tuple[int, int]]] = []
        boson_present: List[bool] = []
        boson_blocks: List[List[Tuple[int, int, int]]] = []
        qubit_codes: List[List[int]] = []
        mapped_present: List[bool] = []
        mapped_codes: List[List[int]] = []
        qudit_present: List[bool] = []
        qudit_triples: List[List[Tuple[int, int, int]]] = []
        coefficients: List[complex] = []
        for term in raw_terms:
            fermion_present.append(term.fermion is not None)
            fermion_factors.append(
                []
                if term.fermion is None
                else [(mode, 0) for mode in term.fermion.creation_modes]
                + [(mode, 1) for mode in term.fermion.annihilation_modes]
            )
            boson_factors.append(
                []
                if term.boson is None
                else [
                    (mode, 0)
                    for mode, create, _ in term.boson.blocks
                    for _ in range(create)
                ]
                + [
                    (mode, 1)
                    for mode, _, annihilate in term.boson.blocks
                    for _ in range(annihilate)
                ]
            )
            boson_present.append(term.boson is not None)
            boson_blocks.append(
                list(term.boson.blocks) if term.boson is not None else []
            )
            qubit_codes.append(list(term.qubit))
            mapped_present.append(term.mapped_fermion is not None)
            mapped_codes.append(
                list(term.mapped_fermion)
                if term.mapped_fermion is not None
                else [0] * space.fermions
            )
            qudit_present.append(term.qudit is not None)
            qudit_triples.append([] if term.qudit is None else list(term.qudit.triples))
            coefficients.append(term.coefficient)
        real = [value.real for value in coefficients]
        imaginary = [value.imag for value in coefficients]
        if any(mapped_present):
            result = _native.structured_hybrid_jordan_wigner(
                space.fermions,
                space.bosons,
                space.qubits,
                len(space.qudits),
                space.qudits[0] if space.qudits else 0,
                (
                    fermion_present,
                    [
                        [mode for mode, action in factors if action == 0]
                        for factors in fermion_factors
                    ],
                    [
                        [mode for mode, action in factors if action == 1]
                        for factors in fermion_factors
                    ],
                    boson_present,
                    boson_blocks,
                    qubit_codes,
                    mapped_present,
                    mapped_codes,
                    qudit_present,
                    qudit_triples,
                    real,
                    imaginary,
                ),
                _effective_max_bytes(max_bytes),
            )
        else:
            result = _native.structured_hybrid_canonicalize(
                space.fermions,
                space.bosons,
                space.qubits,
                len(space.qudits),
                space.qudits[0] if space.qudits else 0,
                (
                    fermion_factors,
                    boson_factors,
                    qubit_codes,
                    qudit_present,
                    qudit_triples,
                    real,
                    imaginary,
                ),
                _effective_max_bytes(max_bytes),
            )
        return _hybrid_from_native(space, result, cls)

    def compile(
        self,
        target: str,
        *,
        storage: Literal["lazy", "eager"] = "lazy",
        mapping: Union[str, "FermionQubitMapping"] = "jordan_wigner",
        boson_cutoffs: Optional[Mapping[object, object]] = None,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> CompileResult:
        """Compile a hybrid operator with optional explicit Fermion mapping.

        The default native-MVP route retains raw Fermion descriptors; calling
        :meth:`map_fermions` first selects mapped-Pauli execution instead.
        """
        return _compile_operator(
            self,
            target,
            storage=storage,
            mapping=mapping,
            boson_cutoffs=boson_cutoffs,
            max_bytes=max_bytes,
        )


HybridTerm = _Term


class OperatorBuilder:
    """Single-owner batched structured-term builder.

    Add raw products incrementally, then call :meth:`finish` once to perform
    native canonicalization and duplicate aggregation in one coarse-grained
    operation.
    """

    def __init__(self, space: OperatorSpace) -> None:
        self.space = space
        self._products: List[Dict[str, Any]] = []

    def add_product(
        self,
        coefficient: complex = 1.0,
        *,
        fermions: Sequence[Tuple[int, str]] = (),
        bosons: Sequence[Tuple[int, str]] = (),
        qubits: Sequence[Tuple[int, object]] = (),
        qudits: Sequence[Tuple[int, int, int]] = (),
    ) -> "OperatorBuilder":
        """Append one raw hybrid product to the construction buffer.

        The builder accepts sparse factors for each domain and returns itself
        so products can be chained. Validation and canonicalization occur in
        :meth:`finish`.
        """
        self._products.append(
            {
                "coefficient": _finite_complex(coefficient),
                "fermions": tuple(fermions),
                "bosons": tuple(bosons),
                "qubits": tuple(qubits),
                "qudits": tuple(qudits),
            }
        )
        return self

    def finish(self, *, max_bytes: Optional[int] = DEFAULT_MAX_BYTES) -> HybridOperator:
        """Canonicalize all buffered products in one native batch call.

        Returns a deterministic :class:`HybridOperator`; repeated equal
        products are aggregated and exact zeros are removed.
        """
        fermion_factors: List[List[Tuple[int, int]]] = []
        boson_factors: List[List[Tuple[int, int]]] = []
        qubit_codes: List[List[int]] = []
        qudit_present: List[bool] = []
        qudit_triples: List[List[Tuple[int, int, int]]] = []
        coefficients: List[complex] = []
        for product_spec in self._products:
            normalized_factors: List[Tuple[int, int]] = []
            for factor in product_spec["fermions"]:
                mode, action = _factor(factor, self.space.fermions)
                normalized_factors.append((mode, int(action == "annihilate")))
            normalized_bosons: List[Tuple[int, int]] = []
            for factor in product_spec["bosons"]:
                mode, action = _factor(factor, self.space.bosons)
                normalized_bosons.append((mode, int(action == "annihilate")))
            qcodes = [0] * self.space.qubits
            qword: Optional[QuditWeylWord] = None
            qphase = 1.0 + 0j
            for index, code in product_spec["qubits"]:
                if isinstance(code, str):
                    code = code.upper()
                    if code not in _IDENTITY_CODES:
                        raise ValueError("Pauli code must be one of I, X, Y, Z")
                    code_value = _IDENTITY_CODES[code]
                else:
                    code_value = normalize_pauli_code(code)
                qubit_index = _positive_mode(index, self.space.qubits, "qubit")
                current_code = qcodes[qubit_index]
                qcodes[qubit_index], local_phase = _PAULI_PRODUCT[current_code][
                    code_value
                ]
                qphase *= local_phase
            if self.space.qudits:
                qword = QuditWeylWord(self.space.qudits[0])
                for site, a, b in product_spec["qudits"]:
                    site = _positive_mode(site, len(self.space.qudits), "site")
                    factor = QuditWeylWord(self.space.qudits[0], ((site, a, b),))
                    product_result = qword.multiply(factor)
                    qword = product_result.word
                    qphase *= cmath.exp(
                        2j
                        * math.pi
                        * product_result.phase_exponent
                        / self.space.qudits[0]
                    )
            fermion_factors.append(normalized_factors)
            boson_factors.append(normalized_bosons)
            qubit_codes.append(qcodes)
            qudit_present.append(qword is not None)
            qudit_triples.append(list(qword.triples) if qword is not None else [])
            coefficients.append(product_spec["coefficient"] * qphase)
        result = _native.structured_hybrid_canonicalize(
            self.space.fermions,
            self.space.bosons,
            self.space.qubits,
            len(self.space.qudits),
            self.space.qudits[0] if self.space.qudits else 0,
            (
                fermion_factors,
                boson_factors,
                qubit_codes,
                qudit_present,
                qudit_triples,
                [value.real for value in coefficients],
                [value.imag for value in coefficients],
            ),
            _effective_max_bytes(max_bytes),
        )
        return _hybrid_from_native(self.space, result)
