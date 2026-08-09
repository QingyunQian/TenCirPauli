"""Canonical structured value objects and their small algebra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from ..hamiltonian import DEFAULT_MAX_BYTES
from .base import _U32_MAX, _nonnegative_int, _positive_mode


if TYPE_CHECKING:
    from .base import BosonOperator, FermionOperator


@dataclass(frozen=True)
class FermionWord:
    """Canonical fermionic monomial ``creations * annihilations``.

    Creation modes are stored in increasing order and annihilation modes in
    decreasing order. Use :meth:`from_factors` for arbitrary raw products so
    CAR contractions and signs are canonicalized.
    """

    n_modes: int
    creation_modes: Tuple[int, ...] = ()
    annihilation_modes: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n_modes = _nonnegative_int(self.n_modes, "n_modes")
        creations = tuple(self.creation_modes)
        annihilations = tuple(self.annihilation_modes)
        if any(
            not isinstance(mode, int)
            or isinstance(mode, bool)
            or not 0 <= mode < n_modes
            for mode in (*creations, *annihilations)
        ):
            raise ValueError("fermion mode is outside the operator space")
        if creations != tuple(sorted(creations)) or len(set(creations)) != len(
            creations
        ):
            raise ValueError("creation_modes must be strictly increasing")
        if annihilations != tuple(sorted(annihilations, reverse=True)) or len(
            set(annihilations)
        ) != len(annihilations):
            raise ValueError("annihilation_modes must be strictly decreasing")
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "creation_modes", creations)
        object.__setattr__(self, "annihilation_modes", annihilations)

    @property
    def is_identity(self) -> bool:
        """Whether the word contains no fermionic ladder factors."""
        return not self.creation_modes and not self.annihilation_modes

    @property
    def parity(self) -> int:
        """Return the fermion-number parity of the word."""
        return (len(self.creation_modes) + len(self.annihilation_modes)) & 1

    @property
    def factors(self) -> Tuple[Tuple[int, str], ...]:
        """Return the canonical raw factor sequence."""
        return tuple((mode, "create") for mode in self.creation_modes) + tuple(
            (mode, "annihilate") for mode in self.annihilation_modes
        )

    @classmethod
    def from_factors(
        cls,
        n_modes: int,
        factors: Sequence[Tuple[int, str]],
        *,
        max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    ) -> "FermionOperator":
        """Construct and canonicalize one raw fermion word."""
        from .base import FermionOperator

        return FermionOperator.from_terms(
            n_modes, ((tuple(factors), 1.0),), max_bytes=max_bytes
        )

    def adjoint(self) -> "FermionOperator":
        """Return the coefficient-free adjoint word as an operator."""
        from .base import FermionOperator

        factors = tuple(
            (mode, "create") for mode in reversed(self.annihilation_modes)
        ) + tuple((mode, "annihilate") for mode in reversed(self.creation_modes))
        return FermionOperator.from_terms(self.n_modes, ((factors, 1.0),))

    def multiply(self, other: "FermionWord") -> "FermionOperator":
        """Multiply two words and retain all CAR contraction terms."""
        if not isinstance(other, FermionWord) or other.n_modes != self.n_modes:
            raise ValueError("fermion words require equal n_modes")
        from .base import FermionOperator

        return FermionOperator.from_terms(
            self.n_modes, ((self.factors + other.factors, 1.0),)
        )

    def __mul__(self, other: "FermionWord") -> "FermionOperator":
        return self.multiply(other)


@dataclass(frozen=True)
class BosonWord:
    """Canonical normal-ordered bosonic power blocks.

    Each block is ``(mode, creation_power, annihilation_power)``. Raw products
    should be constructed through :class:`BosonOperator` so CCR contractions
    are retained exactly.
    """

    n_modes: int
    blocks: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        n_modes = _nonnegative_int(self.n_modes, "n_modes")
        blocks = tuple(tuple(block) for block in self.blocks)
        previous = -1
        normalized: List[Tuple[int, int, int]] = []
        for block in blocks:
            if len(block) != 3:
                raise ValueError(
                    "boson blocks must be (mode, creation_power, annihilation_power)"
                )
            mode, creation, annihilation = block
            mode = _positive_mode(mode, n_modes)
            creation = _nonnegative_int(creation, "creation_power")
            annihilation = _nonnegative_int(annihilation, "annihilation_power")
            if mode <= previous:
                raise ValueError("boson blocks must be strictly increasing by mode")
            previous = mode
            if creation or annihilation:
                normalized.append((mode, creation, annihilation))
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "blocks", tuple(normalized))

    @property
    def is_identity(self) -> bool:
        """Whether the word contains no bosonic ladder factors."""
        return not self.blocks

    @property
    def factors(self) -> Tuple[Tuple[int, str], ...]:
        """Expand canonical power blocks into raw ladder factors."""
        result: List[Tuple[int, str]] = []
        for mode, creation, annihilation in self.blocks:
            result.extend((mode, "create") for _ in range(creation))
            result.extend((mode, "annihilate") for _ in range(annihilation))
        return tuple(result)

    def adjoint(self) -> "BosonOperator":
        """Return the coefficient-free adjoint word as an operator."""
        from .base import BosonOperator

        factors = tuple(
            (mode, "create")
            for mode, _, annihilation in reversed(self.blocks)
            for _ in range(annihilation)
        ) + tuple(
            (mode, "annihilate")
            for mode, creation, _ in reversed(self.blocks)
            for _ in range(creation)
        )
        return BosonOperator.from_terms(self.n_modes, ((factors, 1.0),))

    def multiply(self, other: "BosonWord") -> "BosonOperator":
        """Multiply two words and retain all CCR contraction terms."""
        if not isinstance(other, BosonWord) or other.n_modes != self.n_modes:
            raise ValueError("boson words require equal n_modes")
        from .base import BosonOperator

        return BosonOperator.from_terms(
            self.n_modes, ((self.factors + other.factors, 1.0),)
        )

    def __mul__(self, other: "BosonWord") -> "BosonOperator":
        return self.multiply(other)


@dataclass(frozen=True)
class QuditWeylWord:
    """Direct-convention ``X**a Z**b`` word with modular exponents.

    Exponents are reduced modulo ``dimension`` and triples are sorted by site.
    Multiplication returns a :class:`QuditProduct` with the modular phase
    exponent kept separate from the canonical word.
    """

    dimension: int
    triples: Tuple[Tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        dimension = _nonnegative_int(self.dimension, "dimension")
        if not 3 <= dimension <= _U32_MAX:
            raise ValueError("qudit dimension must satisfy 3 <= dimension <= 2**32-1")
        triples = tuple(tuple(triple) for triple in self.triples)
        previous = -1
        normalized: List[Tuple[int, int, int]] = []
        for triple in triples:
            if len(triple) != 3:
                raise ValueError("Weyl triples must be (site, a, b)")
            site, a, b = triple
            site = _nonnegative_int(site, "site")
            a = _nonnegative_int(a, "a") % dimension
            b = _nonnegative_int(b, "b") % dimension
            if site <= previous:
                raise ValueError("Weyl triples must be strictly increasing by site")
            previous = site
            if a or b:
                normalized.append((site, a, b))
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "triples", tuple(normalized))

    @property
    def is_identity(self) -> bool:
        """Whether every site carries the identity Weyl factor."""
        return not self.triples

    @property
    def n_sites(self) -> int:
        """Return one past the largest explicitly stored site index."""
        return 0 if not self.triples else self.triples[-1][0] + 1

    def multiply(self, other: "QuditWeylWord") -> "QuditProduct":
        """Multiply words and return the modular phase exponent separately."""
        if not isinstance(other, QuditWeylWord) or other.dimension != self.dimension:
            raise ValueError("Weyl words require equal local dimensions")
        left = {site: (a, b) for site, a, b in self.triples}
        right = {site: (a, b) for site, a, b in other.triples}
        exponent = 0
        result: List[Tuple[int, int, int]] = []
        for site in sorted(set(left) | set(right)):
            a, b = left.get(site, (0, 0))
            c, e = right.get(site, (0, 0))
            exponent = (exponent + b * c) % self.dimension
            aa, bb = (a + c) % self.dimension, (b + e) % self.dimension
            if aa or bb:
                result.append((site, aa, bb))
        return QuditProduct(QuditWeylWord(self.dimension, tuple(result)), exponent)

    def commutes_with(self, other: "QuditWeylWord") -> bool:
        """Check the exact modular Weyl commutation condition."""
        if not isinstance(other, QuditWeylWord) or other.dimension != self.dimension:
            raise ValueError("Weyl words require equal local dimensions")
        left = {site: (a, b) for site, a, b in self.triples}
        right = {site: (a, b) for site, a, b in other.triples}
        exponent = sum(
            b * c - e * a
            for site in set(left) | set(right)
            for a, b in (left.get(site, (0, 0)),)
            for c, e in (right.get(site, (0, 0)),)
        )
        return exponent % self.dimension == 0

    def adjoint(self) -> "QuditProduct":
        """Return the adjoint word and its modular phase exponent."""
        exponent = sum(a * b for _, a, b in self.triples) % self.dimension
        result = tuple(
            (site, (-a) % self.dimension, (-b) % self.dimension)
            for site, a, b in self.triples
            if a or b
        )
        return QuditProduct(QuditWeylWord(self.dimension, result), exponent)


@dataclass(frozen=True)
class QuditProduct:
    """A Weyl word together with its modular phase exponent."""

    word: QuditWeylWord
    phase_exponent: int


@dataclass(frozen=True)
class FermionTerm:
    """One canonical fermion word and its complex coefficient."""

    word: FermionWord
    coefficient: complex


@dataclass(frozen=True)
class BosonTerm:
    """One canonical boson word and its complex coefficient."""

    word: BosonWord
    coefficient: complex


@dataclass(frozen=True)
class QuditWeylTerm:
    """One direct-convention Weyl word and its complex coefficient."""

    word: QuditWeylWord
    coefficient: complex
