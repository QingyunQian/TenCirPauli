"""Stable public facade for structured operator algebra.

The implementation lives in the private :mod:`tencirpauli._structured`
package.  Public values keep this module name so import identity, repr, and
pickle contracts remain stable while the implementation is split internally.
"""

from __future__ import annotations

from typing import Any

from ._structured import base as _base
from ._structured.base import (
    _fermion_from_integral_blocks as _fermion_from_integral_blocks,
)
from ._structured.base import (
    _fermion_from_native as _fermion_from_native,
)
from ._structured.base import (
    _finite_complex as _finite_complex,
)
from ._structured.base import (
    _hybrid_from_native as _hybrid_from_native,
)
from ._structured.base import (
    _StructuredOperator as _StructuredOperator,
)
from ._structured.base import (
    _Term as _Term,
)
from ._structured.boson import BosonOperator
from ._structured.fermion import FermionOperator
from ._structured.hybrid import HybridOperator, OperatorBuilder
from ._structured.mapping import QuditWeylOperator
from ._structured.mapping import _jordan_wigner_word as _jordan_wigner_word
from ._structured.mapping import _make_operator as _make_operator
from ._structured.mapping import _with_plan_metadata as _with_plan_metadata
from ._structured.space import OperatorSpace
from ._structured.words import (
    BosonTerm,
    BosonWord,
    FermionTerm,
    FermionWord,
    QuditProduct,
    QuditWeylTerm,
    QuditWeylWord,
)


HybridTerm = _base.HybridTerm

__all__ = [
    "BosonOperator",
    "BosonTerm",
    "BosonWord",
    "FermionOperator",
    "FermionTerm",
    "FermionWord",
    "HybridOperator",
    "HybridTerm",
    "OperatorBuilder",
    "OperatorSpace",
    "QuditProduct",
    "QuditWeylOperator",
    "QuditWeylTerm",
    "QuditWeylWord",
]

for _public_name in __all__:
    _public_value = globals()[_public_name]
    if isinstance(_public_value, type):
        _public_value.__module__ = __name__


def __getattr__(name: str) -> Any:
    """Expose private compatibility imports without duplicating implementation."""
    return getattr(_base, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_base)))


del _public_name, _public_value
