"""Shared two-dimensional Hubbard workloads for the time-evolution study."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def lattice_bonds(rows: int, cols: int) -> Tuple[Tuple[int, int], ...]:
    bonds: List[Tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            site = row * cols + col
            if col + 1 < cols:
                bonds.append((site, site + 1))
            if row + 1 < rows:
                bonds.append((site, site + cols))
    return tuple(bonds)


def hubbard_terms(
    rows: int, cols: int, hopping: float, interaction: float
) -> List[Tuple[Tuple[Tuple[int, str], ...], complex]]:
    """Return spin-blocked open-boundary Hubbard terms."""
    sites = rows * cols
    terms: List[Tuple[Tuple[Tuple[int, str], ...], complex]] = []
    for site in range(sites):
        up = site
        down = sites + site
        terms.append(
            (
                (
                    (up, "create"),
                    (up, "annihilate"),
                    (down, "create"),
                    (down, "annihilate"),
                ),
                complex(interaction),
            )
        )
    for left, right in lattice_bonds(rows, cols):
        for spin in (0, 1):
            left_mode = left + spin * sites
            right_mode = right + spin * sites
            terms.append(
                (((left_mode, "create"), (right_mode, "annihilate")), -hopping)
            )
            terms.append(
                (((right_mode, "create"), (left_mode, "annihilate")), -hopping)
            )
    return terms


def model_info(rows: int, cols: int) -> Dict[str, int]:
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    sites = rows * cols
    n_modes = 2 * sites
    return {
        "rows": rows,
        "cols": cols,
        "sites": sites,
        "bonds": len(lattice_bonds(rows, cols)),
        "n_modes": n_modes,
        "dimension": 2**n_modes,
    }


def spinful_neel_occupations(
    rows: int, cols: int, *, up_on_even: bool = True
) -> Tuple[int, ...]:
    """Return alternating one-particle-per-site spin occupations."""
    sites = rows * cols
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    up = [0] * sites
    down = [0] * sites
    for site in range(sites):
        up_site = (site % 2 == 0) == up_on_even
        if up_site:
            up[site] = 1
        else:
            down[site] = 1
    return tuple(up + down)


def basis_state_from_occupations(
    occupations: Tuple[int, ...], *, sector: Any = None
) -> np.ndarray[Any, Any]:
    """Create a one-hot full-Fock or charge-sector basis state."""
    if any(value not in (0, 1) for value in occupations):
        raise ValueError("fermion occupations must be binary")
    if sector is None:
        index = 0
        for value in occupations:
            index = (index << 1) | value
        dimension = 1 << len(occupations)
    else:
        index = int(sector.rank(occupations))
        dimension = int(sector.dimension)
    state = np.zeros(dimension, dtype=np.complex128)
    state[index] = 1.0
    return state


def spinful_neel_state(
    rows: int, cols: int, *, sector: Any = None, up_on_even: bool = True
) -> np.ndarray[Any, Any]:
    """Return the same Néel product state in full or restricted ordering."""
    occupations = spinful_neel_occupations(rows, cols, up_on_even=up_on_even)
    return basis_state_from_occupations(occupations, sector=sector)


def spectral_radius_bound(
    rows: int, cols: int, hopping: float, interaction: float
) -> float:
    """Return a conservative row-sum bound for real-time Chebyshev scaling."""
    sites = rows * cols
    return abs(interaction) * sites + 4.0 * abs(hopping) * len(
        lattice_bonds(rows, cols)
    )


def workload(rows: int, cols: int, hopping: float, interaction: float) -> Tuple[
    Dict[str, int],
    List[Tuple[Tuple[Tuple[int, str], ...], complex]],
    np.ndarray[Any, Any],
]:
    info = model_info(rows, cols)
    return (
        info,
        hubbard_terms(rows, cols, hopping, interaction),
        spinful_neel_state(rows, cols),
    )
