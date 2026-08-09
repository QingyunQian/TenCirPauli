"""Record eager representation evidence for capability classes."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from typing import Any

import numpy as np

import tencirpauli as tcp


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def measure(
    name: str, operator: Any, *, cutoffs: dict[int, int] | None = None
) -> dict[str, Any]:
    """Measure one fixed eager plan and its authoritative CSR expansion."""
    kwargs = {} if cutoffs is None else {"boson_cutoffs": cutoffs}
    construction_start = time.perf_counter()
    if hasattr(operator, "mvp_plan"):
        plan = operator.mvp_plan(storage="eager")
        csr_owner = operator
        csr_kwargs: dict[str, Any] = {}
    else:
        plan = operator.compile("native_mvp", storage="eager", **kwargs)
        csr_owner = operator
        csr_kwargs = kwargs
    construction_seconds = time.perf_counter() - construction_start
    state = np.arange(plan.dimension, dtype=np.complex128) + 0.2j
    output = np.empty_like(state)
    for _ in range(2):
        plan.apply_into(state, output)
    first_start = time.perf_counter()
    plan.apply_into(state, output)
    first_seconds = time.perf_counter() - first_start
    steady_seconds = []
    for _ in range(7):
        start = time.perf_counter()
        plan.apply_into(state, output)
        steady_seconds.append(time.perf_counter() - start)
    csr_start = time.perf_counter()
    csr = (
        csr_owner.csr(**csr_kwargs)
        if hasattr(csr_owner, "csr")
        else csr_owner.compile("csr", **csr_kwargs)
    )
    csr_seconds = time.perf_counter() - csr_start
    csr_action = np.zeros_like(state)
    for row in range(plan.dimension):
        start, stop = int(csr.indptr[row]), int(csr.indptr[row + 1])
        csr_action[row] = np.dot(csr.data[start:stop], state[csr.indices[start:stop]])
    max_abs_error = (
        float(np.max(np.abs(output - csr_action))) if plan.dimension else 0.0
    )
    record = {
        "capability": name,
        "dimension": plan.dimension,
        "source_term_count": plan.source_term_count,
        "schedule_count": plan.schedule_count,
        "execution_family": plan.execution_family,
        "native_variant": getattr(plan, "native_variant", "unknown"),
        "retained_format": plan.retained_format,
        "transition_count": plan.transition_count,
        "estimated_retained_bytes": plan.estimated_retained_bytes,
        "estimated_construction_peak_bytes": plan.estimated_construction_peak_bytes,
        "csr_bytes": int(csr.indptr.nbytes + csr.indices.nbytes + csr.data.nbytes),
        "csr_nnz": int(csr.data.size),
        "construction_seconds": construction_seconds,
        "first_apply_seconds": first_seconds,
        "steady_apply_seconds": steady_seconds,
        "steady_apply_median_seconds": float(np.median(steady_seconds)),
        "csr_seconds": csr_seconds,
        "max_abs_error_vs_csr": max_abs_error,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    native_eager = getattr(plan, "_native_plan", None)
    if native_eager is not None and hasattr(native_eager, "construction_profile"):
        profile = native_eager.construction_profile()
        record["csr_construction_profile"] = {
            "generation_nanos": int(profile[0]),
            "chunk_count": int(profile[1]),
            "sorted_entries": int(profile[2]),
            "merged_entries": int(profile[3]),
            "heap_operations": int(profile[4]),
            "final_array_allocation_entries": int(profile[5]),
            "chunk_metadata_bytes": int(profile[6]),
            "sort_nanos": int(profile[7]),
            "merge_nanos": int(profile[8]),
            "final_array_allocation_nanos": int(profile[9]),
        }
    return record


def main() -> None:
    """Run bounded capability-class measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=8)
    args = parser.parse_args()
    pauli_terms = []
    for index in range(8):
        word = [0] * args.qubits
        word[0] = 1
        word[index] = 3 if index else 1
        pauli_terms.append((tuple(word), 0.25 + 0.01j * index))
    pauli = tcp.PauliOperator.from_terms(args.qubits, pauli_terms)
    space = tcp.OperatorSpace(bosons=2)
    cartesian = space.boson.create(0) * space.boson.annihilate(1)
    charge = tcp.AdditiveCharge(space, bosons={0: 1, 1: 1})
    product = (space.boson.create(0) * space.boson.annihilate(1)).restrict_charge(
        charge.sector(2, boson_cutoffs={0: 3, 1: 3})
    )
    qubit_space = tcp.OperatorSpace(qubits=4)
    qubit_charge = tcp.AdditiveCharge(
        qubit_space, qubits={index: (0, 1) for index in range(4)}
    )
    qubit_product = (
        qubit_space.qubit.x(0) * qubit_space.qubit.x(1)
        + qubit_space.qubit.y(0) * qubit_space.qubit.y(1)
    ).restrict_charge(qubit_charge.sector(2))
    qudit_space = tcp.OperatorSpace(qudits=(3, 3))
    qudit_charge = tcp.AdditiveCharge(qudit_space, qudits={0: 1, 1: 0})
    qudit_product = tcp.QuditWeylOperator.from_terms(
        3, [(((0, 0, 1),), 1.0)], n_sites=2
    ).restrict_charge(qudit_charge.sector(0))
    independent_space = tcp.OperatorSpace(qubits=2, bosons=1)
    independent_sector = tcp.ChargeSector(
        (
            (tcp.AdditiveCharge(independent_space, qubits={0: (0, 1), 1: (0, 1)}), 1),
            (tcp.AdditiveCharge(independent_space, bosons={0: 1}), 0),
        ),
        boson_cutoffs={0: 1},
    )
    independent_product = (
        independent_space.qubit.z(0)
        + independent_space.boson.create(0) * independent_space.boson.annihilate(0)
    ).restrict_charge(independent_sector)
    reversed_space = tcp.OperatorSpace(qubits=3, bosons=2)
    reversed_boson = tcp.AdditiveCharge(reversed_space, bosons={0: 1, 1: 1})
    reversed_qubit = tcp.AdditiveCharge(
        reversed_space, qubits={index: (0, 1) for index in range(3)}
    )
    reversed_sector = tcp.ChargeSector(
        ((reversed_boson, 1), (reversed_qubit, 1)), boson_cutoffs={0: 1, 1: 1}
    )
    reversed_product = (
        reversed_space.boson.create(0) * reversed_space.boson.annihilate(1)
        + reversed_space.boson.create(1) * reversed_space.boson.annihilate(0)
        + reversed_space.qubit.z(0)
    ).restrict_charge(reversed_sector)
    fermion_space = tcp.OperatorSpace(fermions=4)
    fermion_charge = tcp.AdditiveCharge(
        fermion_space, fermions={index: 1 for index in range(4)}
    )
    fermion_balance = tcp.AdditiveCharge(
        fermion_space, fermions={0: 1, 1: 1, 2: -1, 3: -1}
    )
    fermion_sector = tcp.ChargeSector(((fermion_charge, 2), (fermion_balance, 0)))
    fermion_product = (
        fermion_space.fermion.create(0) * fermion_space.fermion.annihilate(1)
        + fermion_space.fermion.create(1) * fermion_space.fermion.annihilate(0)
    ).restrict_charge(fermion_sector)
    generic_space = tcp.OperatorSpace(fermions=3, bosons=1)
    generic_total = tcp.AdditiveCharge(generic_space, fermions={0: 1, 1: 1, 2: 0})
    generic_balance = tcp.AdditiveCharge(generic_space, fermions={1: 1, 2: 1})
    generic_sector = tcp.ChargeSector(
        ((generic_total, 1), (generic_balance, 1)), boson_cutoffs={0: 1}
    )
    generic_operator = (
        generic_space.fermion.create(0) * generic_space.fermion.annihilate(0)
        + generic_space.fermion.create(1) * generic_space.fermion.annihilate(1)
        + generic_space.fermion.create(2) * generic_space.fermion.annihilate(2)
        + generic_space.boson.create(0) * generic_space.boson.annihilate(0)
    ).restrict_charge(generic_sector)
    records = [
        measure("pauli_xor", pauli),
        measure("cartesian_shift_stride", cartesian, cutoffs={0: 3, 1: 3}),
        measure("product_boson_factor", product),
        measure("product_qubit_factor", qubit_product),
        measure("product_qudit_factor", qudit_product),
        measure("product_independent_factors", independent_product),
        measure("product_reversed_factor_sizes", reversed_product),
        measure("product_compact_fermion_factor", fermion_product),
        measure("generic_irregular_csr", generic_operator),
    ]
    print(json.dumps({"records": records}, sort_keys=True))


if __name__ == "__main__":
    main()
