"""Acceptance tests for built-in propagation channels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tencirpauli as tcp
from tencirpauli import advanced


def _engine(
    tape: advanced.GateTape,
    code: str,
    *,
    state: tcp.ProductBlochState | None = None,
) -> advanced.PropagationEngine:
    return advanced.PropagationEngine(
        tape,
        tcp.PauliOperator.from_terms(1, [(code, 1.0)]),
        initial_state=state or tcp.ProductBlochState(np.array([[0.31, -0.27, 0.44]])),
    )


def _kraus_expectation(code: str, kraus: list[np.ndarray], bloch: np.ndarray) -> float:
    paulis = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.diag([1.0, -1.0]).astype(np.complex128),
    }
    rho = (
        paulis["I"]
        + sum(value * paulis[label] for value, label in zip(bloch, ("X", "Y", "Z")))
    ) / 2.0
    transformed = sum(
        (operator.conj().T @ paulis[code] @ operator for operator in kraus)
    )
    return float(np.trace(rho @ transformed).real)


def test_builtin_channels_match_independent_kraus_references() -> None:
    bloch = np.array([0.31, -0.27, 0.44], dtype=np.float64)
    state = tcp.ProductBlochState(bloch[None, :])
    px, py, pz = 0.11, 0.07, 0.13
    channels = [
        (
            "depolarizing",
            {"px": px, "py": py, "pz": pz},
            [
                np.sqrt(1 - px - py - pz) * np.eye(2),
                np.sqrt(px) * np.array([[0, 1], [1, 0]]),
                np.sqrt(py) * np.array([[0, -1j], [1j, 0]]),
                np.sqrt(pz) * np.diag([1.0, -1.0]),
            ],
        ),
        (
            "phasedamping",
            {"gamma": 0.23},
            [
                np.diag([1.0, np.sqrt(1 - 0.23)]),
                np.diag([0.0, np.sqrt(0.23)]),
            ],
        ),
        (
            "amplitudedamping",
            {"gamma": 0.29, "p": 0.63},
            [
                np.sqrt(0.63) * np.diag([1.0, np.sqrt(1 - 0.29)]),
                np.sqrt(0.63) * np.array([[0, np.sqrt(0.29)], [0, 0]]),
                np.sqrt(1 - 0.63) * np.diag([np.sqrt(1 - 0.29), 1.0]),
                np.sqrt(1 - 0.63) * np.array([[0, 0], [np.sqrt(0.29), 0]]),
            ],
        ),
    ]
    for name, values, kraus in channels:
        tape = advanced.GateTape(1)
        getattr(tape, name)(0, **values)
        for code in ("I", "X", "Y", "Z"):
            expected = _kraus_expectation(code, kraus, bloch)
            assert _engine(tape, code, state=state).expectation([]) == pytest.approx(
                expected, abs=1.0e-11, rel=1.0e-10
            )


def test_channel_gradients_match_finite_difference_and_boundary_contract() -> None:
    state = tcp.ProductBlochState(np.array([[0.31, -0.27, 0.44]], dtype=np.float64))
    cases = [
        ("depolarizing", (0.11, 0.07, 0.13), (1.0e-6, 1.0e-6, 1.0e-6)),
        ("phasedamping", (0.23,), (1.0e-6,)),
        ("amplitudedamping", (0.29, 0.63), (1.0e-6, 1.0e-6)),
    ]
    for name, values, steps in cases:
        tape = advanced.GateTape(1)
        if name == "depolarizing":
            tape.depolarizing(0, px_parameter=0, py_parameter=1, pz_parameter=2)
        elif name == "phasedamping":
            tape.phasedamping(0, gamma_parameter=0)
        else:
            tape.amplitudedamping(0, gamma_parameter=0, p_parameter=1)
        engine = _engine(tape, "X" if name != "amplitudedamping" else "Z", state=state)
        analytic = engine.value_and_grad(values).gradient
        finite = []
        for index, step in enumerate(steps):
            plus = list(values)
            minus = list(values)
            plus[index] += step
            minus[index] -= step
            finite.append(
                (engine.expectation(plus) - engine.expectation(minus)) / (2 * step)
            )
        np.testing.assert_allclose(analytic, finite, atol=1.0e-6, rtol=1.0e-5)

    zero = advanced.GateTape(1)
    zero.depolarizing(0, px_parameter=0, py_parameter=1, pz_parameter=2)
    assert _engine(zero, "X", state=state).value_and_grad(
        [0.2, 0.4, 0.1]
    ).gradient.tolist() == [
        0.0,
        0.0,
        0.0,
    ]
    endpoint = advanced.GateTape(1)
    endpoint.phasedamping(0, gamma_parameter=0)
    assert np.isfinite(
        _engine(endpoint, "Z", state=state).value_and_grad([1.0]).gradient
    ).all()


def test_high_level_channel_layout_qir_batch_and_concurrency() -> None:
    circuit = tcp.PropagationCircuit(2)
    circuit.ry(0, theta=0.17)
    circuit.depolarizing(0, px=0.11, py=0.07, pz=0.13)
    circuit.rzz(0, 1, theta=-0.23)
    circuit.amplitudedamping(1, gamma=0.29, p=0.63)
    assert circuit.angle_count == 2
    assert circuit.channel_parameter_count == 5
    assert tuple(item.parameter_name for item in circuit.channel_parameter_layout) == (
        "px",
        "py",
        "pz",
        "gamma",
        "p",
    )
    observable = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0), ("XI", 0.2)])
    result = circuit.value_and_grad(observable)
    assert result.gradient.shape == (2,)
    assert result.channel_gradient.shape == (5,)
    assert not result.gradient.flags.writeable
    assert not result.channel_gradient.flags.writeable
    restored = tcp.PropagationCircuit.from_qir(circuit.to_qir(), {"nqubits": 2})
    assert restored.expectation(observable) == pytest.approx(
        circuit.expectation(observable)
    )

    batch = advanced.GateTape(1)
    batch.phasedamping(0, gamma_parameter=0)
    observables = [
        tcp.PauliOperator.from_terms(1, [(code, 1.0)]) for code in ("X", "Z")
    ]
    plan = tcp.PropagationBatch(batch, observables)
    values = [0.23]
    expected = plan.values_and_gradients(values)
    with ThreadPoolExecutor(max_workers=3) as executor:
        repeated = list(
            executor.map(lambda _: plan.values_and_gradients(values), range(3))
        )
    for item in repeated:
        np.testing.assert_array_equal(item.values, expected.values)
        np.testing.assert_array_equal(item.gradients, expected.gradients)


def test_channel_validation_qir_and_spps_rejection() -> None:
    with pytest.raises(ValueError, match="sum"):
        advanced.GateTape(1).depolarizing(0, px=0.7, py=0.3, pz=0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tcp.PropagationCircuit(1).phasedamping(0, gamma=1.1)
    qir = [
        {
            "name": "phase_damping",
            "index": (0,),
            "is_channel": True,
            "channel_parameters": {"gamma": 0.2},
            "channel_unitary": False,
            "channel_f": lambda _: (_ for _ in ()).throw(AssertionError()),
        }
    ]
    circuit = tcp.PropagationCircuit.from_qir(qir, {"nqubits": 1})
    assert circuit.to_qir()[0]["name"] == "phase_damping"
    with pytest.raises(ValueError, match="channel_unitary"):
        tcp.PropagationCircuit.from_qir(
            [
                {
                    "name": "phase_damping",
                    "index": (0,),
                    "is_channel": True,
                    "channel_parameters": {"gamma": 0.2},
                }
            ],
            {"nqubits": 1},
        )
    spps = tcp.SPPSCircuit(1)
    with pytest.raises(ValueError, match="does not support"):
        spps.phasedamping(0, gamma=0.2)
