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


@pytest.mark.parametrize("code, component", (("X", 0), ("Y", 1)))
def test_phase_damping_gamma_one_drops_transverse_edges(
    code: str, component: int
) -> None:
    bloch = np.zeros((1, 3), dtype=np.float64)
    bloch[0, component] = 0.37
    tape = advanced.GateTape(1)
    tape.phasedamping(0, gamma_parameter=0)
    result = _engine(
        tape,
        code,
        state=tcp.ProductBlochState(bloch),
    ).value_and_grad([1.0])
    assert result.value == 0.0
    assert result.gradient[0] == 0.0
    assert np.isfinite(result.gradient).all()


@pytest.mark.parametrize("code, component", (("X", 0), ("Y", 1)))
def test_amplitude_damping_gamma_one_drops_transverse_edges(
    code: str, component: int
) -> None:
    bloch = np.zeros((1, 3), dtype=np.float64)
    bloch[0, component] = -0.29
    tape = advanced.GateTape(1)
    tape.amplitudedamping(0, gamma_parameter=0, p_parameter=1)
    result = _engine(
        tape,
        code,
        state=tcp.ProductBlochState(bloch),
    ).value_and_grad([1.0, 0.63])
    np.testing.assert_allclose(result.value, 0.0, atol=0.0)
    np.testing.assert_array_equal(result.gradient, [0.0, 0.0])
    assert np.isfinite(result.gradient).all()


@pytest.mark.parametrize("p", (0.0, 0.5, 0.63, 1.0))
def test_amplitude_damping_z_split_has_frozen_support_derivatives(p: float) -> None:
    z = 0.44
    gamma = 0.37
    tape = advanced.GateTape(1)
    tape.amplitudedamping(0, gamma_parameter=0, p_parameter=1)
    result = _engine(
        tape,
        "Z",
        state=tcp.ProductBlochState(np.asarray([[0.0, 0.0, z]])),
    ).value_and_grad([gamma, p])
    expected_value = gamma * (2.0 * p - 1.0) + (1.0 - gamma) * z
    expected_gamma = (2.0 * p - 1.0) - z
    expected_p = 0.0 if p == 0.5 else 2.0 * gamma
    np.testing.assert_allclose(result.value, expected_value, atol=1.0e-12)
    np.testing.assert_allclose(
        result.gradient, [expected_gamma, expected_p], atol=1.0e-12
    )
    assert np.isfinite(result.gradient).all()


def test_amplitude_damping_split_cancellation_is_absent_from_reverse() -> None:
    tape = advanced.GateTape(1)
    tape.amplitudedamping(0, gamma_parameter=0, p_parameter=1)
    observable = tcp.PauliOperator.from_terms(1, [("I", 0.4), ("Z", 1.0)])
    engine = advanced.PropagationEngine(tape, observable)
    propagated = engine.propagate_operator([0.4, 0.0])
    assert [(term.word.to_codes(), term.coefficient) for term in propagated.terms] == [
        ((3,), 0.6 + 0.0j)
    ]
    result = engine.value_and_grad([0.4, 0.0])
    np.testing.assert_allclose(result.value, 0.6, atol=1.0e-12)
    np.testing.assert_allclose(result.gradient, [-1.0, 0.0], atol=1.0e-12)


def test_weight_projection_drops_channel_dependent_overweight_continuation() -> None:
    tape = advanced.GateTape(2)
    tape.rxx(0, 1, parameter=0)
    tape.amplitudedamping(0, gamma_parameter=1, p_parameter=2)
    observable = tcp.PauliOperator.from_terms(2, [("ZI", 1.0)])
    state = tcp.ProductBlochState(np.asarray([[0.0, 0.0, 0.44], [0.0, 0.0, 0.0]]))
    engine = advanced.PropagationEngine(
        tape, observable, initial_state=state, max_weight=1
    )
    theta, gamma, p = 0.41, 0.37, 0.73
    result = engine.value_and_grad([theta, gamma, p])
    expected_value = gamma * (2.0 * p - 1.0) + (1.0 - gamma) * np.cos(theta) * 0.44
    expected_gradient = [
        -(1.0 - gamma) * np.sin(theta) * 0.44,
        (2.0 * p - 1.0) - np.cos(theta) * 0.44,
        2.0 * gamma,
    ]
    np.testing.assert_allclose(result.value, expected_value, atol=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected_gradient, atol=1.0e-12)


def test_high_level_channel_gradient_arrays_follow_occurrence_order() -> None:
    values = np.asarray([0.13, 0.23, -0.21, 0.29, 0.63], dtype=np.float64)
    observable = tcp.PauliOperator.from_terms(1, [("X", 0.8), ("Z", -0.4)])

    def build(point: np.ndarray) -> tcp.PropagationCircuit:
        circuit = tcp.PropagationCircuit(
            1, initial_state=tcp.ProductBlochState(np.asarray([[0.31, -0.27, 0.44]]))
        )
        circuit.ry(0, theta=point[0])
        circuit.phasedamping(0, gamma=point[1])
        circuit.rx(0, theta=point[2])
        circuit.amplitudedamping(0, gamma=point[3], p=point[4])
        return circuit

    circuit = build(values)
    assert tuple(
        (item.operation_index, item.parameter_name)
        for item in circuit.channel_parameter_layout
    ) == ((1, "gamma"), (3, "gamma"), (3, "p"))
    result = circuit.value_and_grad(observable)
    finite = np.empty_like(values)
    step = 1.0e-6
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (
            build(plus).expectation(observable) - build(minus).expectation(observable)
        ) / (2.0 * step)
    np.testing.assert_allclose(
        result.gradient, finite[[0, 2]], atol=3.0e-6, rtol=2.0e-6
    )
    np.testing.assert_allclose(
        result.channel_gradient, finite[[1, 3, 4]], atol=3.0e-6, rtol=2.0e-6
    )


def test_channel_only_and_rotation_only_public_gradient_arrays_are_empty_and_read_only() -> (
    None
):
    observable = tcp.PauliOperator.from_terms(1, [("Z", 1.0)])
    channels = tcp.PropagationCircuit(1)
    channels.phasedamping(0, gamma=0.23)
    channel_result = channels.value_and_grad(observable)
    assert channel_result.gradient.shape == (0,)
    assert channel_result.gradient.flags.c_contiguous
    assert not channel_result.gradient.flags.writeable

    rotations = tcp.PropagationCircuit(1)
    rotations.ry(0, theta=0.19)
    rotation_result = rotations.value_and_grad(observable)
    assert rotation_result.channel_gradient.shape == (0,)
    assert rotation_result.channel_gradient.flags.c_contiguous
    assert not rotation_result.channel_gradient.flags.writeable


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
