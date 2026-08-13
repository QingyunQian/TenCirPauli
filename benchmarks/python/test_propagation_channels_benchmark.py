"""Release-boundary benchmarks for built-in noise channels."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import tencirpauli as tcp
from tencirpauli import advanced


def _depolarizing_ptm(px: float, py: float, pz: float) -> np.ndarray:
    return np.diag(
        [1.0, 1.0 - 2.0 * (py + pz), 1.0 - 2.0 * (px + pz), 1.0 - 2.0 * (px + py)]
    ).astype(np.float64)


def _phase_damping_ptm(gamma: float) -> np.ndarray:
    return np.diag([1.0, np.sqrt(1.0 - gamma), np.sqrt(1.0 - gamma), 1.0]).astype(
        np.float64
    )


def _amplitude_damping_ptm(gamma: float, p: float) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = 1.0
    matrix[1, 1] = matrix[2, 2] = np.sqrt(1.0 - gamma)
    matrix[0, 3] = gamma * (2.0 * p - 1.0)
    matrix[3, 3] = 1.0 - gamma
    return matrix


def _observable(nqubits: int) -> tcp.PauliOperator:
    terms = []
    for wire in range(nqubits):
        codes = [0] * nqubits
        codes[wire] = 1 if wire % 2 == 0 else 3
        terms.append((codes, 0.1))
    return tcp.PauliOperator.from_terms(nqubits, terms)


def _channel_pair(name: str) -> tuple[advanced.GateTape, advanced.GateTape, np.ndarray]:
    channel = advanced.GateTape(1)
    static = advanced.GateTape(1)
    parameters: np.ndarray
    if name == "depolarizing":
        parameters = np.asarray([0.11, 0.07, 0.13], dtype=np.float64)
        channel.depolarizing(0, px_parameter=0, py_parameter=1, pz_parameter=2)
        static.ptm((0,), _depolarizing_ptm(*parameters))
    elif name == "phasedamping":
        parameters = np.asarray([0.23], dtype=np.float64)
        channel.phasedamping(0, gamma_parameter=0)
        static.ptm((0,), _phase_damping_ptm(float(parameters[0])))
    else:
        parameters = np.asarray([0.29, 0.63], dtype=np.float64)
        channel.amplitudedamping(0, gamma_parameter=0, p_parameter=1)
        static.ptm((0,), _amplitude_damping_ptm(*parameters))
    return channel, static, parameters


@pytest.mark.parametrize("name", ("depolarizing", "phasedamping", "amplitudedamping"))
def test_channel_setup(benchmark: BenchmarkFixture, name: str) -> None:
    channel_tape, _static_tape, _parameters = _channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("Z", -0.4)])
    result = benchmark(advanced.PropagationEngine, channel_tape, observable)
    assert result.nqubits == 1
    benchmark.extra_info.update(channel=name, observable_terms=len(observable.terms))


@pytest.mark.parametrize("name", ("depolarizing", "phasedamping", "amplitudedamping"))
def test_channel_setup_and_steady_value(benchmark: BenchmarkFixture, name: str) -> None:
    channel_tape, static_tape, parameters = _channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("Z", -0.4)])
    channel = advanced.PropagationEngine(channel_tape, observable, max_weight=1)
    control = advanced.PropagationEngine(static_tape, observable, max_weight=1)
    expected = control.expectation([])
    assert channel.expectation(parameters) == pytest.approx(expected, abs=1.0e-12)
    result = benchmark.pedantic(
        channel.expectation,
        args=(parameters,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert result == pytest.approx(expected, abs=1.0e-12)
    benchmark.extra_info.update(
        channel=name,
        scalar_buffer_bytes=int(parameters.nbytes),
        observable_terms=len(observable.terms),
        max_weight=1,
    )


@pytest.mark.parametrize("name", ("depolarizing", "phasedamping", "amplitudedamping"))
def test_channel_first_execution(benchmark: BenchmarkFixture, name: str) -> None:
    channel_tape, _static_tape, parameters = _channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("Z", -0.4)])
    engine = advanced.PropagationEngine(channel_tape, observable)
    result = benchmark.pedantic(
        engine.expectation, args=(parameters,), rounds=1, iterations=1, warmup_rounds=0
    )
    assert np.isfinite(result)


@pytest.mark.parametrize("name", ("depolarizing", "phasedamping", "amplitudedamping"))
def test_channel_value_and_gradient(benchmark: BenchmarkFixture, name: str) -> None:
    channel_tape, _static_tape, parameters = _channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("Z", -0.4)])
    engine = advanced.PropagationEngine(channel_tape, observable)
    expected = engine.value_and_grad(parameters)
    result = benchmark.pedantic(
        engine.value_and_grad,
        args=(parameters,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert result.value == pytest.approx(expected.value)
    np.testing.assert_allclose(result.gradient, expected.gradient)


def test_mixed_channel_value_and_gradient(benchmark: BenchmarkFixture) -> None:
    nqubits = 12
    tape = advanced.GateTape(nqubits)
    for layer in range(2):
        for wire in range(nqubits):
            tape.ry(wire, parameter=layer % 2)
            tape.phasedamping(wire, gamma_parameter=2 + wire % 2)
        for wire in range(0, nqubits - 1, 2):
            tape.cnot(wire, wire + 1)
    observable = _observable(nqubits)
    parameters = np.asarray([0.13, -0.21, 0.23, 0.31], dtype=np.float64)
    engine = advanced.PropagationEngine(tape, observable, max_weight=3)
    expected = engine.value_and_grad(parameters)
    result = benchmark.pedantic(
        engine.value_and_grad,
        args=(parameters,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert result.value == pytest.approx(expected.value)
    np.testing.assert_allclose(result.gradient, expected.gradient)
    benchmark.extra_info.update(
        nqubits=nqubits,
        scalar_buffer_bytes=int(parameters.nbytes),
        observable_terms=len(observable.terms),
        final_term_count=len(engine.propagate_operator(parameters).terms),
    )


def test_jax_noise_aware_scalar_objective(benchmark: BenchmarkFixture) -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0), ("XI", 0.2)])

    def objective(values: object) -> object:
        circuit = tcp.PropagationCircuit(2)
        circuit.ry(0, theta=values[0])  # type: ignore[index]
        circuit.depolarizing(
            0,
            px=0.05 * jax.nn.sigmoid(values[1]),  # type: ignore[index]
            py=0.03 * jax.nn.sigmoid(values[2]),  # type: ignore[index]
            pz=0.02 * jax.nn.sigmoid(values[3]),  # type: ignore[index]
        )
        circuit.amplitudedamping(
            1,
            gamma=jax.nn.sigmoid(values[4]),  # type: ignore[index]
            p=jax.nn.sigmoid(values[5]),  # type: ignore[index]
        )
        return circuit.expectation_jax(observable)

    runner = jax.jit(jax.value_and_grad(objective))
    point = jnp.asarray([0.13, -0.21, 0.07, 0.11, -0.17, 0.31])
    warm_value, warm_gradient = runner(point)
    warm_value.block_until_ready()
    warm_gradient.block_until_ready()

    def execute() -> tuple[object, object]:
        value, gradient = runner(point)
        value.block_until_ready()
        gradient.block_until_ready()
        return value, gradient

    value, gradient = benchmark(execute)
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(gradient)).all()
    benchmark.extra_info.update(
        nqubits=2,
        scalar_buffer_bytes=int(point.size * 8),
        observable_terms=len(observable.terms),
    )
