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
@pytest.mark.parametrize("backend", ("built_in", "static_ptm"))
def test_channel_setup(benchmark: BenchmarkFixture, name: str, backend: str) -> None:
    channel_tape, static_tape, _parameters = _channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("Z", -0.4)])
    tape = channel_tape if backend == "built_in" else static_tape
    result = benchmark(advanced.PropagationEngine, tape, observable)
    assert result.nqubits == 1
    benchmark.extra_info.update(
        channel=name, backend=backend, observable_terms=len(observable.terms)
    )


@pytest.mark.parametrize("name", ("depolarizing", "phasedamping", "amplitudedamping"))
@pytest.mark.parametrize("backend", ("built_in", "static_ptm"))
def test_channel_first_and_steady_value(
    benchmark: BenchmarkFixture, name: str, backend: str
) -> None:
    channel_tape, static_tape, parameters = _channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 1.0), ("Z", -0.4)])
    tape = channel_tape if backend == "built_in" else static_tape
    runtime_parameters = parameters if backend == "built_in" else np.empty(0)
    engine = advanced.PropagationEngine(tape, observable, max_weight=1)
    expected = engine.expectation(runtime_parameters)
    first = engine.expectation(runtime_parameters)
    assert first == pytest.approx(expected, abs=1.0e-12)
    steady = benchmark.pedantic(
        engine.expectation,
        args=(runtime_parameters,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert steady == pytest.approx(expected, abs=1.0e-12)
    benchmark.extra_info.update(
        channel=name,
        backend=backend,
        first_expectation=float(first),
        steady_expectation=float(steady),
        scalar_buffer_bytes=int(runtime_parameters.nbytes),
        observable_terms=len(observable.terms),
        max_weight=1,
    )


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
    benchmark.extra_info.update(
        channel=name,
        scalar_buffer_bytes=int(parameters.nbytes),
        observable_terms=len(observable.terms),
        gradient_count=int(result.gradient.size),
    )


def _rotation_channel_pair(
    name: str,
) -> tuple[advanced.GateTape, advanced.GateTape, np.ndarray, np.ndarray]:
    built_in = advanced.GateTape(1)
    static = advanced.GateTape(1)
    built_in.ry(0, parameter=0)
    static.ry(0, parameter=0)
    if name == "depolarizing":
        channel_values = np.asarray([0.11, 0.07, 0.13], dtype=np.float64)
        built_in.depolarizing(0, px_parameter=1, py_parameter=2, pz_parameter=3)
        static.ptm((0,), _depolarizing_ptm(*channel_values))
    elif name == "phasedamping":
        channel_values = np.asarray([0.23], dtype=np.float64)
        built_in.phasedamping(0, gamma_parameter=1)
        static.ptm((0,), _phase_damping_ptm(float(channel_values[0])))
    else:
        channel_values = np.asarray([0.29, 0.63], dtype=np.float64)
        built_in.amplitudedamping(0, gamma_parameter=1, p_parameter=2)
        static.ptm((0,), _amplitude_damping_ptm(*channel_values))
    return (
        built_in,
        static,
        np.concatenate(([0.17], channel_values)),
        np.asarray([0.17]),
    )


@pytest.mark.parametrize("name", ("depolarizing", "phasedamping", "amplitudedamping"))
@pytest.mark.parametrize("backend", ("built_in", "static_ptm"))
def test_rotation_gradient_through_channel_or_ptm(
    benchmark: BenchmarkFixture, name: str, backend: str
) -> None:
    built_in, static, built_parameters, static_parameters = _rotation_channel_pair(name)
    observable = tcp.PauliOperator.from_terms(1, [("X", 0.7), ("Z", -0.4)])
    tape = built_in if backend == "built_in" else static
    parameters = built_parameters if backend == "built_in" else static_parameters
    engine = advanced.PropagationEngine(
        tape,
        observable,
        initial_state=tcp.ProductBlochState(np.asarray([[0.31, -0.27, 0.44]])),
    )
    expected = engine.value_and_grad(parameters)
    result = benchmark.pedantic(
        engine.value_and_grad,
        args=(parameters,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert result.value == pytest.approx(expected.value, abs=1.0e-12)
    np.testing.assert_allclose(result.gradient, expected.gradient)
    benchmark.extra_info.update(
        channel=name,
        backend=backend,
        scalar_buffer_bytes=int(parameters.nbytes),
        rotation_gradient=float(result.gradient[0]),
        gradient_count=int(result.gradient.size),
    )


def _mixed_channel_tapes(
    nqubits: int,
) -> tuple[advanced.GateTape, advanced.GateTape, np.ndarray, np.ndarray, list[int]]:
    built_in = advanced.GateTape(nqubits)
    static = advanced.GateTape(nqubits)
    parameters: list[float] = []
    angle_slots: list[int] = []
    for layer in range(2):
        for wire in range(nqubits):
            angle_slot = len(parameters)
            angle_slots.append(angle_slot)
            parameters.append(0.11 + 0.01 * layer + 0.001 * wire)
            built_in.ry(wire, parameter=angle_slot)
            static.ry(wire, parameter=len(angle_slots) - 1)
            if wire % 3 == 0:
                gamma_slot = len(parameters)
                parameters.append(0.21)
                built_in.phasedamping(wire, gamma_parameter=gamma_slot)
                static.ptm((wire,), _phase_damping_ptm(0.21))
            elif wire % 3 == 1:
                px_slot = len(parameters)
                parameters.extend([0.02, 0.03, 0.04])
                built_in.depolarizing(
                    wire,
                    px_parameter=px_slot,
                    py_parameter=px_slot + 1,
                    pz_parameter=px_slot + 2,
                )
                static.ptm((wire,), _depolarizing_ptm(0.02, 0.03, 0.04))
            else:
                gamma_slot = len(parameters)
                parameters.extend([0.27, 0.61])
                built_in.amplitudedamping(
                    wire, gamma_parameter=gamma_slot, p_parameter=gamma_slot + 1
                )
                static.ptm((wire,), _amplitude_damping_ptm(0.27, 0.61))
        for wire in range(0, nqubits - 1, 2):
            built_in.cnot(wire, wire + 1)
            static.cnot(wire, wire + 1)
    fixed_ptm = np.diag([1.0, 0.93, 0.87, 0.91]).astype(np.float64)
    built_in.ptm((0,), fixed_ptm)
    static.ptm((0,), fixed_ptm)
    return (
        built_in,
        static,
        np.asarray(parameters, dtype=np.float64),
        np.asarray([parameters[index] for index in angle_slots], dtype=np.float64),
        angle_slots,
    )


def test_mixed_channel_value_and_gradient(benchmark: BenchmarkFixture) -> None:
    nqubits = 12
    tape, static_tape, parameters, angle_parameters, angle_slots = _mixed_channel_tapes(
        nqubits
    )
    observable = _observable(nqubits)
    engine = advanced.PropagationEngine(tape, observable, max_weight=3)
    static_engine = advanced.PropagationEngine(static_tape, observable, max_weight=3)
    expected = engine.value_and_grad(parameters)
    static_expected = static_engine.value_and_grad(angle_parameters)
    assert expected.value == pytest.approx(static_expected.value, abs=1.0e-12)
    np.testing.assert_allclose(expected.gradient[angle_slots], static_expected.gradient)
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
        peak_term_count=engine.profile(parameters).profile.peak_term_count,
        estimated_peak_bytes=engine.profile(parameters).profile.estimated_peak_bytes,
        channel_parameter_count=int(parameters.size - angle_parameters.size),
        angle_parameter_count=int(angle_parameters.size),
        maximum_angle_error=float(
            np.max(np.abs(expected.gradient[angle_slots] - static_expected.gradient))
        ),
    )


def test_propagation_batch_channels_value_and_gradient(
    benchmark: BenchmarkFixture,
) -> None:
    nqubits = 4
    tape = advanced.GateTape(nqubits)
    tape.ry(0, parameter=0)
    tape.phasedamping(0, gamma_parameter=1)
    tape.amplitudedamping(1, gamma_parameter=2, p_parameter=3)
    tape.depolarizing(2, px_parameter=4, py_parameter=5, pz_parameter=6)
    tape.rzz(2, 3, parameter=7)
    observables = [
        tcp.PauliOperator.from_terms(nqubits, [("XIII", 0.7), ("ZIII", -0.2)])
        for _ in range(16)
    ]
    batch = tcp.PropagationBatch(tape, observables, max_weight=3)
    parameters = np.asarray(
        [0.17, 0.23, 0.29, 0.61, 0.02, 0.03, 0.04, -0.11], dtype=np.float64
    )
    expected_values = np.asarray(
        [
            advanced.PropagationEngine(tape, observable, max_weight=3).expectation(
                parameters
            )
            for observable in observables
        ]
    )
    expected_gradients = np.asarray(
        [
            advanced.PropagationEngine(tape, observable, max_weight=3)
            .value_and_grad(parameters)
            .gradient
            for observable in observables
        ]
    )
    values = batch.expectations(parameters)
    value_and_grad = batch.values_and_gradients(parameters, checkpoint_interval=2)
    np.testing.assert_allclose(values, expected_values, atol=1.0e-12)
    np.testing.assert_allclose(value_and_grad.values, expected_values, atol=1.0e-12)
    np.testing.assert_allclose(
        value_and_grad.gradients, expected_gradients, atol=1.0e-12
    )
    result = benchmark.pedantic(
        batch.values_and_gradients,
        args=(parameters,),
        kwargs={"checkpoint_interval": 2},
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    np.testing.assert_allclose(result.values, expected_values, atol=1.0e-12)
    np.testing.assert_allclose(result.gradients, expected_gradients, atol=1.0e-12)
    benchmark.extra_info.update(
        nqubits=nqubits,
        observable_count=len(observables),
        scalar_buffer_bytes=int(parameters.nbytes),
        maximum_value_error=float(
            np.max(np.abs(value_and_grad.values - expected_values))
        ),
        maximum_gradient_error=float(
            np.max(np.abs(value_and_grad.gradients - expected_gradients))
        ),
    )


def test_propagation_batch_channels_values(benchmark: BenchmarkFixture) -> None:
    nqubits = 4
    tape = advanced.GateTape(nqubits)
    tape.ry(0, parameter=0)
    tape.phasedamping(0, gamma_parameter=1)
    tape.amplitudedamping(1, gamma_parameter=2, p_parameter=3)
    tape.depolarizing(2, px_parameter=4, py_parameter=5, pz_parameter=6)
    tape.rzz(2, 3, parameter=7)
    observables = [
        tcp.PauliOperator.from_terms(nqubits, [("XIII", 0.7), ("ZIII", -0.2)])
        for _ in range(16)
    ]
    batch = tcp.PropagationBatch(tape, observables, max_weight=3)
    parameters = np.asarray(
        [0.17, 0.23, 0.29, 0.61, 0.02, 0.03, 0.04, -0.11], dtype=np.float64
    )
    expected = batch.expectations(parameters)
    result = benchmark.pedantic(
        batch.expectations,
        args=(parameters,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    np.testing.assert_allclose(result, expected, atol=1.0e-12)
    benchmark.extra_info.update(
        nqubits=nqubits,
        observable_count=len(observables),
        scalar_buffer_bytes=int(parameters.nbytes),
        maximum_value_error=float(np.max(np.abs(result - expected))),
    )


def test_jax_noise_aware_scalar_objective(benchmark: BenchmarkFixture) -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    observable = tcp.PauliOperator.from_terms(2, [("ZZ", 1.0), ("XI", 0.2)])
    state = tcp.ProductBlochState(
        np.asarray([[0.31, -0.27, 0.44], [-0.22, 0.35, 0.51]], dtype=np.float64)
    )

    def objective(values: object) -> object:
        circuit = tcp.PropagationCircuit(2, initial_state=state)
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
    jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready(), (warm_value, warm_gradient)
    )

    def execute() -> tuple[object, object]:
        result = runner(point)
        return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), result)

    value, gradient = benchmark(execute)
    point_numpy = np.asarray(point, dtype=np.float64)

    def concrete(values: np.ndarray) -> float:
        def sigmoid(item: float) -> float:
            return 1.0 / (1.0 + np.exp(-item))

        circuit = tcp.PropagationCircuit(2, initial_state=state)
        circuit.ry(0, theta=values[0])
        circuit.depolarizing(
            0,
            px=0.05 * sigmoid(values[1]),
            py=0.03 * sigmoid(values[2]),
            pz=0.02 * sigmoid(values[3]),
        )
        circuit.amplitudedamping(1, gamma=sigmoid(values[4]), p=sigmoid(values[5]))
        return circuit.expectation(observable)

    finite = np.empty(point_numpy.size, dtype=np.float64)
    step = 1.0e-6
    for index in range(point_numpy.size):
        plus = point_numpy.copy()
        minus = point_numpy.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (concrete(plus) - concrete(minus)) / (2.0 * step)
    value_error = abs(float(np.asarray(value)) - concrete(point_numpy))
    gradient_error = float(np.max(np.abs(np.asarray(gradient) - finite)))
    assert np.isfinite(value_error)
    assert np.isfinite(gradient_error)
    assert value_error < 1.0e-10
    assert gradient_error < 5.0e-6
    benchmark.extra_info.update(
        nqubits=2,
        scalar_buffer_bytes=int(point.size * 8),
        observable_terms=len(observable.terms),
        value_error=value_error,
        gradient_error=gradient_error,
    )
