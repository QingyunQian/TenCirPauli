# Phase 12 Common Noise Channels and Native Scalar Gradients Specification

Status: implemented; focused remediation complete. The remediation evidence is recorded in `../reports/phase-12-review-2026-08-13.md`, the source, `tests/test_common_noise_channels.py`, `tests/test_propagation_gradient.py`, `tests/test_circuit_jax.py`, and `benchmarks/python/test_propagation_channels_benchmark.py`; the project environment passed the JAX oracle tests and release benchmark record `20260813T052717Z_c42a754cca87-dirty`. This phase is the propagation/noise successor to the completed Phase 11 factorized additive-charge work; it does not replace the existing Phase 11 specification.

## 1. Purpose and scope

Phase 12 completes the deterministic Pauli propagation path for a focused set of common one-qubit noise channels. It adds TensorCircuit-compatible channel constructors, native forward kernels, native gradients with respect to the supported channel scalars, QIR round trips, and JAX `jit`/`grad` composition through the existing coarse callback.

The existing static real one- and two-qubit PTM path remains supported for forward propagation. A static PTM continues to participate in the reverse transpose action needed to differentiate surrounding rotations and channels, but this phase does not differentiate PTM entries, does not add runtime PTM buffers, and does not add a parameterized PTM API.

The phase prioritizes observable-first Heisenberg propagation. Generic static PTMs remain finite real linear Pauli maps and may be non-trace-preserving; they return the existing unnormalized linear expectation. Imaginary-time propagation, a Schrödinger Pauli-state API, normalized non-trace-preserving terminals, Kraus-level sampling, and noise or PTM support in SPPS remain out of scope.

## 2. Binding decisions

1. Existing `GateTape.ptm(wires, matrix)` and `PropagationCircuit.ptm(wires, matrix)` retain their current static one- and two-qubit forward semantics. Their entries never appear in a returned gradient.
2. Static PTMs retain the Hermitian Pauli basis ordered as local `I/X/Y/Z` digits and the row-major layout `R[out * d + in]`, where `d=4^k`. For two wires, the local index is `4 * code(wires[0]) + code(wires[1])`.
3. No `parameterized_ptm`, `ptm_values`, `ptm_gradient`, runtime PTM metadata, or PTM-entry custom VJP is added in this phase.
4. The supported public channel methods and parameter names match TensorCircuit: `depolarizing(px, py, pz)`, `phasedamping(gamma)`, and `amplitudedamping(gamma, p)`.
5. Built-in channels have native kernels. Python records the logical operation and friendly scalar values; it does not rebuild a PTM on every execution.
6. Rotation angles and channel scalars use one native scalar-parameter buffer. Advanced `GateTape` returns one gradient in explicit slot order. High-level `PropagationCircuit` exposes separate angle and channel-gradient vectors as defined in Section 3.4 while retaining one coarse native call.
7. Channel gradients retain the existing deterministic frozen-support contract. Exact-zero local branches, outputs removed by exact aggregation, and words removed by `max_weight` projection do not enter the reverse pass.
8. At ordinary support-stable parameter points, the native gradient must agree with finite differences of the executed projected forward map. At support-change points such as an exact-zero multiplier or exact cancellation, the frozen-support result is the contract and no finite-difference equality is claimed.
9. The Rust core remains independent of TensorCircuit and JAX. TensorCircuit compatibility is expressed by the public Python spelling, channel mathematics, and QIR schema.
10. SPPS rejects all built-in channel operations in this phase. A deterministic local channel rule is not automatically a valid stochastic Pauli-path sampling rule.

## 3. Public API

### 3.1 Advanced `GateTape`

`GateTape` remains the typed advanced builder for Schrödinger-order tapes. Existing rotation slots keep their current integer slot contract. The following methods are added:

```python
tape.depolarizing(
    wire,
    *,
    px=None,
    py=None,
    pz=None,
    px_parameter=None,
    py_parameter=None,
    pz_parameter=None,
) -> None
tape.phasedamping(
    wire,
    *,
    gamma=None,
    gamma_parameter=None,
) -> None
tape.amplitudedamping(
    wire,
    *,
    gamma=None,
    p=None,
    gamma_parameter=None,
    p_parameter=None,
) -> None
```

For each named channel scalar, exactly one concrete value or non-negative runtime slot is required. For example, `px` and `px_parameter` are mutually exclusive and one must be supplied. Static values are validated when the tape is constructed. Runtime values are validated on every execution. Different channel scalars and rotations may share an advanced slot; all retained contributions to that slot are accumulated deterministically.

`nparameters` remains one plus the largest used scalar slot, with the existing requirement that slots cover `0..nparameters-1` without holes. No PTM-related count or execution buffer is added.

### 3.2 `PropagationEngine`

`PropagationEngine.expectation`, `value_and_grad`, `propagate_operator`, and `profile` retain their current signatures. Their `parameters` input now supplies both rotation slots and supported channel-scalar slots.

`PropagationValueAndGradient` remains:

```text
(value, gradient)
```

`gradient` is one read-only contiguous `float64[nparameters]` vector in advanced slot order. Static PTM entries never occupy positions in this vector. A static PTM still applies its fixed transpose during reverse propagation so gradients of earlier rotations or channels remain correct under the existing frozen-support contract.

### 3.3 `PropagationBatch`

`PropagationBatch.expectations` and `values_and_gradients` retain their current signatures. `PropagationBatchValueAndGradient.gradients` remains a read-only contiguous array with shape `(observable_count, nparameters)`, now including supported channel slots in the same advanced slot order. Empty batches retain shapes `(0,)` for values and `(0, nparameters)` for gradients.

The batch path continues to share one immutable compiled tape and to keep independent per-observable propagation state. No PTM-gradient output or shared aggregation state is introduced.

### 3.4 `PropagationCircuit` angle and channel gradients

The high-level facade mirrors TensorCircuit's channel method names and parameter names:

```python
circuit.depolarizing(wire, *, px, py, pz) -> None
circuit.phasedamping(wire, *, gamma) -> None
circuit.amplitudedamping(wire, *, gamma, p) -> None
```

Each argument may be a concrete finite real scalar or a scalar real JAX value under the same rules as an existing `theta`. Concrete terminals reject stored JAX values and direct the user to `expectation_jax()`.

`PropagationCircuit.value_and_grad()` returns a high-level result that preserves the existing `value` and `gradient` attributes and adds a separate channel-gradient vector:

```python
@dataclass(frozen=True)
class PropagationCircuitValueAndGradient:
    value: float
    gradient: np.ndarray
    channel_gradient: np.ndarray

    @property
    def angle_gradient(self) -> np.ndarray:
        return self.gradient
```

`gradient` remains the read-only contiguous rotation-angle gradient in rotation occurrence order, so `len(result.gradient) == circuit.angle_count` remains true for circuits with or without channels. `channel_gradient` is a read-only contiguous vector produced by scanning channel operations in append order and adding each channel's scalars in TensorCircuit signature order:

| Operation | Channel-gradient positions contributed by that occurrence |
| --- | --- |
| `depolarizing` | `px`, `py`, `pz` |
| `phasedamping` | `gamma` |
| `amplitudedamping` | `gamma`, `p` |

For example:

```python
circuit.ry(0, theta=theta0)
circuit.depolarizing(0, px=px, py=py, pz=pz)
circuit.rzz(0, 1, theta=theta1)
circuit.amplitudedamping(1, gamma=gamma, p=p)
```

returns:

```text
gradient         = [dtheta0, dtheta1]
channel_gradient = [dpx, dpy, dpz, dgamma, dp]
```

The existing `angle_count` remains the number of rotation occurrences. `channel_parameter_count` is the total number of channel-scalar occurrences and equals `len(result.channel_gradient)`. `parameter_count` is their sum.

A circuit without rotations returns `gradient.shape == (0,)`; a circuit without channels returns `channel_gradient.shape == (0,)`. Both arrays remain C-contiguous, read-only `float64` arrays, including their empty forms.

The facade exposes a read-only `channel_parameter_layout: tuple[PropagationChannelParameter, ...]`, where the public metadata record is:

```python
@dataclass(frozen=True)
class PropagationChannelParameter:
    operation_index: int
    operation: str
    wires: tuple[int, ...]
    parameter_name: str
```

The tuple has exactly `channel_parameter_count` entries and is the authoritative public meaning of each `channel_gradient` position. Repeated names are allowed because operation index and wires identify the occurrence. `PropagationChannelParameter` is exported beside `PropagationCircuit`; it contains metadata only and never owns a native buffer or numerical value.

The high-level facade privately lowers every rotation and channel scalar to an occurrence slot for `value_and_grad()` and `expectation_jax()`. Its private native slot vector is always `concat(rotation_values, channel_values)`: rotation occurrences are first in their existing order, followed by channel scalars in `channel_parameter_layout` order. The returned native gradient is split at `angle_count` into the two public read-only vectors without changing the Rust core result contract. Ordinary forward terminals may retain concrete channel values in the compiled tape, matching the existing concrete-angle strategy.

### 3.5 QIR support

`PropagationCircuit.to_qir()` emits supported channels with TensorCircuit-compatible channel names and fields. Examples are:

```python
{
    "name": "depolarizing",
    "index": (0,),
    "is_channel": True,
    "channel_parameters": {"px": 0.1, "py": 0.02, "pz": 0.03},
    "channel_unitary": True,
}
{
    "name": "phase_damping",
    "index": (0,),
    "is_channel": True,
    "channel_parameters": {"gamma": 0.2},
    "channel_unitary": False,
}
{
    "name": "amplitude_damping",
    "index": (1,),
    "is_channel": True,
    "channel_parameters": {"gamma": 0.3, "p": 0.9},
    "channel_unitary": False,
}
```

`PropagationCircuit.from_qir()` accepts exactly the TensorCircuit QIR channel names `depolarizing`, `phase_damping`, and `amplitude_damping` with the parameter dictionaries shown above. Public Python method names remain `depolarizing`, `phasedamping`, and `amplitudedamping`; no additional QIR aliases are accepted. QIR channel values must be concrete finite real scalars; JAX tracers are not serialized.

For a supported channel record, `is_channel` is required and must be `True`. `channel_unitary` is required and emitted by `to_qir()` as shown above; `from_qir()` accepts either boolean value because TensorCircuit's `Circuit` and `DMCircuit` use that field as an execution hint and do not emit the same value for every equivalent channel record. The channel name and scalar dictionary, not this hint, determine the imported semantics. Missing parameters, extra parameters, wrong arity, invalid wires, unsupported channels, a missing or non-boolean `channel_unitary`, and invalid physical ranges fail explicitly. `channel_f` callables in external TensorCircuit QIR are ignored, never invoked, and never retained; the supported name and scalar dictionary are sufficient. `SPPSCircuit.from_qir()` rejects all channel entries supported here.

Automatic conversion of arbitrary TensorCircuit Kraus objects remains out of scope. This section requires conversion from the explicit supported QIR records, not execution of arbitrary external channel factories.

## 4. Channel semantics

All built-in channels act in the Heisenberg Pauli basis and match the TensorCircuit channel functions with the same parameter names.

| Constructor | `E^dagger(I)` | `E^dagger(X)` | `E^dagger(Y)` | `E^dagger(Z)` |
| --- | --- | --- | --- | --- |
| `depolarizing(px, py, pz)` | `I` | `(1-2(py+pz))X` | `(1-2(px+pz))Y` | `(1-2(px+py))Z` |
| `phasedamping(gamma)` | `I` | `sqrt(1-gamma)X` | `sqrt(1-gamma)Y` | `Z` |
| `amplitudedamping(gamma, p)` | `I` | `sqrt(1-gamma)X` | `sqrt(1-gamma)Y` | `gamma(2p-1)I + (1-gamma)Z` |

`depolarizing` is the Pauli mixture with probabilities `1-px-py-pz`, `px`, `py`, and `pz` for `I`, `X`, `Y`, and `Z`. It requires `px >= 0`, `py >= 0`, `pz >= 0`, and `px + py + pz <= 1`.

`phasedamping` requires `gamma` in `[0,1]`. Its TensorCircuit-compatible parameterization damps X/Y by `sqrt(1-gamma)`; it is not the earlier proposed `(1-lambda)` dephasing parameterization.

`amplitudedamping` is TensorCircuit's generalized amplitude-damping channel with `gamma` and `p` both in `[0,1]`. The usual ground-state amplitude damping is the special case `p=1`.

The independent dense reference uses the following Kraus operators, which also freeze the meaning of the TensorCircuit-compatible parameters. `depolarizing` uses `sqrt(1-px-py-pz) I`, `sqrt(px) X`, `sqrt(py) Y`, and `sqrt(pz) Z`. `phasedamping` uses

```text
K0 = [[1, 0], [0, sqrt(1-gamma)]]
K1 = [[0, 0], [0, sqrt(gamma)]]
```

`amplitudedamping` uses

```text
K0 = sqrt(p)   [[1, 0], [0, sqrt(1-gamma)]]
K1 = sqrt(p)   [[0, sqrt(gamma)], [0, 0]]
K2 = sqrt(1-p) [[sqrt(1-gamma), 0], [0, 1]]
K3 = sqrt(1-p) [[0, 0], [sqrt(gamma), 0]]
```

All three channels are trace-preserving and therefore preserve the Heisenberg identity. `depolarizing` and `phasedamping` never create a new Pauli word. `amplitudedamping` has one possible local split from Z to I/Z. Native kernels apply these rules directly and must agree with both an equivalent explicit static PTM and an independent dense Kraus reference.

## 5. Forward representation

Static custom PTMs continue to compile finite nonzero entries into immutable local transitions at tape construction. They support only one or two distinct in-range wires, never materialize a global PTM, aggregate equal global words deterministically, and apply `max_weight` only after aggregation.

Built-in channel operations compile their channel kind, wire, and static values or runtime scalar-slot references. The forward kernel resolves each runtime scalar once per operation, validates the complete channel domain, and then applies the direct local rule while the GIL is released.

The depolarizing and phase-damping kernels scale coefficients in place and do not allocate transition lists or an aggregation map. The amplitude-damping kernel scales I/X/Y lanes and aggregates the Z-to-I/Z split directly. Specialized kernels must produce the same canonical words and coefficients as the corresponding explicit static PTM.

## 6. Native reverse and frozen-support semantics

Channel parameters use the same executed-trace reverse contract as existing rotation parameters. A local contribution enters reverse only when its forward multiplier and input coefficient produce a nonzero contribution and its aggregated output survives `max_weight` projection. Exact-zero contributions, outputs deleted by exact cancellation, and projected words are not reintroduced solely to compute a derivative.

Consequently, exact support-change points are intentionally not dense or fixed-basis derivatives. This includes a depolarizing Pauli eigenvalue that is exactly zero, `sqrt(1-gamma)=0` at `gamma=1`, an amplitude-damping split coefficient that is exactly zero, and exact cross-term cancellation. Those zero branches do not contribute a derivative. Retained branches still contribute their ordinary analytic local derivative.

No special gradient rejection is applied at `gamma=1`. Forward execution accepts the closed physical range, and reverse execution simply does not evaluate the singular derivative of a zero `sqrt(1-gamma)` edge because that edge is absent from the executed trace. Any other retained amplitude-damping branch contributes its finite local derivative.

At support-stable interior points, analytic channel gradients must match central finite differences of `PropagationEngine.expectation()` under the same `max_weight`. Boundary and exact-cancellation tests assert the frozen-support result directly instead of using finite differences as the oracle.

Static PTMs retain their current reverse transpose action on retained edges. They transport cotangents needed by surrounding scalar parameters but never accumulate a PTM-entry gradient.

Checkpoint/replay stores or reconstructs the same scalar values used by the forward operation order. Shared advanced slots and high-level occurrence slots are accumulated in deterministic gate/term order. No channel parameter introduces a Python callback or per-term FFI crossing.

## 7. JAX custom VJP

The current `jax.pure_callback` bridge keeps one flat native scalar input. The high-level custom-VJP primitive has two logical JAX inputs, `rotation_values` and `channel_values`, concatenates them in JAX as `concat(rotation_values, channel_values)`, and sends that one fixed-shape array through one callback. The callback returns the scalar value and one native gradient array, which the high-level bridge splits at the static `angle_count` boundary.

The custom VJP backward returns `cotangent * angle_gradient` and `cotangent * channel_gradient` for the two logical inputs. If one upstream JAX value is used by multiple rotations or channels, or simultaneously by a rotation and a channel, the ordinary VJPs of the stack/concatenate operations sum all occurrence contributions back to that value. Rust does not need an upstream sharing graph.

The intended high-level use is:

```python
def objective(theta):
    circuit = PropagationCircuit(nqubits)
    circuit.ry(0, theta=theta)
    circuit.phasedamping(0, gamma=jax.nn.sigmoid(theta))
    return circuit.expectation_jax(observable)

value, total_gradient = jax.jit(jax.value_and_grad(objective))(theta)
```

`total_gradient` includes both the rotation path and the channel path through the shared `theta`.

Required JAX behavior is scalar `jit` and `grad`, every supported channel family, mixed rotations/channels/static PTMs, shared upstream values, support-stable interior finite differences, contract-defined endpoint behavior, and repeated concurrent calls. General `vmap` over independent circuit instances is not required without a separate callback batching design.

Static `PropagationCircuit.ptm()` matrices must be concrete finite real `float64` values before tracing. A JAX-produced PTM and a gradient with respect to PTM entries are explicitly unsupported in this phase.

JAX dtype behavior follows the existing bridge: native computation remains `float64`, while the callback wrapper may cast inputs and outputs to the existing float32 callback dtype when JAX x64 is disabled.

## 8. Validation and failure behavior

- Static custom PTMs retain their existing one-/two-wire, shape, real-`float64`, finite-value, and copy/ownership validation.
- `depolarizing` rejects non-finite or negative `px`, `py`, or `pz`, and rejects `px + py + pz > 1`.
- `phasedamping` and `amplitudedamping` reject non-finite `gamma` outside `[0,1]`; `amplitudedamping` also rejects non-finite `p` outside `[0,1]`.
- Static channel values fail when the operation is constructed. Runtime advanced slots fail on execution after the complete channel tuple has been resolved.
- Concrete high-level terminals reject stored JAX values and direct the user to `expectation_jax()`.
- `PropagationEngine.expectation` and `value_and_grad` retain the exact-Hermitian observable requirement and real scalar output contract.
- Static non-trace-preserving PTMs retain the existing unnormalized linear expectation; no implicit quotient or physical-channel claim is added.
- `SPPSEngine` and `SPPSCircuit.from_qir()` reject built-in channels at construction.
- All scalar-buffer, output, checkpoint, transition, and major-workspace arithmetic uses checked arithmetic. `max_bytes` remains a best-effort major-allocation guard rather than an exact RSS promise.

Wrong Python object kinds raise `TypeError`; invalid scalar values, channel domains, wires, QIR records, shapes, and incompatible arguments raise `ValueError`; checked dimension arithmetic maps to `OverflowError`; enforced allocation limits map to `MemoryError`. Tests assert both exception class and a stable message fragment.

## 9. FFI and ownership

The public Python layer owns logical operations, friendly scalar validation, the small high-level `channel_parameter_layout`, and the no-copy split of the native gradient at `angle_count`. The native execution call carries one contiguous scalar-parameter array. Rust performs complete runtime-domain validation, channel application, aggregation where needed, projection, reverse, and output preparation with the GIL released.

Native gradient read-back remains one flat NumPy array for scalar slots and one flat two-dimensional array for batch scalar gradients. No nested channel records, PTM transitions, or Python lists cross the native read-back ABI.

Plans and tapes are immutable after construction and safe for concurrent execution. Runtime scalar buffers, aggregation state, checkpoints, replay state, and reverse scratch are call-local. Native handles do not retain caller-owned runtime arrays after a call.

The implementation ownership is fixed as follows:

| Responsibility | Authoritative owner |
| --- | --- |
| Advanced construction and advanced scalar-gradient results | `python/tencirpauli/propagation.py` |
| Logical `PropagationCircuit`, high-level result metadata, angle/channel occurrence layouts, and QIR | `python/tencirpauli/propagation_circuit.py` |
| TensorCircuit circuit extraction only when separately supported | `python/tencirpauli/integrations/tensorcircuit.py` |
| JAX stacking and custom VJP | `python/tencirpauli/jax_support.py` |
| Typed channel operations, validation, kernels, aggregation, checkpoint/replay, and reverse | `crates/tencir-pauli-core/src/gate.rs` and `crates/tencir-pauli-core/src/propagation.rs` |
| Flat native construction/execution/read-back and GIL release | `crates/tencirpauli-native/src/propagation.rs` |
| Private ABI typing | `python/tencirpauli/_native.pyi` |

Before adding a new primitive, the implementation must reuse the existing tape sharing, slot coverage/validation, aggregation, projection, checkpoint/replay, batch scheduling, NumPy read-back, and JAX callback paths where their contracts already apply. Rotation `ParameterRef::Static {cos, sin}` remains unchanged because it owns cached trigonometric values; channel scalars use the minimal native static-value-or-slot representation needed by their direct kernels rather than forcing raw channel values into the rotation-specific representation. No Python reference execution path, second channel representation, nested read-back ABI, or compatibility fallback is retained after the native path exists.

## 10. Tests and independent references

The phase requires independent dense one-qubit Kraus references constructed directly from the TensorCircuit-compatible channel definitions. The references must not call the native channel kernels or native PTM builder. Equivalent explicit static PTMs provide a second orientation and wire-order check but do not replace the Kraus reference.

Required correctness coverage includes:

- `depolarizing` for asymmetric `px`, `py`, and `pz`, including single-axis Pauli noise and valid probability boundaries;
- `phasedamping` and generalized `amplitudedamping`, including `p=0`, `p=1`, and mixed `p`;
- built-in channel equality with explicit static PTMs and dense Kraus expectations;
- support-stable scalar channel gradients against central finite differences;
- frozen-support behavior at exact-zero eigenvalues, `gamma=1`, zero split coefficients, exact cancellation, and `max_weight` deletion;
- mixed rotations, all supported channels, and static one-/two-qubit PTMs;
- advanced shared slots, high-level angle order, high-level channel order, exact `channel_parameter_layout` assertions, and read-only contiguous empty/non-empty result arrays;
- product computational states and Bloch states supported by the existing engine;
- batch values and scalar gradients, empty batches, checkpoint intervals, and deterministic repeated/concurrent calls;
- `PropagationCircuit.to_qir()` and `from_qir()` round trips plus accepted TensorCircuit channel QIR records;
- JAX `jit`/`grad` for every supported channel and for one upstream scalar shared between rotation and channel occurrences; and
- rejection of unsupported SPPS channels, unsupported channel names, invalid physical domains, malformed QIR, complex/static-invalid PTMs, and JAX-valued PTM matrices.

Existing static one- and two-qubit PTM forward and transpose-reverse regressions remain required. No test or public result may imply a gradient with respect to static PTM entries.

Default float64 value/PTM/Kraus comparisons use `atol=1e-11` and `rtol=1e-10`; analytic-gradient comparisons use `atol=1e-9` and `rtol=1e-8`; central finite differences use `atol=1e-6` and `rtol=1e-5`. JAX-x64 follows the same analytic tolerances. JAX float32 callback tests may use `atol=1e-5` and `rtol=1e-5`. A stress case may use a wider local tolerance only with a documented numerical reason beside that test; no global relaxation is permitted.

## 11. Performance and acceptance gates

Benchmarks are release-mode and include complete Python input handling, native handle construction, callback boundaries, synchronization, value/gradient materialization, and steady execution. At minimum record setup, first call, steady value, steady value-plus-gradient, scalar-buffer bytes, term counts, peak estimate, numerical error, and thread settings.

Required workloads are asymmetric one-qubit depolarizing, phase damping, generalized amplitude damping, equivalent static-PTM controls, a 12-qubit mixed rotation/channel/static-PTM tape, batch observables, and a JAX noise-aware VQE-style scalar objective. Specialized channel kernels are compared with equivalent static PTMs under identical semantics.

The implementation must profile before optimizing. A specialized kernel is retained only when its release-mode end-to-end contribution is material on a representative workload or when it removes a measured allocation/branching class. Performance records remain local and informational under the repository policy; they are not CI wall-time gates.

Acceptance requires the full Rust/Python quality gate, dense Kraus agreement, support-stable finite-difference agreement, explicit frozen-support boundary tests, TensorCircuit-compatible QIR evidence, JAX `jit/grad`, deterministic concurrency, and no regression in existing rotation, static PTM, propagation batch, or SPPS rejection tests.

Required final validation is:

```bash
python scripts/check.py --fix --benchmark smoke
python benchmarks/run.py record
```

The first command must complete the repository's formatting, Rust checks, Python lint/type checks, release native build, tests, examples, and benchmark-smoke workflow in the project environment. The second command records the required release-mode Phase 12 workloads on one identified machine; the result remains untracked and informational under repository policy.

## 12. Explicit non-goals and follow-up

The following are not implemented or implied by Phase 12:

- runtime or parameterized PTM values;
- gradients or JAX VJPs with respect to custom PTM entries;
- channels beyond `depolarizing`, `phasedamping`, and `amplitudedamping`;
- arbitrary Kraus operator APIs or automatic CPTP/CP certification;
- automatic execution of TensorCircuit `channel_f` callables or arbitrary channel-object conversion;
- imaginary-time propagation, normalized numerator/denominator terminals, or Schrödinger Pauli-state evolution;
- PTM/noise support in SPPS or stochastic path sampling;
- complex Hermitian-basis PTMs; and
- general JAX callback batching or `vmap` over independent circuit instances.

A later phase may add parameterized PTM entry VJPs only after a concrete channel-learning or system-identification workload justifies the additional runtime layout, gradient output, zero-entry semantics, JAX input, memory accounting, and public result complexity.
