# Why I Developed TenCirPauli

*A technical note from the author of [TensorCircuit-NG](https://github.com/tensorcircuit/tensorcircuit-ng).*

[TensorCircuit-NG](https://github.com/tensorcircuit/tensorcircuit-ng) is already a very capable framework. I built it to make quantum-circuit programming expressive, differentiable, and practical across several numerical backends. It gives users a clean circuit frontend, a flexible backend abstraction, automatic differentiation, JIT compilation, and access to the tensor-network and accelerator ecosystems. For regular tensor programs, these choices work extremely well. They let a researcher describe a circuit at a high level and still obtain compiled numerical execution underneath.

Many of the workflows I care about fit this model beautifully. State preparation, parameterized gates, expectation values, dense tensor contractions, and repeated optimization steps all benefit from JAX's transformation system. Once a program has been traced and compiled, its steady-state execution can be remarkably fast. TensorCircuit-NG has become strong precisely because it takes this model seriously instead of hiding the backend behind a collection of unrelated special cases.

[TenCirPauli](https://github.com/tensorcircuit/TenCirPauli) grew out of the next question: what happens when a quantum workflow contains a substantial amount of computation that does not look like a regular tensor program? The answer exposed a useful boundary in TensorCircuit-NG. The framework is very good at the numerical work for which JAX was chosen. Some of the surrounding operator work has a different computational character, and that character deserves a different runtime.

## Framework selection creates a performance profile

TensorCircuit-NG's high-performance path is closely connected to JAX. This brings major advantages and makes JAX's preferred workload shape visible in TensorCircuit workflows. Every runtime has such a shape, and performance depends on how closely a problem matches it.

JAX is strongest when a computation can be expressed as a regular array program. Dense linear algebra, large matrix and vector operations, batching, automatic differentiation, and accelerator execution are all natural targets. XLA can trace the program, optimize it, fuse operations, and produce a fast executable. For a dense matrix multiplication or another BLAS/LAPACK-class kernel, moving the operation into Rust usually changes very little. Rust will call the same optimized numerical libraries, while JAX may have additional opportunities for fusion, compilation, or execution on a GPU or TPU.

This gives the first design principle for TenCirPauli: dense numerical work should stay close to the TensorCircuit-NG backend, while irregular surrounding work can use a native systems language.

## Where the JAX model becomes expensive

The first difficulty is dynamic shape. A Pauli propagation recurrence can create new words, merge equal words, cancel terms, and move contributions between weight sectors. Symmetry analysis, mappings, and grouping can likewise produce variable numbers of basis states, terms, transitions, groups, or constraints.

JAX can express these algorithms, yet the natural representation often conflicts with its compilation model. A dynamic algorithm consequently needs padding, masking, sorting, bounded buffers, static limits, or custom control-flow encoding. These techniques can be valuable for a carefully designed workload, while adding representation work and potentially changing the algorithm's semantics.

The second difficulty is compilation latency. JIT compilation can produce an extremely fast steady-state kernel, while the first call carries tracing, lowering, optimization, and executable creation. This is a good trade for long-running workloads with many repeated calls; it is less attractive for short jobs or frequently changing structures.

JAX can avoid this cost very effectively when the workflow has a stable shape. Hamiltonian coefficients, circuit angles, and other numerical inputs can live inside one parameterized compiled function, allowing broad scans to reuse the same executable. TensorCircuit-NG handles this style of work extremely well. The compilation boundary becomes important when term counts, Pauli supports, circuit control flow, sector dimensions, grouping results, or propagated-operator shapes change. Each new structure may require a new trace or a more elaborate static encoding, so separating setup, first execution, and steady execution helps identify where a native complementary path is useful.

The third difficulty is a workload dominated by bit strings and bit manipulation. Pauli words, occupation states, symmetry generators, basis indices, and transition keys can all be represented as packed integers. Their operations include XOR, parity, masks, shifts, comparisons, hash lookups, and small logical branches; the computation is controlled by the bits themselves more than by dense floating-point arithmetic.

JAX supports bitwise operations, loops, and integer arrays. A single XOR does not become faster simply because it is written in Rust. The opportunity appears in the complete workload: millions of small operations, changing output sizes, duplicate-key aggregation, branch-heavy recurrences, and data structures that do not map cleanly to a dense rectangular array. A Python fallback exposes interpreter overhead, while a tensor encoding can introduce padding, sorting, masking, recompilation, or extra preparation.

These cases reveal a specific type of gap. TensorCircuit-NG has a powerful numerical engine, while some operator workflows need a compact control-oriented engine for irregular discrete computation. That gap is where a Rust companion can create real value.

## Why Rust is a useful complement

Rust's scientific-computing ecosystem is smaller than Python's, and its linear-algebra stack is less mature. That is acceptable for this design because dense matrix multiplication, eigensolvers, tensor contractions, and accelerator kernels already have mature homes in the Python and JAX ecosystems. TenCirPauli gains little by rebuilding those foundations.

Rust has a different set of advantages. It compiles ordinary loops and branches to native code, gives direct control over memory and integer representations, handles dynamic collections without Python object overhead, and makes packed-data algorithms natural to express. Hash maps, bit masks, transition tables, and variable-length work queues fit this runtime well. The core can release the Python GIL while processing a complete batch, so the Python boundary stays outside the hot loop.

The complement can be summarized as follows:

| Workload characteristic | JAX / TensorCircuit-NG | Rust / TenCirPauli |
| --- | --- | --- |
| Dense matrix and tensor arithmetic | Strong fit | Usually delegates to established numerical libraries |
| Regular batching and static array programs | Strong fit | Capable, with less ecosystem leverage |
| Automatic differentiation through numerical code | Strong fit | Explicit local derivative rules where needed |
| Dynamic term counts and variable shapes | Requires encoding for compilation | Natural control flow and dynamic storage |
| Hash-based aggregation and graph-like analysis | Possible, often awkward inside traced arrays | Direct and compact |
| Packed integers, bit strings, parity, masks | Expressible, with limited advantage from tensorization | Natural inner-loop representation |
| Long-running accelerator workloads | Strong fit | Usually the wrong execution target |
| Short or frequently changing workloads | Compilation cost can dominate | Native setup can remain small and predictable |

This division of labor is the starting point for TenCirPauli. It also explains why the project is designed as a companion to TensorCircuit-NG. The two runtimes can cooperate on one scientific workflow, with each one handling the part of the program that matches its performance profile.

## What this implies for Pauli computation

Pauli algebra is a particularly clear example of the boundary. A Hamiltonian may be written as

```text
H = c₀ P₀ + c₁ P₁ + ··· + cₘ Pₘ,
```

and the eventual numerical task may be an expectation value or a gradient. Between those points, the program may combine equal words, track phases, compute commutation relations, construct measurement groups, discover symmetries, restrict a physical sector, map fermions, and propagate observables through circuits.

These operations are symbolic and discrete, dominated by compact keys, parity, support inspection, branching, aggregation, and variable-size results. The final state-vector or tensor-network calculation may still be dense and backend-friendly. Pauli-heavy workflows therefore benefit from native preparation and transformation followed by a regular TensorCircuit-NG or JAX numerical plan when appropriate.

That makes Pauli computation a natural Rust target. Packed X/Z data, phase bookkeeping, parity checks, aggregation, and variable-size collections all fit native integer loops and compact data structures. From this workload, the surrounding capabilities follow naturally: the same native machinery can support Pauli algebra, measurement grouping, symmetry analysis, compilation, and observable propagation.

## The functionality follows from the runtime choice

The design of TenCirPauli's public API follows directly from this framework analysis. The matrix is one useful destination for a small system. Larger workflows need algebraic transformations, measurement plans, reduced bases, matrix-free actions, observable trajectories, or differentiable circuit objectives. The architecture preserves the operator across those stages.

The resulting workflow can be viewed as a sequence of transformations:

```text
structured model → canonical operator → algebra and analysis → reduction or mapping → execution plan → measurement or gradient
```

Each arrow has a different computational profile: symbolic expansion, canonicalization, bit-packed keys, graph analysis, and integer indexing at the front; dense state tensors, backend autodiff, sparse linear algebra, or native CPU kernels at the end. The representation must cross this boundary without losing its meaning.

### From physical models to canonical operators

Pauli operators are the central representation for qubit Hamiltonians, observables, and propagation. TenCirPauli extends the same native approach to fermionic, bosonic, qudit-Weyl, Majorana, and hybrid operators. Jordan–Wigner, parity, and Bravyi–Kitaev mappings transform structured terms into Pauli form, while optional chemistry adapters bring molecular Hamiltonians into the same pipeline. Canonical aggregation, explicit phase handling, deterministic ordering, and compact keys give every later capability one stable operator.

### From operators to measurements and structural information

Measurement planning is an analysis stage. A qubit-wise commuting grouping uses Pauli supports and local bases, so its useful result contains more than a partition of term indices. It also describes the basis rotations and the reconstruction of Pauli eigenvalues from rotated computational-basis samples. The grouping result can therefore travel from symbolic preprocessing to experimental post-processing without a second interpretation layer.

Symmetry analysis follows the same pattern. Z₂ generators, tapering sectors, fixed-particle-number spaces, and general additive-charge restrictions use bit operations, constraint solving, basis indexing, and leakage checks. They can reduce the computational object before a circuit, sparse matrix, or matrix-free plan is executed. For a 60-qubit, two-particle system, the relevant sector contains `choose(60, 2) = 1,770` basis states, compared with `2**60` states in the full computational space. This is a structural reduction that comes before numerical optimization.

### From one operator to several execution targets

Compilation chooses the endpoint that matches the next consumer: dense, COO, CSR, SciPy linear operator, Rust-native matrix-vector product, or a pure-array TensorCircuit-NG/JAX backend plan. For large systems the useful result may be a reusable MVP plan that never materializes the full matrix; for a restricted sector it may be a compact transition plan over the physical basis. Rust prepares the fixed structure before backend tracing, and the public API keeps the target choice explicit.

### From circuits to observable execution and gradients

The same boundary applies when the operator meets a circuit. A fixed-particle-number circuit can execute directly in a restricted basis. A Pauli observable can propagate backwards through a circuit using dynamic native storage, with terms expanded, merged, cancelled, and projected by weight. A stochastic Pauli-path estimator can use the same operator semantics with an explicit sampling contract. Native value-and-gradient paths use local derivative and vector-Jacobian rules for supported gates, while TensorCircuit-NG and JAX remain available when the objective belongs inside a backend-traced tensor program.

The user-facing interface remains entirely in Python. Users construct operators, pass them to grouping or compilation APIs, connect them to TensorCircuit-NG circuits and backends, and receive ordinary Python objects, NumPy arrays, or backend tensors. They do not need to write Rust, manage FFI handles, or choose native data layouts. The workflow should feel as fluent as TensorCircuit-NG itself; Rust stays behind the boundary and handles the work that scales with terms, gates, groups, transitions, or basis states through coarse-grained native calls.

## What the Early Benchmarks Show

The first benchmark suite provides evidence for this division of labor. The headline result comes from a representative 60-qubit, two-particle U(1) VQE: TenCirPauli's first compiled value-and-gradient call was about 688× faster than the corresponding TensorCircuit/JAX call, and the steady native path was about 2.6× faster. In a matched stochastic Pauli-path value-and-gradient workload, the native path was about 6.5–7.6× faster at 12 qubits and about 12.4× faster at 16 qubits in the recorded steady-state comparisons.

These results show where the architecture creates leverage. The first-call improvement reflects the cost of tracing and compilation, while the steady-state improvement reflects the benefit of native handling for irregular Pauli structure. A JAX implementation can still be the best choice after a long compilation has been amortized, and a Rust implementation becomes attractive when the workload changes shape frequently, contains heavy bit manipulation, or spends most of its time in symbolic preparation. The measurements separate native setup, first execution, steady execution, memory, and numerical agreement so that these cases remain visible.

Later technical posts will open up these comparisons and individual workloads in more detail. Here the benchmarks support one framework-level conclusion: runtime choice should follow the shape of the computation.

## The larger lesson

TensorCircuit-NG solved an important problem by making tensor-based circuit computation expressive, differentiable, and backend-aware. Its JAX-centered design is a major part of that success. The same design makes dynamic, branch-heavy, shape-changing, and bit-string-heavy tasks stand out as a separate class of workload.

TenCirPauli extends TensorCircuit-NG around that boundary. It gives irregular Pauli workloads a native path while preserving TensorCircuit-NG's strengths for dense numerical computation, parameterized execution, automatic differentiation, and accelerated backends. The result is a complementary architecture: JAX handles regular numerical programs, Rust handles irregular Pauli structure, and Python connects the two into one scientific workflow.

That is why I developed TenCirPauli. The project began with a framework-selection question, and the Pauli algebra, measurement planning, symmetry tools, restricted sectors, propagation engines, and backend plans followed from the answer.
