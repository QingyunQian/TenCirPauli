---
hide:
  - navigation
  - toc
---

<div class="tp-hero">
  <div class="tp-hero-copy">
    <p class="tp-kicker">RUST-NATIVE · PYTHON-FIRST · TENSORCIRCUIT</p>
    <h1>Pauli structure,<br><em>made practical.</em></h1>
    <p class="tp-hero-lede">TenCirPauli turns structured quantum objects into practical computation and evidence: Pauli algebra, Hamiltonians, symmetry and code analysis, observable execution, and classical-shadow estimation for the TensorCircuit ecosystem.</p>
    <div class="tp-actions">
      <a class="tp-button tp-button-primary" href="quickstart/">Start with Python</a>
      <a class="tp-button tp-button-secondary" href="api/">Browse the API</a>
    </div>
  </div>
  <div class="tp-hero-mark" aria-label="TenCirPauli logo">
    <img src="assets/tencirpauli-logo.png" alt="" width="220" height="220">
    <span>structured<br>quantum work</span>
  </div>
</div>

<div class="tp-note"><strong>One package, one workflow.</strong> Describe the structure, organize it for the task, compute in the right runtime, then inspect the result as a value, estimate, or diagnosis.</div>

## The library in four verbs

<div class="tp-flow" markdown>
  <div class="tp-flow-item"><span>01</span><strong>Describe</strong><small>Operators, circuits, Hamiltonians, and code spaces.</small></div>
  <div class="tp-flow-arrow">→</div>
  <div class="tp-flow-item"><span>02</span><strong>Organize</strong><small>Canonicalize, map, group, reduce, and choose a plan.</small></div>
  <div class="tp-flow-arrow">→</div>
  <div class="tp-flow-item"><span>03</span><strong>Compute</strong><small>Build, apply, propagate, differentiate, or sample.</small></div>
  <div class="tp-flow-arrow">→</div>
  <div class="tp-flow-item"><span>04</span><strong>Inspect</strong><small>Read out values, estimates, syndromes, and logical errors.</small></div>
</div>

## What it covers

<div class="tp-card-grid" markdown>
  <div class="tp-card">
    <p class="tp-card-index">01 / DESCRIBE</p>
    <h3>Build quantum structure</h3>
    <p>Pauli and structured operators, Hamiltonians, circuits, stabilizer codes, and measurement states.</p>
  </div>
  <div class="tp-card">
    <p class="tp-card-index">02 / ORGANIZE</p>
    <h3>Exploit the structure</h3>
    <p>Canonicalize, map, group, taper, restrict to sectors, and prepare execution or measurement plans.</p>
  </div>
  <div class="tp-card">
    <p class="tp-card-index">03 / COMPUTE</p>
    <h3>Run the workload</h3>
    <p>Use dense, sparse, matrix-free, native, TensorCircuit, or JAX execution with values and gradients.</p>
  </div>
  <div class="tp-card">
    <p class="tp-card-index">04 / INSPECT</p>
    <h3>Turn computation into evidence</h3>
    <p>Estimate observables with classical shadows, reconstruct grouped measurements, and diagnose syndromes and logical errors.</p>
  </div>
</div>

## A first result

```python
import tencirpauli as tcp

hamiltonian = tcp.PauliOperator.from_terms(
    2, [("XX", 0.5), ("ZI", -1.25j)]
)

matrix = hamiltonian.dense()
print(matrix.shape)
# (4, 4)
```

The public surface stays in Python; the computationally dense work is handled behind one coarse native boundary. Read the [concepts](concepts.md) page for the mental model, or go straight to the [quickstart](quickstart.md) when you already know what you need.

<div class="tp-footer-links">
  <a href="concepts/">How it fits together</a>
  <a href="quickstart/">Run a small example</a>
  <a href="api/">Find a public symbol</a>
  <a href="https://github.com/tensorcircuit/TenCirPauli">Read the source</a>
</div>
