# Stabilizer、QLDPC 与容错电路分析能力调研

状态：探索性技术调研，2026-08-21。本文整理适合 TenCirPauli 现有 packed-Pauli、binary-symplectic、Clifford 和 coarse-FFI 技术栈的量子纠错能力、数学边界、输入输出合同、工程难度与风险；它不是已批准的实现规格、版本承诺或路线图。

## 1. 结论摘要

Qubit stabilizer code 是主流量子纠错和容错计算的核心代数抽象，但不是所有量子纠错码的统一抽象。它覆盖常见 CSS、surface/toric/color、绝大多数 qubit QLDPC、concatenated stabilizer codes，并可通过独立的 subsystem/gauge 抽象扩展到 Bacon–Shor 和 gauge codes；它不直接覆盖 non-additive codes、一般 approximate/operator-algebra QEC、针对非-Pauli 噪声设计的特殊代码、bosonic cat/binomial/GKP codes 或一般 qudit stabilizer codes。

对 TenCirPauli 而言，stabilizer-code 分析与当前设计原则高度一致：主要计算是 packed GF(2) elimination、symplectic inner product、Pauli conjugation、XOR、popcount、bounded enumeration 和 deterministic hash aggregation，不要求浮点线性代数库，也不要求通用符号计算引擎。现有 `PauliWord`、Pauli commutation、Z2 nullspace、Clifford operations、`GateTape`、`StabilizerState`、counter-based random streams 和 native-handle/flat-array ABI 可以直接复用，但 code space、logical quotient、detector timeline 和 subsystem gauge 语义仍需要新的明确 owner。

“输出 physical logical X/Z”在纯代数意义上是容易的：从 stabilizer check matrix 的 symplectic centralizer 中取商空间基，再做 symplectic pairing，即可在多项式时间内得到任意一组 logical `X_i/Z_i`。困难的是输出低权、局域、几何自然、硬件友好或最优的代表；logical representative 可以乘任意 stabilizer 而保持同一逻辑作用，选择最低权代表等价于一个 decoding/distance 类优化问题，对一般代码没有廉价通用算法。这是许多 QLDPC 工作给出 parity checks、参数和构造，却不在正文列出一整套 physical logical strings 的主要原因之一。

## 2. 调研边界与工程约束

本文只考虑以下计算风格：

- Qubit Pauli group 的 binary symplectic representation；第一版概念不自动推广到 prime/composite qudit Weyl groups。
- Rust core 持有 scalable state，Python 只做友好输入、控制元数据和显式结果物化。
- 批量 syndrome、classification、fault propagation 和搜索通过 coarse native calls 完成，不逐 error、generator、gate 或 fault 跨 PyO3。
- 公开结果 deterministic；logical basis 虽然数学上不唯一，但给定输入顺序和 canonicalization policy 后输出必须稳定。
- Distance、decoder 和 fault-combination 问题必须明确 exact、bounded、lower bound、upper bound 和 witness 语义，不能把未完成搜索报告为精确答案。
- 不引入 BLAS/LAPACK、dense eigensolver、通用 CAS、SAT/ILP solver 或完整图优化框架作为默认依赖。
- 不以复刻通用 stabilizer simulator、MWPM decoder 或外部工具的完整功能面为目标；优先补充 code algebra、logical classification、fault witness 和 TensorCircuit-facing integration。

## 3. Stabilizer code 的数学对象

对 `n` 个 qubits，忽略全局 phase 后，一个 Pauli 可写成二进制 symplectic vector

```text
p = (x | z) in GF(2)^(2n).
```

两个 Pauli 的对易关系由

```text
<p, q>_s = x_p · z_q + z_p · x_q mod 2
```

决定。一个 stabilizer code 由 Abelian Pauli subgroup

```text
S = <S_1, ..., S_r>,   -I not in S
```

定义，其 code space 是所有 `S_i` 的共同 `+1` eigenspace。若 independent generator rank 为 `r`，则 code space dimension 为 `2**k`，其中

```text
k = n - r.
```

当前项目的 `StabilizerState` 对应 `r=n, k=0` 的特殊情形；一般 `StabilizerCode` 必须允许 `r<n`，因此表示整个 logical code space，而不是一条确定的 statevector 或唯一 stabilizer state。

Stabilizer 的 Pauli normalizer/centralizer 为

```text
N(S) = {P : [P, S_i] = 0 for every i}.
```

Logical Pauli group 是 quotient

```text
N(S) / S,
```

其 binary dimension 为 `2k`。选择 `k` 对 symplectic canonical generators 即得到 physical representatives of logical `X_i` and `Z_i`。

## 4. Pauli error 的分类

给定 Pauli error `E`，最直接的 syndrome 是它与 stabilizer generators 的 commutation vector：

```text
s_i(E) = <E, S_i>_s.
```

需要区分三类零或非零 syndrome 情形：

| 条件 | Syndrome | 对 code space 的作用 | 解释 |
| --- | --- | --- | --- |
| `E not in N(S)` | 非零 | 把 code space 映到其他 syndrome sector | syndrome-visible error |
| `E in S` | 零 | 在 code space 上等价于恒等，忽略全局 phase | degenerate/harmless stabilizer action |
| `E in N(S) \ S` | 零 | 保持 code space，但执行非平凡 logical Pauli | harmful undetectable logical error |

因此“syndrome 没有响”不等于“发生了 logical error”：error 也可能只是 stabilizer，物理态在 code space 内没有变化。反过来，非零 syndrome 只说明 error 被发现，不保证 decoder 能唯一判断它是哪一个 physical error。

### 4.1 三比特 repetition code 例子

取

```text
S = <ZZI, IZZ>,
C = span{|000>, |111>}.
```

它编码一个 logical qubit。一个稳定的 logical representative 选择可以是 `X_L=XXX`、`Z_L=ZII`。

| Error | Syndrome against `(ZZI, IZZ)` | Classification |
| --- | --- | --- |
| `XII` | `10` | syndrome-visible single bit flip |
| `IXI` | `11` | syndrome-visible single bit flip |
| `IIX` | `01` | syndrome-visible single bit flip |
| `ZZI` | `00` | stabilizer，logical identity |
| `XXX` | `00` | logical `X` |
| `ZII` | `00` | logical `Z` |

`XII` 与 `IXX` 具有相同 syndrome，因为它们的差为 `(XII)^dagger(IXX)=XXX`，即 logical `X`。若 decoder 对该 syndrome 选择 `XII` 作为 correction，而实际 error 是 `IXX`，correction 后会残留 logical `XXX`。这说明 correctability 是 error set 与 decoder/correction policy 的性质，不应被压缩成脱离上下文的单 error `is_correctable()` 布尔值。

### 4.2 Error set 的纠正条件

对一组 candidate errors `{E_a}`，stabilizer 版本的 Knill–Laflamme 条件可表述为：对任意 `a,b`，`E_a^dagger E_b` 要么属于 `S`，表示两种 errors 在 code space 上作用相同并允许 degeneracy；要么不属于 `N(S)`，表示它们具有不同 syndrome 并可被区分。若 `E_a^dagger E_b` 属于 `N(S) \ S`，两种 errors syndrome 相同但相差 logical operator，不能被同一个 recovery 同时纠正。

## 5. 检测 distance、纠正 distance 与 erasure distance

Stabilizer code distance 定义为最轻的非平凡 logical Pauli：

```text
d = min weight(P),  P in N(S) \ S.
```

由此得到标准的 worst-case 保证：

```text
detect arbitrary Pauli errors of weight <= d - 1,
correct arbitrary Pauli errors of weight <= floor((d - 1) / 2),
correct erasures on at most d - 1 known locations.
```

第一条中的“detect”采用 QEC 的无害等价语义：weight `<d` 的 Pauli 不可能是 harmful undetectable logical；它要么产生非零 syndrome，要么属于 stabilizer 并在 code space 上无害。若把 detect 狭义理解成“syndrome bit 必须非零”，低权 stabilizer 是例外，但它不构成纠错失败。

纠正半径来自 pairwise ambiguity。若 `weight(E_a), weight(E_b) <= t`，则

```text
weight(E_a^dagger E_b) <= 2t.
```

当 `2t < d` 时，该差不可能是 nontrivial logical，所以任意两种 weight-`t` errors 要么 syndrome 可区分，要么只差 stabilizer。因此 guaranteed correction radius 为

```text
t = floor((d - 1) / 2).
```

这些是 worst-case guarantees，不是某个具体 decoder 和 noise model 的完整能力。Degenerate codes 可能纠正某些超过 `t` 的 errors；biased-noise decoder 也可能对特定高权 patterns 成功。相反，具体 syndrome-extraction circuit 可能因为 hook errors、measurement faults 或时间方向上的 detector ambiguity，使 circuit distance 低于理想 code distance。

对 CSS code 常分别定义 `d_X` 和 `d_Z`。它们分别控制 X-type 与 Z-type harmful undetectable logical 的最小权，并给出相应的 asymmetric detection/correction guarantees。报告一个总 distance 时通常取 `min(d_X,d_Z)`，但 biased noise workflow 应保留两者。

## 6. Physical logical X/Z 是否容易输入和输出

### 6.1 任意 logical basis：代数上容易

设 stabilizer check matrix 为 `H_S=[X|Z]`。Normalizer 是 symplectic constraint 的 nullspace：

```text
N(S) = kernel(H_S Lambda),
Lambda = [[0, I], [I, 0]].
```

先计算该 kernel，再 quotient 掉 `rowspace(H_S)`，可得到 dimension `2k` 的 logical vector space。随后通过 symplectic Gram–Schmidt 或等价 elimination，把 basis 配成满足

```text
<X_i, Z_j>_s = delta_ij,
<X_i, X_j>_s = <Z_i, Z_j>_s = 0
```

的 logical pairs。这些操作是 polynomial-time packed GF(2) elimination，不需要数值线性代数库。

自然的输入形式包括 Pauli strings、code arrays、packed `(x_words,z_words)`，以及 CSS `H_X/H_Z` bit matrices。自然的输出形式是两个 flat packed arrays：

```text
logical_x_words: shape (k, word_count)
logical_z_words: shape (k, word_count)
```

必要时再显式 materialize 为 `2k` 个 `PauliWord`。对大 QLDPC instance，flat packed output 比长度为 `n` 的 Python strings 更符合现有 ABI 原则。

### 6.2 CSS 情形更直接

若 `H_X H_Z^T = 0`，则

```text
X logical space = kernel(H_Z) / rowspace(H_X),
Z logical space = kernel(H_X) / rowspace(H_Z).
```

计算任意 quotient bases 很直接；把 X/Z quotient bases 配成 canonical anticommuting pairs 仍需要一次 pairing reduction，但不是困难的组合优化。

### 6.3 为什么 logical representative 不唯一

若 `L` 是 logical representative，任意 `S in stabilizer group` 都给出同一个 logical action：

```text
L and L*S represent the same logical Pauli.
```

此外，`k` 个 logical qubits 的 basis 本身可以经过任意 logical Clifford/symplectic basis change。因而同一个 code 没有一套数学上唯一的 physical logical X/Z strings。软件可以给出 deterministic canonical basis，但“canonical”只表示算法和输入顺序固定，不表示几何上最自然或物理实现最优。

### 6.4 为什么低权或局域 logical representative 很难

从一个 arbitrary logical representative 出发，在其 stabilizer coset 中寻找最低权成员是

```text
minimize weight(L*S) over S in stabilizer group.
```

这与 decoding、coset-leader 和 code-distance 搜索属于同类组合优化。一般 stabilizer/QLDPC code 的 exact minimum-weight logical、exact distance 和通用 decoding 都可能指数增长；一般 quantum decoding 已知为 NP-hard。Polynomial-time elimination 只能保证给出一组正确 logical basis，不能保证低权、局域或硬件友好。

### 6.5 为什么 QLDPC 论文经常不直接列 physical logical operators

- QLDPC 通常描述一族 codes 或一个 construction，而不是一个小型固定 instance；physical logical strings 依赖具体 block length、lift、seed、boundary condition 和 qubit ordering。
- `k` 可能随 `n` 线性增长。完整列出 `2k` 个长度 `n` 的 strings 本身就是 `O(kn)` 输出，正文中没有可读性。
- 任意 elimination 输出往往是 dense、高权且破坏几何对称性的 basis，对理解代码并无帮助。
- 论文更关心 rate、distance bounds、check weight、decoder threshold 和 construction；decoder 通常只需要 check matrices 与 syndrome，不要求预先打印一套 logical basis。
- 真正物理上有意义的 logicals 常由 homology、边界、string/membrane operators 或 construction geometry 描述，而不是逐位 bitstring。
- 若研究 logical failure，软件可以在固定 instance 上按需生成 logical basis；若研究最小 logical，则需要单独的 distance/search computation，不能把普通 elimination 输出当成答案。

因此“论文没有列出 logical operators”通常不表示它们代数上无法计算，而是任意 basis 缺少规范性、输出过大，或者作者真正需要的是难得多的 low-weight/geometric representatives。

## 7. 功能范围与工程难度

以下分数按满分 100 估计，表示从当前 TenCirPauli 技术栈出发，把能力做到可公开发布所需的工程难度，包括 Rust core、PyO3、Python API、正确性测试、property/differential tests、benchmark、文档和资源合同。分数不是日程承诺，误差约为 `±5–10`；后续项目默认基础 `StabilizerCode` 已完成。

| Capability | 难度 | 主要计算 | 主要风险与范围 |
| --- | ---: | --- | --- |
| Generator validation、canonicalization、rank、`[[n,k]]` | 25 | Packed GF(2) row reduction、phase checks | Deterministic basis、dependent generators、`-I` rejection |
| Batch syndrome | 15 | Symplectic matrix-vector/matrix-matrix parity | 输入布局、wide words、批量内存；算法风险低 |
| Canonical logical X/Z pairs | 38 | Centralizer nullspace、quotient、symplectic pairing | 数学不唯一、输出 convention 和 pairing determinism |
| Error classification 与 stabilizer coset | 35 | Syndrome、rowspace membership、logical coordinates | Pure-error representative 与 decoder-independent semantics |
| CSS `H_X/H_Z` facade | 22 | Binary rank/nullspace、orthogonality | X/Z 命名、transpose/order conventions、dependent checks |
| 小系统 exhaustive distance | 35 | Weight-ordered Pauli enumeration | Complete/bounded 语义、identity/stabilizer filtering |
| CSS 专用 bounded distance | 55 | X/Z 分离、support enumeration、syndrome hashing | Search scaling、certified lower bound、degeneracy |
| 通用 meet-in-the-middle distance | 72 | Half-error tables、syndrome/logical joins | 时间/内存爆炸、duplicate equivalence、witness recovery |
| 大型 QLDPC exact distance | 95 | General combinatorial optimization | 不适合作为通用完成承诺，应以 bounds/witness 为主 |
| Clifford encoder/decoder synthesis | 58 | Symplectic elimination translated to H/S/CNOT/SWAP | Gate convention、qubit order、depth/size 非最优、round-trip proof |
| Encoder verification | 32 | Propagate checks/logicals、group equality | Input/output ancilla convention 和 phase |
| Physical Clifford 的 logical action | 40 | Conjugation、mod-stabilizer coordinates | Logical basis dependence、qubit permutation、failure witness |
| QEC circuit IR：reset/measurement/detector | 55 | Clifford/Pauli updates、classical parity timeline | Measurement semantics、round boundaries、不要污染现有 GateTape |
| Syndrome-extraction verification | 58 | Backward measurement-observable propagation | Ancilla preparation/measurement basis、hook propagation |
| Single-fault detector/logical signature | 62 | Batched Pauli fault propagation、detector XOR | Fault location convention、reset/measurement errors、coarse FFI |
| Detector error model aggregation/export | 30 | Signature sorting/hash aggregation、flat arrays | 前提是 circuit/fault semantics 已冻结；schema 本身较简单 |
| Correlated fault mechanisms | 52 | Multi-Pauli branches、probability aggregation | Exclusive/correlated probability semantics 与组合规模 |
| Circuit distance search | 75 | XOR-combination search under zero-detector constraint | 组合爆炸、certified bounds、具体 fault witness |
| Malignant pair/triple enumeration | 65 | Low-order fault products、signature dedup | `O(F^2/F^3)` scaling、位置等价和概率解释 |
| 低权 logical-noise channel | 52 | Syndrome、decoder correction、logical aggregation | Decoder contract、truncation order、遗漏概率质量 |
| Monte Carlo logical error rate | 48 | Counter-based sampling、decoder calls、statistics | 置信区间、rare events、parallel replay、decoder boundary |
| 通用高性能 decoder | 90 | MWPM/BP/OSD/union-find 等独立算法族 | 已是独立专业项目，且一般 quantum decoding NP-hard |
| Subsystem/gauge code | 65 | Gauge group、center、bare/dressed quotient | 非交换 generators、多种 logical equivalence |
| Gauge fixing | 78 | Stabilizer/gauge changes、logical correspondence | 前后 code space、measurement choice、fault semantics |
| Lattice-surgery merge/split analysis | 85 | Time-indexed stabilizers、detectors、logical tracking | 时序 code deformation、boundary conventions、circuit distance |

## 8. 各功能层的输入、输出与边界

### 8.1 Static code algebra

核心输入是 qubit count 和 stabilizer generators。可选 CSS 输入是 packed `H_X/H_Z`，但内部应统一成 binary symplectic rows。核心输出包括 independent canonical generators、rank、`k`、logical pairs、relation metadata 和稳定的 code fingerprint。

适合的批量分类结果不是 Python object per error，而是 flat arrays：

```text
syndrome_words: shape (batch, syndrome_word_count)
logical_x_bits: shape (batch, logical_word_count)
logical_z_bits: shape (batch, logical_word_count)
flags: shape (batch,)  # stabilizer / syndrome-visible / logical
```

`is_correctable(error)` 不应成为无上下文的核心方法；更准确的能力是 error classification、pairwise distinguishability、error-set correctability，以及给定 decoder correction 后的 residual logical action。

### 8.2 Distance and witness search

Distance API 必须同时返回 lower bound、upper bound、witness 和 completion status。Exact exhaustive search 只适合小 codes；CSS、support-first 和 meet-in-the-middle variants 可以扩大边界，但不能消除 worst-case exponential scaling。

Distance witness 应包含 physical Pauli、logical coordinates、weight 和搜索 provenance。若结果只是一个普通 elimination logical basis 的最小行权，它只能是 upper bound，不是 code distance。

### 8.3 Encoder and logical Clifford analysis

Encoder synthesis 可以利用 symplectic row/column operations 生成一个正确但未必最浅的 Clifford circuit。验证必须检查 stabilizers 和 logical pairs，而不依赖 dense unitary。物理 circuit 若保持 stabilizer group，则其在 quotient `N(S)/S` 上诱导一个 `2k x 2k` binary symplectic map；该 map 是 logical Clifford action 的紧凑输出。

Encoder optimization、hardware connectivity routing 和 minimum-CNOT synthesis 是独立优化问题，不应与“生成一个正确 encoder”混为同一合同。

### 8.4 Fault signatures and detector model

容错电路需要独立、窄化的 QEC circuit IR，至少包含 Clifford gates、Pauli reset/measurement、measurement record references、detector parities 和 logical observables。它不应通过给现有 propagation `GateTape` 添加任意动态 measurement 和 classical control 来隐式扩张后者的语义。

对每个 fault mechanism，compiler 输出

```text
detector_mask + logical_mask + probability + source/location metadata.
```

多个 independent Pauli faults 的 signatures 通过 XOR 组合。相同 detector/logical signature 可以确定性聚合。该表示可以支持 detector-error-model 导出、circuit-distance search、malignant-set enumeration 和外部 decoder integration，而无需在 TenCirPauli 中实现通用 decoder。

### 8.5 Logical noise and decoder boundary

给定 physical error `E`、syndrome `s(E)` 和 decoder correction `C(s)`，residual `C(s)E` 必须属于 normalizer；其 quotient coordinates 给出 logical outcome。聚合 error probabilities 后得到 `P(syndrome, logical Pauli)` 或 marginal logical channel。

低权 exact enumeration 可以按 fault weight 保存数值 coefficients，不需要通用多变量符号引擎。若截断到某个 fault order，结果必须同时报告 retained probability、omitted probability bound 或明确的未覆盖质量。Monte Carlo 路径需要统计置信区间，不能只给一个 point estimate。

### 8.6 Subsystem code and gauge fixing

Subsystem code 的 gauge group 可以非交换，stabilizer 是其适当中心。需要区分 stabilizer、gauge、bare logical 和 dressed logical；普通 `StabilizerCode` 的 `N(S)/S` 分类不足以表达“只改变 gauge subsystem、逻辑信息不变”。因此 subsystem/gauge 应是独立 public type 或明确 adapter，而不是在普通 stabilizer class 中增加大量可选分支。

Gauge fixing 选择额外 commuting gauge observables 并改变 stabilizer group。正确输出必须追踪 fixing 前后的 code parameters、logical correspondence、measurement outcomes 和可能的 correction frame；加入时序 measurement 后会自然接近 code deformation 和 lattice surgery，工程难度显著上升。

## 9. QLDPC 特有的能力与限制

QLDPC 的“LDPC”描述 stabilizer checks 稀疏、每个 check 和 qubit 的 degree 有界或较小，并不保证 logical representatives 稀疏，也不保证 distance、decoder 或 encoder 容易计算。适合 bit-manipulation 技术栈的基础能力包括：

- Sparse/packed CSS check ingestion 与 `H_X H_Z^T=0` validation。
- Independent check rank、encoded `k`、check weight、qubit degree 和 relation basis。
- 任意 deterministic logical basis 及 flat packed read-back。
- Syndrome throughput、batch error classification 和 decoder-output verification。
- Bounded X/Z distance search、upper/lower bounds 和 witnesses。
- Tanner-graph metadata 或 girth 等轻量图统计，但不把完整图优化库设为核心依赖。

不应默认承诺：大型 QLDPC exact distance、minimum-weight logical basis、通用最优 decoder、最浅 encoder 或 construction-family 的自动几何解释。这些问题即使输入 checks 很稀疏，也可能具有很高组合复杂度。

## 10. 与现有工具生态的边界

Stim 已经针对大规模 stabilizer circuit simulation、measurement 和 Pauli-frame sampling 做了高度优化；TenCirPauli 不应以复制其完整 simulator 为差异化目标。PyMatching 等工具专注 detector-based decoding；TenCirPauli 更适合输出稳定 detector/logical signatures 或兼容格式，并验证 decoder correction 的 residual logical action，而不是默认自建完整 MWPM stack。

可能形成差异化的范围是：从任意 Pauli/CSS/subsystem code generators 出发，提供 native logical quotient、deterministic physical logical basis、bounded distance witnesses、logical Clifford action、TensorCircuit-facing Clifford gadget analysis，以及与现有 Pauli/structured Hamiltonian 数据模型共享的 code/fault 分析。这一定位强调 code algebra 与 witness-producing analysis，而不是第二个通用 simulator 或 decoder framework。

## 11. 验证矩阵与代表性工作负载

独立 correctness oracles 应覆盖：

- 三比特 bit-flip/phase-flip codes：syndrome、degeneracy、logical classification 和 decoder ambiguity。
- Five-qubit perfect code：non-CSS symplectic conventions、distance 3 和 canonical logical pairs。
- Steane code：CSS quotient、transversal Clifford action 和 `d_X=d_Z=3`。
- Shor code：dependent-looking structure、degeneracy 与 asymmetric physical representatives。
- Small toric/surface/color instances：geometric string logicals 与 elimination logicals 的 coset equality。
- Bacon–Shor：gauge center、bare/dressed logicals 和 gauge fixing。
- Small hypergraph-product/QLDPC instances：sparse checks、rank dependencies、logical output volume 和 bounded distance。

关键 property tests 包括：

- Canonical generators span 与输入 stabilizer group 相同。
- 每个输出 logical 与所有 stabilizers commute，logical pairs 满足 canonical symplectic relations。
- Error classification 在乘 stabilizer 后 syndrome/logical class 不变。
- 任意低于已知 distance 的 harmful zero-syndrome Pauli 不存在。
- Encoder conjugation把标准 ancilla checks/logicals 映到目标 code，decoder 为其 inverse contract。
- Fault signatures 在 serial/parallel worker counts 下 bitwise replay，组合通过 XOR 与直接 propagation 一致。

性能工作负载应覆盖 `n={64,256,1024+}` 的 packed rows、低密度 CSS checks、batch syndrome、logical basis construction、bounded low-weight search 和大量 fault signatures。应分别记录 setup、steady throughput、retained bytes、search frontier、lower/upper bound、worker count 和 witness verification；不可只报告 qubit 数或单个 wall time。

## 12. 主要风险

### 12.1 Canonical 不等于 optimal

一个 deterministic logical basis 可以很快产生，但可能高权、非局域且不适合作为 physical implementation。API 和文档必须避免把 canonical output 描述成 minimum-weight 或 geometry-aware output。

### 12.2 Distance 和 decoder 的组合爆炸

Packed bit operations 可以显著提高常数和可处理规模，但不能改变一般 exact distance/decoding 的 worst-case hardness。所有搜索型结果必须有 completion/bound semantics 和 witness verification。

### 12.3 Circuit semantics 比静态 code algebra 更难

Reset、measurement record、detector parity、round boundary、leakage、correlated mechanisms 和 observable sign conventions 会迅速扩张语义。Static code core 不应提前携带这些字段；fault circuit 应有独立 IR 和 owner。

### 12.4 Subsystem code 不能伪装成普通 stabilizer code

Gauge operators、bare/dressed logical 和 gauge-only errors 需要独立 quotient 结构。把 gauge generators 强行当作 commuting stabilizers 会产生错误 code dimension 和 error classification。

### 12.5 与成熟 simulator/decoder 的重复建设

没有明确差异化 workload 时，不应复制 Stim 的 circuit sampler 或 PyMatching 类 decoder。Flat export、cross-check 和 decoder boundary 比内置所有算法更符合当前项目范围。

## 13. 参考资料

1. Daniel Gottesman, “Stabilizer Codes and Quantum Error Correction,” PhD thesis, 1997, [arXiv:quant-ph/9705052](https://arxiv.org/abs/quant-ph/9705052).
2. A. R. Calderbank, E. M. Rains, P. W. Shor, and N. J. A. Sloane, “Quantum Error Correction via Codes over GF(4),” [arXiv:quant-ph/9608006](https://arxiv.org/abs/quant-ph/9608006).
3. Barbara M. Terhal, “Quantum Error Correction for Quantum Memories,” Rev. Mod. Phys. 87, 307 (2015), [arXiv:1302.3428](https://arxiv.org/abs/1302.3428), DOI: [10.1103/RevModPhys.87.307](https://doi.org/10.1103/RevModPhys.87.307).
4. Nikolas P. Breuckmann and Jens Niklas Eberhardt, “Quantum Low-Density Parity-Check Codes,” PRX Quantum 2, 040101 (2021), [arXiv:2103.06309](https://arxiv.org/abs/2103.06309), DOI: [10.1103/PRXQuantum.2.040101](https://doi.org/10.1103/PRXQuantum.2.040101).
5. Min-Hsiu Hsieh and François Le Gall, “NP-hardness of decoding quantum error-correction codes,” Phys. Rev. A 83, 052331 (2011), [arXiv:1009.1319](https://arxiv.org/abs/1009.1319), DOI: [10.1103/PhysRevA.83.052331](https://doi.org/10.1103/PhysRevA.83.052331).
6. Craig Gidney, “Stim: a fast stabilizer circuit simulator,” Quantum 5, 497 (2021), [arXiv:2103.02202](https://arxiv.org/abs/2103.02202), DOI: [10.22331/q-2021-07-06-497](https://doi.org/10.22331/q-2021-07-06-497).

## 14. 文档使用说明

本文用于后续讨论时统一术语、范围和难度基线。若某一能力被正式选择，应另写聚焦的 specification，冻结 qubit/order/phase、input/output ABI、exact/bounded semantics、independent references、representative workloads 和 non-goals；本文本身不应被当作所有表中能力都要实现的承诺。
