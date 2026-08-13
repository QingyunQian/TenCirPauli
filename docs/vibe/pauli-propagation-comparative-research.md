# Pauli propagation 能力对比与补全路线

状态：专题调研；不是当前实现合同。调研日期：2026-08-13。

Owner scope decision after this research: Phase 12 implements TensorCircuit-compatible common noise channels and their native scalar gradients while retaining existing static one- and two-qubit PTM forward propagation. Runtime/parameterized PTM values and PTM-entry VJPs are deferred to a workload-gated follow-up. The Phase 12 specification is authoritative where the recommendations below describe the broader parameterized-PTM route.

## 结论摘要

仓库里已经存在一份 [TenCirPauli 与 PauliPropagation.jl 的 28-qubit VQE 对比实验](../../examples/research/pauli_propagation_julia_vqe/README.md)，也存在一条关于“参数化 PTM + native entry VJP + JAX custom-VJP”的 feature-incubator 设计片段。此前缺少的是针对 PauliPropagation.jl 非幺正、噪声和 PTM 反传方案的集中调研；本文补上这一层，并把可落地的 TenCirPauli 补全路线写成接口和语义建议。

PauliPropagation.jl 当前的优势不是一个单独的“更快传播 kernel”，而是它把几类局部线性变换都纳入了 Pauli-sum dispatch：Pauli rotation 和 Clifford gate 用专门规则，Pauli noise 和 amplitude damping 用内置局部规则，任意固定局部线性 map 可以用 `TransferMapGate`，虚时 Pauli rotation 则作为 Schrödinger 侧的非幺正 sandwich update。它的边界同样明确：`ImaginaryPauliRotation` 不定义 Heisenberg 语义，`rewindgradient` 只接受 PauliRotation 参数化门并明确排除 noise。

TenCirPauli 当前已经有统一的 dynamic Heisenberg recurrence、聚合后 weight projection、静态 real PTM 和 rotation 的 native reverse；因此最自然的补全不是引入一套平行的 noise engine，而是把当前 `CustomPtm` 扩展成固定布局的 runtime local map，并复用同一套 transitions、aggregation、checkpoint/replay 和 reverse machinery。内置 noise 应作为这个 local-map 层的优化构造器，虚时演化应作为带归一化终端的特殊非幺正 Pauli map。

建议的优先次序是：先完成参数化 PTM 的 native VJP，随后加入 Pauli noise 和 amplitude damping 的 native fast path，再加入带显式归一化的 imaginary Pauli rotation。这样参数化噪声、用户自定义局部线性 map 和虚时演化共享 ABI，但不会把“物理 CPTP channel”“一般线性 map”和“非归一化 imaginary-time update”混成一个含义不清的 `ptm()` API。

当前范围决策：暂不把 imaginary-time propagation 纳入 TenCirPauli 的近期主线。PauliPropagation.jl 的 imaginary-time 局部 `cosh/sinh` 规则本身是有数学内容的，但当前公开路径主要是 Schrödinger state-side 的 normalized update，缺少与我们 observable-first Heisenberg engine 对应的 normalized value/gradient 证据；它也不是当前 noise/PTM/JAX VJP 需求的必要依赖。近期只推进参数化 PTM 的 native VJP、常见 trace-preserving noise 和其 JAX 组合；imaginary-time、Schrödinger Pauli-state API、general non-TP map 和 PTM-SPPS 保留为后续独立专题。

## 现有材料与调研范围

本仓库已有的 Julia 对比实验使用 PauliPropagation.jl v0.8.0，比较相同的 28-qubit TFIM VQE、`max_weight=5`、native/JIT 边界和 value-and-gradient 调用。该实验已经发现一个重要语义差异：Julia 的 `rewindgradient` 在 weight projection 后与它自身截断 value 的 finite difference 不一致，而 TenCirPauli 的 frozen-support reverse 在该实验中与自身 projected value 的 finite difference 一致。因此本文不重新评价已有 benchmark 数字，而是补充其没有覆盖的能力轴。

本次外部核查以 PauliPropagation.jl 当前 `main` 分支的 README 和源码结构为依据，并以 v0.8.0 作为仓库已有 benchmark 的固定版本。当前包文档由 Documenter 生成，README 说明了 Heisenberg/Schrödinger 方向、noise、imaginary rotation、TransferMap、gradient 和 Monte Carlo 能力；源码核查了 gate 定义、局部 apply 规则、picture conversion、TransferMap 和 `rewindgradient` 的约束。由于上游仍在开发，涉及“当前 main”的判断应在实现前重新 pin 一个 commit。

## 能力对照

| 能力 | PauliPropagation.jl | TenCirPauli 当前状态 | 判断 |
| --- | --- | --- | --- |
| Pauli rotation、Clifford 和动态 Pauli-sum recurrence | 有；按 gate type 多重派发，默认对 observable 做 Heisenberg adjoint propagation | 有；Rust dynamic recurrence，支持首批 Clifford/Pauli rotation | 两边都有，现有截断和梯度语义不同 |
| weight/系数截断 | `max_weight`、`min_abs_coeff`、frequency/sine truncation、custom truncation 和 Monte Carlo resampling | 聚合相同 Pauli word 后按 `max_weight` projection；不采用 coefficient cutoff/top-k | 不应为了对齐而复制 Julia 的截断合同 |
| 固定局部 PTM/transfer map | `TransferMapGate` 接受固定 TransferMap、PTM 或 unitary matrix；列存储稀疏输出 | `GateTape.ptm()` 接受固定一/二 qubit real `float64` PTM，native 编译为稀疏 transitions | 当前已有同层 forward primitive，但缺少 runtime payload |
| 内置 Pauli noise | 有 `DepolarizingNoise`、Pauli X/Y/Z noise 和 `DephasingNoise`；Pauli basis 中只缩放系数 | 没有一等 noise gate；静态 PTM 可以手工表达一部分 noise | 这是明确的 native completeness gap |
| amplitude damping | 有内置局部规则：X/Y 乘 `sqrt(1-gamma)`，Z 分裂成 `(1-gamma)Z + gamma I` | 没有内置实现；静态 PTM 可表达 forward map | 应作为第二个 native noise fast path 和 runtime PTM 测试样例 |
| 任意参数化 local map | `TransferMapGate` 本身是 StaticGate；`rewindgradient` 不是参数化一般 map 的 VJP 接口 | 静态 PTM 没有 parameter gradient | TenCirPauli 可以在这个缺口上建立更清晰的 fixed-layout runtime PTM contract |
| 虚时 Pauli rotation | 有 `ImaginaryPauliRotation`，只允许 `heisenberg=false` 的 Schrödinger propagation；默认可按 identity coefficient 归一化 | 当前 propagation 只计算 Heisenberg observable expectation，没有 imaginary-time gate | 两边的语义都不能简单等同于“一个普通 unitary PTM” |
| 非幺正后的归一化 | Julia 在 imaginary Pauli update 后可按 identity coefficient 归一化；一般 Heisenberg noise 仍按线性 map 传播 | 当前没有非幺正 normalization terminal | 需要显式 numerator/denominator 或 state-path contract |
| native/custom reverse | `rewindgradient` 仅 PauliRotation 参数化门；noise 明确不支持 | rotation 有 retained-edge frozen-support reverse；静态 PTM 只有 transpose action | 参数化 PTM VJP 是最有价值的共同补全点 |
| JAX bridge | Julia 侧主要依赖 ForwardDiff/ReverseDiff 和自定义 `rewindgradient`，不是 JAX PTM VJP | 已有 scalar `pure_callback` + custom VJP，但输入目前是 angle vector | 可自然扩展为 `(angles, flat_ptm_values)` 双输入 VJP |

## PauliPropagation.jl 的方案

### 局部 gate type 与 picture 方向

PauliPropagation.jl 把 gate 分成 `ParametrizedGate` 和 `StaticGate`。Pauli rotation、imaginary Pauli rotation 和 parametrized noise channel 是参数化类型；Clifford、TransferMap 和冻结后的 gate 是静态类型。`propagate()` 默认把按 Schrödinger 顺序写下的 circuit 反转，并在 Heisenberg 方向作用于 observable；`heisenberg=false` 时则走 Schrödinger 侧。

这不是简单的矩阵乘法层。每个 gate 都可以给 `applytoall!` 或 `applymergetruncate!` 提供专门实现，而通用 propagation cache 负责主 sum、辅助 sum、merge、truncate 和 vector backend。这样 Pauli rotation 的两分支、amplitude damping 的两分支、Pauli noise 的单纯 coefficient damping 和 Clifford 的 lookup map 可以各自走不同的低层路径。

### Schrödinger 侧的实际意义

在 `heisenberg=false` 时，输入的 Pauli sum 被当作 forward-propagating state/operator，例如密度矩阵的 Pauli 展开；库按照 Schrödinger 顺序应用 channel 或 gate，最后可以对这个 propagated Pauli sum 做 overlap 或其他支持的 state-side evaluation。它仍然不是 dense statevector simulator，Pauli-state 的 term 数同样可能增长；它的价值主要在于需要显式 forward state、需要处理 state-dependent normalization，或者需要做 Heisenberg 单 observable 形式不自然的非幺正 update。

picture conversion 不是简单地把同一个 gate 名字搬过去：PauliRotation 在 Schrödinger side 要反转角度，Clifford 要使用 transpose lookup map，PauliNoise 因为 Pauli conjugation channel 自伴随而保持不变，ImaginaryPauliRotation 保持 `tau` 不变。当前上游没有为 AmplitudeDampingNoise 和一般 TransferMapGate 提供完整的 Schrödinger conversion，因此不能把 README 中的“支持 Schrödinger propagation”理解为所有 gate 都同时支持两个 picture。

### Pauli noise 的支持方式

`PauliNoise` 是一个抽象的、在 Pauli basis 中对角的 noise 类型。Depolarizing、Pauli X/Y/Z noise 和 Z-noise alias `DephasingNoise` 都只需判断当前 term 在 noise site 上的 Pauli label，然后把受影响的 coefficient 乘以 `1-lambda`。因此它们不会产生新 Pauli word，也不需要 merge；vector backend 可以直接并行缩放 coefficient。

`AmplitudeDampingNoise` 不是 Pauli-diagonal map，但仍然有很小的局部规则。对 Heisenberg observable，X 和 Y 分量乘 `sqrt(1-gamma)`，Z 分量变为 `(1-gamma)Z + gamma I`，所以只有遇到 Z 时才产生一个 I 分支。这个规则正是一个一 qubit PTM 的稀疏列表示，但直接写专门 kernel 可以避免每个 term 都扫描完整 4×4 矩阵。

这两类 noise 在“底层数学”上都可以写成 Pauli-transfer map，但当前上游实现不是对所有 gate 统一调用一个 dense PTM kernel。Pauli noise 走 diagonal coefficient scaling，amplitude damping 走 X/Y scaling 加 Z-to-I/Z branching，只有一般 `TransferMapGate` 才走压缩的 PTM-column lookup。picture 支持也不完全对称：当前 `heisenbergschrodinger.jl` 为 `PauliNoise` 提供 Schrödinger conversion，但没有为 `AmplitudeDampingNoise` 提供对应的 `_toschrodinger`；README 的 Schrödinger 支持列表也只列 PauliRotation、CliffordGate 和 PauliNoise。因此 amplitude damping 当前应理解为主要支持 Heisenberg observable propagation，而不是两个 picture 都完整支持。

“Pauli noise 不产生新 Pauli word”不等于物理上没有 Kraus branch。以 Pauli channel 为例，`E(rho)=sum_P p_P P rho P` 确实有多个随机 Pauli branch，但每个 Pauli basis operator `Q` 都是这个 channel 的 eigenoperator：`E^dagger(Q)=chi_Q Q`。传播的是已经对 Kraus outcomes 做完 ensemble average 的 density operator/observable，因此多个 branch 重新落回同一个 `Q`，只表现为 coefficient 乘以 `chi_Q`，不会在 sparse Pauli dictionary 中新增四个不同 word。只有非对角 PTM 的 input column 同时含有多个 output Pauli 时，才会真正产生多个 Pauli-word branches。

这些 noise 参数由 gate 运行时传入，并且 constructor 对 `lambda`/`gamma` 做 `[0,1]` 检查。它们首先解决的是 forward propagation；PauliPropagation.jl 自定义的 `rewindgradient` 文档明确写着 noise channels 不支持，因为 backward sweep 需要 undo gate，而 noise 一般不可逆。不能把“forward 支持 noise”误读成“noise 已有完整 native reverse”。

### TransferMap 与 PTM 的支持方式

`TransferMap` 是按输入 Pauli column 存储的压缩 lookup table。每个 column 是若干 `(output_pauli, coefficient)`，`TransferMapGate` 再把局部 Pauli index 映射回任意物理 qubit。它可以从显式 columns、PTM 或 computational-basis unitary matrix 构造；局部 map 的维度是 `4^k × 4^k`，非零项在传播时直接展开。

这个设计与 TenCirPauli 当前 `GateOperation::CustomPtm` 的 transitions 很接近，但有两个重要差别。第一，Julia 的 TransferMap 是静态 gate，参数变化时需要重新构造或重新冻结；第二，Julia 的 `calculateptm` 可以从 matrix 生成 PTM，但不等于为任意 runtime PTM 提供参数化 reverse。TenCirPauli 不需要复制其 Python/Julia 对象层，只需保留现有 row-major `R[output, input]` 约定并增加一次性 batched runtime payload。

### 虚时 Pauli rotation 的支持方式

`ImaginaryPauliRotation(P)` 实现的是 `A rho A`，其中 `A = exp(-tau P/2)`，而不是 unitary conjugation。对 Pauli term `Q`，如果 `P` 与 `Q` anti-commute，则 `A Q A = Q`；如果 commute，则会产生两项 `cosh(tau) Q` 和一个 `sinh(tau)` 乘 Pauli-product 的分支。源码中的 vector specialization 只对 commuting terms 做 branch，这一点与普通 unitary Pauli rotation 对 anti-commuting terms 做 branch 正好相反。

Julia 在 `applymergetruncate!` 后默认用 propagated state 的 identity coefficient 做 normalization。这相当于在 Pauli-sum state 表示中保持归一化，避免 imaginary-time coefficient 整体指数增长；也可以用 `normalize_coeffs=false` 关闭。由于 normalization 是 state-dependent 的非线性操作，源码主动禁止 `ImaginaryPauliRotation` 的 Heisenberg conversion，并在 README 中注明它只能用于 `heisenberg=false`。

因此它支持的不是任意非幺正电路求解器，而是一个很具体、很实用的 imaginary-time state update：局部 Pauli generator、可截断的 Pauli-sum state、每步可选归一化。这个边界值得保留；如果 TenCirPauli 将 imaginary update 塞进现有“observable-only expectation”而不暴露 normalization 语义，结果会在非幺正 map 上产生歧义。

### 梯度方案和它的边界

`rewindgradient` 的文档约束是：只有 PauliRotation 可以参数化，其他 gate 必须是静态或 frozen，noise channel 无论是否 frozen 都不支持，`overlapfunc` 必须对 Pauli-sum coefficients 线性。它先做一次正常 truncated forward，再在反向顺序中重放 gate；对每个 PauliRotation，用 generator commutator 与 dual sum 做局部梯度贡献。

这个 reverse sweep 通过支持相交和 operator cache 来控制开销，但它依赖“能 undo gate”的假设。对 noise 和 imaginary-time map，这个假设通常不成立或需要带归一化的 quotient rule，因此 Julia 的这一接口不能直接作为我们参数化 PTM reverse 的参考实现。已有本地实验还显示，重复 weight projection 后，Julia `rewindgradient` 与 `propagate` 自身 projected value 的 finite difference 可以不一致；TenCirPauli 应继续坚持“reverse target 必须和实际 executed projected forward map 一致”的更严格合同。

## TenCirPauli 当前缺口的准确表述

当前 `GateTape.ptm()` 已经让用户可以把一/二 qubit 的 real finite `float64` PTM 放进 native forward recurrence，因此“没有噪声”不能简单表述为“不能表达 noise”。静态 depolarizing、dephasing 或 amplitude damping PTM 原则上都能手工传入。真正的缺口有四个：没有 built-in noise descriptor 和专门 kernel；PTM entries 不能在每次 runtime call 中变化；static PTM 的 reverse 只做 transpose action、不返回 PTM-entry gradient；非幺正 map 的 normalization/unnormalized semantics 没有公共 contract。

当前 JAX bridge 位于 `python/tencirpauli/jax_support.py`。它把 runtime angle vector 作为一次 `jax.pure_callback` 输入，native 返回 scalar value 和 angle gradient，custom VJP 的 backward 是 `cotangent * gradient`。`PropagationCircuit.expectation_jax()` 也要求 observable Hermitian。这个设计已经证明“一个粗粒度 callback + native gradient + JAX custom VJP”可行，但 PTM 目前在 tape construction 时被复制成 static payload，所以没有可供 JAX 链式组合的 PTM cotangent。

## 建议的统一补全模型

### 第一层：固定局部布局的 runtime Pauli map

保留现有 static `ptm()` API，并增加一个参数化 local-map 描述。每个 map 固定作用于一或二个 wire，固定局部维度 `d=4^k`，固定 flat slice offset；一次 execution 把所有 gate 的 runtime entries 放入一个 contiguous `float64` buffer。第一版仍采用当前 ABI 的 row-major 语义：`R[out * d + in]` 表示输入 Pauli basis column `in` 映射到输出 basis row `out` 的系数。

native tape 只保存每个 map 的 wires、dimension、offset 和 fixed layout metadata；每次调用传入 flat PTM values，Rust 在 GIL-released section 内完成 transitions、term aggregation 和 reverse。为了避免动态 sparsity 改变 compiled layout，第一版可以接受 dense fixed-layout PTM；runtime 的 zero entry 仍然合法，但不会改变 ABI。若 profile 证明必要，再增加 fixed sparse pattern，不能让每次调用重新构造 Python nested lists。

这一层应覆盖三种输入：上游 JAX 函数生成的 unitary-derived PTM，参数化 Pauli/depolarizing/amplitude-damping PTM，以及用户提供的 Hermiticity-preserving local linear map。它不自动声称输入是 CPTP channel；shape、finite、wire order 和 Hermitian-basis realness 是第一版的必检条件，CPTP/CP 验证可以作为显式 validation mode。

### 第二层：内置 noise fast path

在统一 local-map 语义之上加入一等 gate constructor，而不是要求用户每次手工构造 4×4 matrix。Pauli noise 直接执行按 site label 的 coefficient damping；amplitude damping 直接执行 X/Y scale 与 Z-to-I/Z split。参数可以是 static value 或 runtime slot，前者复用现有静态 transitions，后者进入同一个 local VJP。

实现上建议采用“一个语义、两类 kernel”：所有 noise 先有同一个 PTM/transition reference，保证 orientation、merge、projection 和 VJP 语义一致；Rust 再为 Pauli-diagonal noise 写无 branch 的 coefficient-scale fast path，为 amplitude damping 写直接的 scale-plus-single-split fast path。常见 noise 这样可以少一次完整 local-column 扫描、少一些 temporary transition handling，并避免本来不会产生新 word 的 Pauli noise 进入 merge；但是否值得保留专门 kernel 仍要用 release-mode end-to-end profile 判断，不能仅凭 4×4 矩阵大小宣称收益。参数化 kernel 的 gradient 也应与通用 `dL/dR` reference 做 differential test，而不是维护两套未经校验的数学实现。

建议首先支持 Pauli X/Y/Z noise、dephasing/depolarizing 和 amplitude damping。它们覆盖“Pauli diagonal noise”和“非 Pauli diagonal 但局部稀疏”的两种代表，并且可以用 dense one-qubit reference 精确验证。更复杂的 Kraus channel 不应在第一版扩展成一套 Kraus-level engine；只要先转换成 Hermitian Pauli-basis PTM，就能进入统一 local-map 路径。

对物理约束要区分三件事：CPTP channel 的 Schrödinger 语义、其 Heisenberg adjoint 的 PTM 语义，以及一般 real linear map。对 TP channel，Heisenberg map 满足 `E^dagger(I)=I`；这与 Schrödinger channel 是否 unital 是不同条件。不要因为 amplitude damping 在 Schrödinger picture 非 unital，就错误地给其 Heisenberg propagation 强行引入 normalization denominator。

非幺正、非 unital 和非 trace-preserving 不是同一个概念。Amplitude damping 是非幺正、Schrödinger-side 非 unital，但仍是 trace-preserving channel，所以 Heisenberg-side 有 `E^dagger(I)=I`，普通 observable expectation 不需要分母。Imaginary-time `rho -> A rho A` 则一般不是 trace-preserving；postselected Kraus branch、no-click/loss conditioned map 和某些 unnormalized effective evolution 也可能如此，这些才需要 denominator guard。对于任意线性但 trace-non-preserving circuit，可以把 `O` 与 `I` 用同一个 Heisenberg recurrence 分别传播，返回 `N/D`；若 `D` 接近零，必须显式报错或返回带状态的 invalid result，不能静默产生 NaN。

### 后续专题：带归一化终端的 imaginary Pauli rotation

如果目标是沿用当前 Heisenberg observable engine，推荐把 imaginary Pauli evolution 建模成线性但非归一化的 local map，再显式传播 numerator 和 denominator。给定初态 `rho`、observable `O` 和整体非幺正 map `E`，返回

```text
numerator   = Tr[rho E^dagger(O)]
denominator = Tr[rho E^dagger(I)]
expectation = numerator / denominator
```

对 `A=exp(-tau P/2)` 的 imaginary Pauli update，`E(rho)=A rho A`，所以 `E^dagger` 仍由同一个 `A Q A` 局部规则给出。需要同时传播 observable numerator `O` 和 identity denominator `I`；对 commute 分支使用 `cosh/sinh`，对 anti-commute 分支保持原项。CPTP noise 通常保持 denominator 为一，因此可以走普通单 observable 快路径；imaginary map 则必须显式暴露 ratio 语义。

如果 `N` 和 `D` 的 native reverse 分别给出 `dN`、`dD`，归一化 value 的梯度是

```text
d(N / D) = (dN * D - N * dD) / D^2
```

这比在每个 gate 后把系数除以 identity coefficient 更适合当前 Rust Heisenberg architecture，因为 normalization 不会被隐藏在 recurrence 内；同时它与 Julia 的 normalized Schrödinger path 在无截断时应当给出相同的最终 normalized expectation。有限 `max_weight` 时必须明确是对 N/D 两条 recurrence 都做同一 projection，然后把 projected ratio 作为定义好的近似目标，并用该目标的 finite difference 验证 gradient。

如果未来需要返回每一步归一化 state、做 state-dependent nonlinear observables 或提供 imaginary-time sampling，则应另建 Schrödinger Pauli-state API，不要把这些语义偷偷塞进 `PropagationEngine.expectation()`。

### 为什么上游没有直接传播 identity 再除

对线性且 trace-non-preserving 的 imaginary map，传播 `O` 和 `I` 后计算 `N/D` 在无截断的数学上完全成立；因此 PauliPropagation.jl README 中“还不知道如何在 Heisenberg picture 处理 normalization”的说法应理解为当前 API/实现边界，而不是理论上的不可能。上游现有 `propagate` 接受一个 Pauli sum，`overlapfunc` 也要求对这个 sum 的 coefficients 线性；`N/D` 需要一个 paired observable terminal、非线性 quotient 和相应的 gradient contract，这不是给 `overlapfunc` 增加一个参数就能保持原有接口的变化。

此外，Julia 的 imaginary-time kernel 是“apply、merge、按 identity coefficient normalization、再 truncate”的 state-side 流程。若 truncation 只有 weight projection，逐步 normalization 与最后做 ratio 在理想线性算术下可以等价；但它还支持 `min_abs_coeff`、frequency/sine truncation 和 custom truncation，系数整体缩放会改变这些保留决定。逐步 normalization 也控制 `cosh/sinh` 造成的数值增长，避免最终 numerator/denominator 在传播结束前已经溢出或下溢。因此要保持它现有 Schrödinger 语义，不能未经定义地用一条未归一化 Heisenberg recurrence 替换。

TenCirPauli 的最小实现甚至不需要新增 normalized terminal：把 imaginary map 定义为未归一化的线性 local map后，用户可以用现有 `PropagationEngine` 分别传播 `O` 和 `I`，或者把两者放入现有 `PropagationBatch`，然后在 Python 外部计算 `N/D`；gradient 也可以在外部按 `(dN*D-N*dD)/D^2` 组合。未来可以再提供 convenience terminal，但它不是支持该能力的必要条件。如果需要逐步归一化 state，再单独提供 Schrödinger state API；这样不会改变现有单 observable、线性 terminal 的 `expectation()` 合同。

这个外部 quotient 方案的主要风险是截断后的条件数，而不是除法本身。若 `N_hat=N+delta_N`、`D_hat=D+delta_D`，则一阶误差近似为 `delta(N/D) ≈ (delta_N-(N/D)delta_D)/D`；当 `D` 很小、或者 `D` 是多个 Pauli expectation 的强抵消结果时，两个独立 truncation error 会被 denominator 放大。梯度的 quotient rule 还含有 `D^2`，因此 near-zero denominator 必须是显式 invalid case。上游源码没有把“quotient instability”单独写成拒绝理由，只明确说 Heisenberg normalization 尚未处理；但其 Schrödinger-side 每步 normalization 的确同时解决了动态范围和 coefficient-truncation scale 问题。

如果 TenCirPauli 采用 `O/I` 外部组合，第一版应至少做共同 rescaling、denominator guard 和误差诊断：在 native 内可以对 N/D 两条 recurrence 施加相同的正标量 rescaling，并记录 rescaling factor，最终 quotient 中该公共因子自动抵消；不能对两条 recurrence 独立 rescale 后忘记还原 scale ratio。当前只有 weight projection 时，公共 rescaling 不会改变保留 word；若未来加入 absolute coefficient cutoff，则必须冻结 relative-cutoff 或 per-step normalization 语义。测试应同时报告 `D`、`abs(D)`、`N/D`、finite-difference error 和 rescaling metadata，而不能只比较最终 ratio。

## 参数化 PTM 的 native VJP 设计

### 局部微分公式

设一个局部 PTM 对 Pauli coefficient vector 的 forward 作用为

```text
y[out] = sum_in R[out, in] * x[in]
```

设 reverse 从该 gate 后方传回的 cotangent 是 `lambda_y[out]`。在当前 real Pauli basis 合同下，PTM-entry 的局部 VJP 是

```text
dL/dR[out, in] += lambda_y[out] * x[in]
lambda_x[in]     = sum_out R[out, in] * lambda_y[out]
```

其中 `x` 是该 gate 的实际 retained forward input coefficient，`lambda_y` 是实际 retained reverse output cotangent；所有 global Pauli-word collisions 都必须在 native aggregation 后按确定顺序累加。对于参数化 rotation，现有 analytic local derivative 仍可保留；对于 runtime PTM，Rust 只需要返回 flat `dL/dR`，不需要知道上游 `R(theta)` 的 Jacobian。

### 推荐的 coarse ABI

概念上，native terminal 应从当前

```text
value_and_grad(angle_values) -> value, grad_angles
```

扩展为

```text
value_and_grad(angle_values, ptm_values) -> value, grad_angles, grad_ptm_values
```

其中 `ptm_values` 和 `grad_ptm_values` 使用同一 flat contiguous layout，按 gate offset 取回每个 local map 的 entry cotangent。若一次 circuit 中只含 PTM 参数，也应允许 angle buffer 为空；如果多个 gate 共享同一个上游参数，sharing 不在 Rust 中复制成多个 Python parameter，而由 JAX 上游函数的 VJP 归约。

native reverse 必须复用现有 checkpoint/replay 和 retained-edge 规则：静态 PTM 仍只有 transpose action；runtime PTM 除了 transpose action，还对本次 forward 实际使用的 map entries 累积 `dL/dR`。参数化 PTM 的 zero entry 不得因为数值恰好为零而改变 fixed layout；否则 JAX `jit` 下的 shape/trace contract 会不稳定。

### JAX 组合方式

JAX 侧不应在 Python 中对每个 PTM entry 调 native，也不应在 Python 中显式构造 dense Jacobian。应把 native value-and-VJP 做成两个 JAX 输入的 coarse custom-VJP primitive，然后让普通 JAX transformation 组合上游 PTM 函数：

```python
def objective(theta):
    runtime_ptms = ptm_fn(theta)  # fixed shape: [n_maps, d, d]
    return native_expectation_jax(
        theta,
        runtime_ptms,
        engine=engine,
        checkpoint_interval=checkpoint_interval,
    )
```

这个 primitive 的 forward callback 一次接收 `theta` 和 flat `runtime_ptms`，一次返回 value、angle gradient 和 PTM-entry gradient。其 custom VJP 对两个输入返回 `cotangent * grad_angles` 和 `cotangent * grad_ptm_values`；外层 `ptm_fn(theta)` 的 JAX VJP 会自动把 PTM-entry cotangent 链回 `theta`，最终得到“直接 angle 路径 + PTM 生成路径”的总梯度。这样 Rust core 不依赖 JAX，JAX 也不需要了解 Pauli term、transition 或 checkpoint。

第一版应限制 runtime PTM 的 rank/layout 为固定 shape，并要求 entry dtype 为 real `float64` 或在 JAX x64 关闭时遵循现有 float32 bridge 的显式转换规则。需要记录的验证包括 `jit`、`grad`、共享上游参数、多个 PTM gate、同时存在 angle 和 PTM 参数、zero entries 不改变 layout、并发调用和 batch scalar objective。complex PTM 可以另立 contract；当前 Hermitian Pauli basis 的 real PTM 足以覆盖 unitary-derived map 和常见 Hermiticity-preserving noise。

## 截断、梯度和 SPPS 的边界

参数化 PTM 和 imaginary map 一旦加入，最容易出错的是把 forward 支持误认为所有 gradient/随机传播路径都自动支持。建议继续维持三个独立条件：

- deterministic value 支持一个 local map，不代表它已拥有 parameter gradient；
- deterministic native gradient 支持一个 local VJP，不代表它适合 SPPS；
- SPPS 支持一个 gate，还需要合法 transition sampling distribution、importance weight、parameter derivative 和稳定的 local PAD rule。

因此第一版 PTM/noise/imaginary 补全不应自动扩展 SPPS。尤其是一般 PTM 可能有负 transition coefficient，不能直接把 PTM column 当 probability distribution；imaginary `cosh/sinh` 分支还会带来非 unitary coefficient growth。要做随机化，需要独立的 sampling semantics 和 variance/certificate 设计。

截断方面，TenCirPauli 应继续坚持现有“same Pauli word 先 aggregate，再按 weight projection”的合同，不复制 Julia 的 `min_abs_coeff`、frequency/sine truncation 或 `max_size` resampling。对 runtime PTM 和 imaginary ratio，所有 correctness tests 都要同时检查 exact/no projection 和 projected value 的自洽 gradient；不能只比较 shape、term count 或一次 forward value。

## 当前实现顺序与验收证据

### 统一 runtime PTM 与 VJP

先扩展 native tape 的 local-map metadata 和 flat runtime payload，保持 static PTM 向后兼容；再把 reverse 从 static transpose 扩展为 PTM-entry cotangent。首批只支持一/二 qubit real fixed-layout PTM，不引入任意动态稀疏结构、不引入复杂 dtype、不在 Python 保留平行 nested-list state。

验收至少包括一/二 qubit dense reference、unitary-derived PTM、depolarizing 和 amplitude-damping parameterization、finite difference 与 dense reverse、多个 gate 共享 runtime 参数、静态 zero entries、checkpoint interval、并发 deterministic repeat，以及一次完整 Python/JAX boundary 的 release-mode benchmark。

### 内置 noise

在 runtime PTM 通过后加入 Pauli noise 和 amplitude damping 的 native constructors/fast paths。它们应与手工构造的 PTM 得到 bitwise-stable 或在文档容差内相同的 value，并分别与 dense channel reference 对照。对 TP noise，应验证 Heisenberg identity column/normalization 不被错误改变；对 amplitude damping，应验证 Z 到 I/Z 的方向和 wire ordering。

验收还需要覆盖 parameter gradient：Pauli damping 的 derivative、amplitude damping 的 `sqrt(1-gamma)` derivative 和 Z split derivative 都要与 finite difference 对照，并测试 gamma/lambda 边界。SPPS 不因这些 tests 通过而自动开启。

### 暂不实施：Imaginary Pauli rotation

该方向暂不进入当前实现。若后续出现明确的 imaginary-time state-preparation workload，再先实现 unnormalized linear local map 或专门 imaginary rotation kernel，并用一/二 qubit dense `A rho A` reference 校验 commute/anti-commute 分支、归一化 ratio 和 tau gradient；denominator 接近零、非有限 tau、projection 后 denominator 为零必须定义为显式错误，而不是静默返回 NaN。

如果需要与 Julia 直接对照，必须固定它的 `normalize_coeffs`、picture、截断时机和 coefficient convention；不能只把同名的 `ImaginaryPauliRotation` 放进电路后比较最终数字。无截断时应比较 normalized expectation；有截断时应分别报告 projected ratio 和 gradient contract。

## 尚需冻结的语义决策

在正式实现前，需要冻结 PTM 是始终表示 Heisenberg map 还是允许每个 gate 声明 picture；当前最安全的是沿用 TenCirPauli 现有 convention，`PropagationCircuit.ptm()` 接受 Heisenberg local map，并为 Schrödinger state API 另设明确入口。

需要冻结 row/column 方向、local wire ordering、global Pauli endian、flat buffer row-major layout、runtime parameter 的 gate occurrence 顺序和 shared-parameter 归约规则。任何一项不明确，都会让 dense reference、JAX VJP 和 Julia 对照同时变得不可靠。

需要决定 runtime map 是否只做 shape/finite/Hermiticity-preserving validation，还是提供显式 CPTP validation。建议默认只做便宜且确定的结构校验，物理性检查作为 opt-in，因为一般 linear map 和 imaginary-time map 本来就不应被误判为 CPTP。

需要决定 `PropagationEngine` 是否直接增加 normalized numerator/denominator terminal，还是新建一个明确标注 non-unitary 的 facade。建议不要改变现有 `expectation()` 的含义；新能力应至少在名称或返回类型上暴露 numerator、denominator 和 normalized value，避免现有 Hermitian observable users 获得隐式 quotient。

## 参考

- [PauliPropagation.jl README](https://github.com/SparqleSim/PauliPropagation.jl/blob/main/README.md)：当前包的 Heisenberg/Schrödinger、noise、imaginary rotation、gradient 和 Monte Carlo 边界。
- [PauliPropagation.jl gate definitions](https://github.com/SparqleSim/PauliPropagation.jl/tree/main/src/Gates)：`PauliRotation`、`ImaginaryPauliRotation`、noise channels、picture conversion 和 `TransferMapGate`。
- [PauliPropagation.jl propagation specializations](https://github.com/SparqleSim/PauliPropagation.jl/blob/main/src/Propagation/specializations.jl)：Pauli rotation、imaginary rotation、Pauli noise 和 amplitude damping 的局部 apply 规则。
- [PauliPropagation.jl TransferMap](https://github.com/SparqleSim/PauliPropagation.jl/blob/main/src/PauliTransferMatrix/transfermaps.jl)：PTM columns 到局部稀疏 transfer map 的转换。
- [PauliPropagation.jl custom gradient](https://github.com/SparqleSim/PauliPropagation.jl/blob/main/src/gradient.jl)：`rewindgradient` 的参数化门限制和 noise 不支持说明。
- [Pauli Propagation: A Computational Framework for Simulating Quantum Systems](https://arxiv.org/abs/2501.13101)：Pauli propagation 方法和应用背景。
- [TenCirPauli existing Julia comparison](../../examples/research/pauli_propagation_julia_vqe/README.md)：v0.8.0 的 matched VQE value/gradient/benchmark 记录。
- [TenCirPauli PTM/JAX feature seed](feature-incubator.md#jax-compatible-gradients-for-parameterized-ptms)：本调研将其从孤立 seed 具体化为 runtime ABI、native VJP 和 JAX composition 方案。
