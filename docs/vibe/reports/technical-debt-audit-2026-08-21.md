# TenCirPauli 技术债修缮方案

审查时间：2026-08-21。审查基线：HEAD `264ff67`（`v0.5.0`）。本文是一次性执行方案；完成状态以当前源码、测试、benchmark 和 trunk 文档为准。

## 结论

当前需要实施的维护性工作合并为三项：清理 Structured Python 的隐藏依赖并修正 compile 所有权；收紧 native plan 的 metadata 与 Python ABI 合同；增加防止同类问题回潮的架构检查并修正 trunk 状态。三项之外不再从原审查清单派生零散任务。

实施必须同时满足以下约束：主要 Python 用户接口、导出路径、公开签名、结果类型、异常语义和数值结果不变；现有 Rust core plan、storage、专用 kernel 和选择逻辑不变；受影响的构造、首次执行和 steady execution 不得出现超出本机重复测量噪声带的性能回退。

## 实施项一：清理 Structured Python 的隐藏依赖并修正 compile 所有权

### 要解决的问题

`python/tencirpauli/_structured/base.py` 在文件末尾导入所有 concrete implementation modules，再把自己的 globals 通过 `module.__dict__.setdefault()` 注入 `space`、`mapping`、`fermion`、`boson`、`hybrid` 和 `compile`。这些模块因此无法仅从 import statements 判断真实依赖，并使用整文件 Ruff suppression。与此同时，`_StructuredOperator.compile()` 提供宽签名，Fermion、Boson 和 Qudit concrete classes 为保持各自正确的用户接口而使用 `type: ignore[override]`。这两类问题属于同一个所有权错误，应在一项改造中解决。

### 实施步骤

1. 固定当前公开合同。保留并扩充 `tests/test_public_api.py::test_structured_public_facade_preserves_identity_module_repr_and_pickle`，增加 clean subprocess 中不同 import 顺序、`__all__`、各 concrete `compile()` signature、异常类别、native handle round-trip，以及 Fermion、Boson、Hybrid 和 Qudit 各一个已知数值结果。
2. 列出每个 sibling module 当前从 bridge 获得的运行时名字，区分真正共享的 model/native helpers、仅用于 typing 的名字和 family-to-family construction dependencies。根据该依赖图确定迁移顺序。
3. 将无 concrete-family 反向依赖的共享 model、validation 和 native-handle helpers 移到窄的私有模块。需要创建具体 family 返回对象的逻辑由一个明确的 composition layer 或函数内显式 import 负责，不再使用 globals copying、module namespace mutation 或底部全家桶导入修复循环依赖。
4. 逐个 sibling module 补齐显式 imports，并删除其整文件 `# ruff: noqa: F401, I001`。迁移后的模块必须在 clean process 中独立 import 和运行，不得依赖 bridge 已经执行。
5. 将 `_StructuredOperator.compile()` 的实际实现移到 private composition helper。Hybrid、Fermion、Boson 和 Qudit concrete classes 保留当前公开方法签名并显式调用该 helper；base class 不再声明一个 concrete classes 无法合法 override 的宽 `compile()` 合同。删除三处 `type: ignore[override]`，不新增 public hierarchy 或 universal operator abstraction。
6. 所有 sibling modules 完成显式依赖后，删除 `base.py` 的 namespace injection loop、仅为该 loop 服务的底部 imports 和相关 suppression。同步删除 `structured.py` 中无引用的 `_PRIVATE_COMPATIBILITY_EXPORTS` tuple；保留当前确有兼容作用的 public facade、对象 `__module__` 设置和 `__getattr__` forwarder，不扩展其职责。

### 完成标准

- `_structured` 中不存在 globals copying、module `__dict__` mutation、为循环依赖服务的全家桶 imports、整文件 F401/I001 suppression 或 `type: ignore[override]`。
- `tencirpauli.structured` 的 public imports、identity、`__module__`、repr、pickle、signatures、异常和数值结果与改造前一致。
- Black、Ruff、strict mypy、完整 pytest 和 `scripts/check.py` 通过。
- 对 Fermion、Boson、Hybrid 和 Qudit 的代表性 compile/construction、首次 apply 和 steady apply 做 release-mode 前后对比；结果必须处于重复测量噪声带内，否则该切片不合入。

## 实施项二：收紧 native plan metadata 与 Python ABI 合同

### 要解决的问题

`NativePlanMetadata` 的取值和 fallback 分散在 `hamiltonian.py`、`charge.py`、`symmetry.py` 和 `_structured`，同一字段可能由 native property、Python 计算值或 family 常量产生。`_native.pyi` 对大量固定 NumPy 返回仍标为 `object`/`Any`，使 tuple arity、handle 类型和 scalar dtype 无法静态检查。现有 plan tests 已覆盖部分 flat apply 和 `apply_into` 行为，但没有形成覆盖所有适用 family/storage 的统一合同矩阵。

### 实施步骤

1. 为 Pauli/Structured、generic Charge lazy/eager、factorized Charge lazy/eager 和 packed U1 建立 metadata snapshot tests，逐字段固定来源、fallback 和公开属性值。
2. 在 `_native_mvp.py` 建立一个 shared typed metadata builder，并为 family-specific semantics 使用显式 adapters。Builder 只负责共同字段的类型规范化和已经确认一致的 fallback；`basis_ordering`、`execution_family`、`retained_format`、transition semantics 和 construction peak 等 family-specific 值必须由 adapter 明确提供。
3. `_with_plan_metadata()` 改为通过 private copy constructor 和 `dataclasses.replace()` 或等价方式复用同一个 native execution handle，只修改 mapping/source metadata，不再重新展开全部 plan fields。所有现有 public constructors 和 properties 保持不变。
4. 在 `_native.pyi` 中定义 `Complex128Array`、`Float64Array`、`UInt64Array`、`UIntpArray`、`UInt32Array` 和 `UInt8Array` 等 `numpy.typing.NDArray` aliases。先完成 plan apply 与 dense/COO/CSR ABI，再完成 Pauli、Fermion、Boson、Majorana 和 Hybrid 的固定 materialization tuple。Rust 中的 `usize` 返回必须标为 platform `uintp`，不能统一写成 `uint64`。
5. 增加直接调用 private `_native` endpoints 的 ABI tests，断言 tuple arity、ndim、dtype、C-contiguous、ownership/writeability 约定和已知数值内容。Shape、contiguity 和 ownership 不依赖 mypy 推断，必须由运行时测试固定。
6. 将现有 plan tests 参数化为 capability matrix，覆盖适用 family/storage 的 metadata、已知数值结果、输入 dtype/shape/contiguity、`apply_into` overwrite/read-only/non-contiguous/overlap、并发复用和异常类别。Optional capabilities 由矩阵明确标记，不使用散落的 `hasattr()` 猜测合同。

### 完成标准

- Metadata 的共同 fallback 只有一个 owner，family-specific semantics 仍显式可读，不存在新的万能 `getattr(..., default)` factory。
- 所有已知固定 numeric returns、handles 和 tuple arities 在 `_native.pyi` 中有准确类型；保留的 `object`/`Any` 均对应真实动态 backend inputs，并有明确登记。
- Direct native ABI tests、strict mypy、完整 pytest 和 `scripts/check.py` 通过。
- Public plan properties、constructors、返回 arrays、异常和数值结果不变；受影响 plan construction、首次 apply、steady `apply`/`apply_into` 的 release A/B 不回退。

## 实施项三：增加架构门禁并修正 trunk 状态

### 要解决的问题

当前维护规则主要依赖人工审查，无法阻止 namespace mutation、宽 ABI annotation 和整文件 suppression 再次进入代码。`implementation-status.md` 同时把 Structured 维护性收敛描述为完成，而源码仍保留 namespace bridge；`maintainability.md` 还把删除 `restrict_u1()` 写成既定路线，与当前“主要 Python 接口不变”的约束冲突。

### 实施步骤

1. 增加 `scripts/check_architecture.py`，使用 AST 检查 production Python 中的 wildcard import、`globals()`/module `__dict__` 写入、未登记的 module-level dynamic `__getattr__` 和 `_native.pyi` 中未登记的 `object`/`Any` 参数或返回，并用窄文本解析检查整文件 Ruff suppression。
2. Allowlist 使用稳定的 `rule + path + qualified symbol/annotation position` 定位，并记录 exception id、owner、reason 和 review/removal condition；不得使用整文件豁免、行数配额或易漂移的纯行号定位。
3. 为 checker 增加命中、合法例外、stale exception 和 syntax error tests，并接入 `scripts/check.py`。Checker 必须扫描完整工作树、秒级完成、不访问网络且不导入 native extension。
4. 更新 `docs/vibe/implementation-status.md`，明确“Structured 职责模块已经拆分，但隐藏 namespace bridge 在实施项一完成前仍未关闭”。更新 `docs/vibe/maintainability.md`，明确 `restrict_u1()` 继续作为 deprecated thin forwarding API 保留；除非未来 owner 单独批准 breaking release，本计划不安排删除。
5. 实施项一和二每删除一个既有例外，就在同一 change 中删除对应 allowlist entry。完成后检查 trunk 文档、当前源码和架构 checker 的例外账本一致。

### 完成标准

- `scripts/check.py` 会拒绝新的隐藏 namespace wiring、整文件 import suppression 和未登记宽 ABI annotation。
- Checker 自身测试通过且不会给常规检查带来可感知的运行时间负担。
- `implementation-status.md`、`maintainability.md`、当前源码和本方案对 Structured 状态及 `restrict_u1()` policy 的描述一致。

## 明确不做

- 不删除、不改签名、不改 warning 或数值语义的 `PauliOperator.restrict_u1()`；现有 warning/equivalence test 保留。
- 不抽象现有 Rust/PyO3 native plan wrappers。它们是薄静态包装，当前没有足够维护收益支撑 trait、macro 或共享 executor 改造；实施项二只固定合同、metadata 和 stub typing。
- 不在本方案中修改 stabilizer sampling、symplectic frame 或 scratch reuse。这些工作由仍在进行的 Phase 13 performance/acceptance remediation 统一 profile、A/B 和决策，不能在技术债清理中另建路线。
- 不合并 Pauli、Fermion、Boson、Hybrid、Weyl、Charge 或 U1 的专用 algebra/kernel/storage，也不引入 universal operator、trait-object executor 或 hot-loop mega-enum。
- 不为了代码行数统一不同 family 的 optional capabilities、budget semantics、basis ordering 或 transition representation。

## 执行顺序

1. 完成 Structured 公开合同基线和实施项一。
2. 完成 metadata/ABI 基线和实施项二。
3. 完成实施项三，把前两项已经消除的问题固化为门禁并同步 trunk 状态。

每项作为独立可回退的 change series 落地，不并行保留新旧实现。任何切片只要改变主要 Python 接口、数值语义、异常合同、native kernel/storage 选择，或造成超出测量噪声的性能回退，就不合入。

## 本次复核证据

复核确认项目环境使用工作区 `python/tencirpauli` 和 `_native.abi3.so`。Focused pytest 覆盖 Structured facade identity/repr/pickle、四类 MVP flat apply、`apply_into` overlap/concurrency 和 `restrict_u1()` warning/equivalence，结果为 4 passed。本次只重写修缮方案，没有修改 Rust、PyO3、Python production、测试或 benchmark 源文件，也没有运行完整质量门禁和 release benchmark；这些验证属于每个实施项的完成条件。
