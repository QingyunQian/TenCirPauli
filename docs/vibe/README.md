# Vibe Design Index

TenCirPauli 的实验性设计、实现状态、规范和审查记录集中在这里。当前架构和公共合同以根目录下的骨干文档与源代码为准；阶段性规格、Spike、Review 和 Audit 默认是历史工作记录。

## 骨干文档

- [Architecture and roadmap](architecture.md)：项目范围、模块边界、算法路线、风险和 go/no-go 门槛。
- [Core semantics](semantics.md)：Pauli 表示、phase、qubit ordering、系数、规范化和矩阵约定。
- [Implementation status](implementation-status.md)：持续工作的状态、验证证据、决策、阻塞项和下一步。
- [Maintainability direction](maintainability.md)：在保持正确性、性能、内存和粗粒度 ABI 的前提下降低实现熵。
- [Local benchmarking](benchmarking.md)：benchmark 协议、本地结果历史和回归比较规则。
- [Release process](releasing.md)：CI、构建产物、GitHub Release 和 PyPI 发布流程。

## 目录

- [Specifications and Spikes](specs/README.md)：各阶段的实现规范、研究提案和非绑定实验方案。
- [Reviews, Audits, and Reports](reports/README.md)：Review、Audit、Acceptance、Closure、Remediation 和结果记录。

## 支持性参考

- [Rust quantum operator competitor research](competitor-research.md)：相关 Rust 算符生态和 TenCirPauli 定位调研。
- [Pauli propagation comparative research](pauli-propagation-comparative-research.md)：PauliPropagation.jl 的噪声、PTM、虚时演化和梯度方案，以及 TenCirPauli 的补全路线。
- [Feature incubator](feature-incubator.md)：延期或尚未成熟的功能想法、设计片段和进入正式规范的条件。
- [P0 reference vectors](reference-vectors.md)：独立 NumPy dense oracle、固定 regression vectors、seed、dtype 和容差。

新增设计文档时，请使用描述性的 lowercase 文件名，并将正式规范放入 `specs/`，将审查和历史证据放入 `reports/`；同时更新相应目录下的 README。若提案尚未成为实现合同，请在文档标题附近明确其状态。
