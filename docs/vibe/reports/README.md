# Reviews, Audits, and Reports

本目录保存阶段 Review、Acceptance、Audit、Closure、Remediation、性能证据和其他历史工作记录。它们用于追溯决策和重建上下文；若内容与骨干文档或当前源代码冲突，以骨干文档和当前源代码为准。

## Audits and release reviews

- [Current-tree audit reconciliation, 2026-08-11](audit-report-2026-08-11-current.md): current `e38533f` disposition of the findings from the deprecated August 8 Round 5 audit.
- [Technical debt remediation plan, 2026-08-21](technical-debt-audit-2026-08-21.md): current `264ff67` execution plan for Structured dependency cleanup, native-plan ABI contracts, and architecture guardrails.
- [Deep audits](audit-report-2026-08-04.md)
- [Deep audit round 2](audit-report-2026-08-04-round2.md)
- [Deep audit round 3](audit-report-2026-08-05-round3.md)
- [Deep audit round 4](audit-report-2026-08-06-round4.md)
- [Deep audit round 5, 2026-08-08 — deprecated](audit-report-2026-08-08-round5.md): historical `ceab218`-dirty baseline; superseded by the current-tree reconciliation above.
- [Deep-scan checklist](audit-scan-checklist.md)
- [Pre-release review](pre-release-review-2026-08-03.md)

## Phase acceptance and implementation reviews

- [Phase Alpha review](phase-alpha-review-2026-08-03.md)
- [Phase 1 review notes](phase-1-review-notes.md)
- [Phase 1 acceptance review](phase-1-acceptance-review-2026-08-01.md)
- [Phase 2 acceptance review](phase-2-acceptance-review-2026-08-02.md)
- [Phase 3 review](phase-3-review-2026-08-02.md)
- [Phase 4 review](phase-4-review-2026-08-02.md)
- [Phase 5 review](phase-5-review-2026-08-02.md)
- [Phase 5 and 5.5 review](phase-5-and-5.5-review-2026-08-02.md)
- [Phase 6 review](phase-6-review-2026-08-02.md)
- [Phase 7 review](phase-7-review-2026-08-03.md)
- [Phase 7 second-round review](phase-7-second-round-review-2026-08-03.md)
- [Phase 7 third-round review](phase-7-third-round-review-2026-08-03.md)
- [Phase 7.5 review](phase-7.5-review-2026-08-03.md)
- [Phase 7.5 second-round review](phase-7.5-second-round-review-2026-08-04.md)
- [Phase 8 review](phase-8-review-2026-08-04.md)
- [Phase 8.5 review](phase-8.5-review-2026-08-04.md)
- [Phase 8.5 second-round review](phase-8.5-second-round-review-2026-08-04.md)
- [Phase 8.5 audit round 2 completion review](phase-8.5-audit-round2-completion-review-2026-08-05.md)
- [Phase 9 review](phase-9-review-2026-08-06.md)
- [Phase 9 second-round review](phase-9-second-round-review-2026-08-06.md)
- [Phase 9 remediation closure](phase-9-remediation-closure-2026-08-06.md)
- [Phase 9.5 review](phase-9.5-review-2026-08-06.md)
- [Phase 9.5 closure evidence](phase-9.5-closure-evidence-2026-08-06.md)
- [Phase 10 review](phase-10-review-2026-08-06.md)
- [Phase 12 common noise channel gradients review](phase-12-review-2026-08-13.md)
- [Phase 13 classical shadow performance review](phase-13-performance-review-2026-08-20.md): separates definite repeated-work and benchmark-coverage findings from performance hypotheses that require release-mode A/B evidence; a small isolated speedup is not sufficient to justify added structural complexity.
- [Phase 13 classical shadow acceptance review](phase-13-acceptance-review-2026-08-20.md): concludes that the public/core feature slice is functionally implemented but acceptance remediation remains open for shadow Rényi-2 memory scaling, dominant large-width frame generation, allocation/worker contracts, retained correctness evidence, and scalable GIL-held preparation.
- [Phase 14 stabilizer-code analysis acceptance review](phase-14-acceptance-review-2026-08-21.md): concludes that the public/core feature slice is functionally implemented but acceptance remediation remains open for lexicographic error-set witnesses, dominant construction scaling, direct packed/fused batch execution, logical-path resource guards and throughput, retained correctness evidence, and ABI/state cleanup.
- [Phase 14 acceptance remediation record](phase-14-acceptance-remediation-2026-08-21.md): records the first-round correctness, packed-path, resource, construction, logical-kernel, ABI, documentation remediation, complete local quality gate, and matched release comparison; the later owner decision selected eager logical construction and waived the eager-versus-lazy A/B.
- [Phase 14 stabilizer-code analysis second-round review](phase-14-second-round-review-2026-08-22.md): accepts eager logical construction by owner decision, confirms the main first-round implementation repairs, and narrows the remaining work to a non-overestimating advisory resource guard, independent correctness evidence, benchmark provenance/workload coverage, and small ABI/helper cleanup.
- [Phase 14 second-round remediation](phase-14-second-round-remediation-2026-08-22.md): closes SR1–SR4, documents the flat ABI and independent correctness evidence, and replaces the over-expanded QEC benchmark matrix with a measured sub-ten-minute compact acceptance profile plus a shorter daily profile.

## Phase 10.5 records

- [Phase 10.5 acceptance evidence](phase-10.5-acceptance-evidence.md)
- [Phase 10.5 first-round review](phase-10.5-review-2026-08-07.md)
- [Phase 10.5 second-round review](phase-10.5-second-round-review-2026-08-08.md)
- [Phase 10.5 third-round review](phase-10.5-third-round-review-2026-08-08.md)
- [Phase 10.5 fourth-round review](phase-10.5-fourth-round-review-2026-08-09.md)
- [Phase 10.5 fifth-round review](phase-10.5-fifth-round-review-2026-08-09.md)
- [Phase 10.5 fifth-round remediation](phase-10.5-fifth-round-remediation-2026-08-09.md)
- [Phase 10.5 sixth-round review](phase-10.5-sixth-round-review-2026-08-09.md)
- [Phase 10.5 entropy-reduction review](phase-10.5-entropy-reduction-review-2026-08-09.md)
- [Phase 10.5 owner decision log](phase-10.5-owner-decision-log.md)
- [Phase 10.5 trusted-handle audit](phase-10.5-trusted-handle-audit.md)

## Capability studies

- [Stabilizer, QLDPC, and fault-tolerant analysis capability study](qec-stabilizer-capability-study.md): exploratory scope, input/output contracts, detection-versus-correction distance, physical logical-operator representation, engineering difficulty, and ecosystem boundaries for a packed-GF(2) QEC direction; not an implementation commitment.

## Implementation and performance records

- [Default lazy operator results](operator-lazy-results.md)
- [Native-backed lazy operator results](pauli-lazy-results.md)
- [Native-backed lazy operator review](native-lazy-operator-review-2026-08-05.md)
- [Native-backed lazy operator remediation review](native-lazy-operator-remediation-review-2026-08-05.md)
