# Proposal: per-target exact-KV trust and proof-driven recovery

Status: local design/prototype, not submitted, not a production routing feature.

Related: [#294 M5](https://github.com/vllm-project/router/issues/294),
[#295 lifecycle/fallback](https://github.com/vllm-project/router/issues/295),
[#331 cost model](https://github.com/vllm-project/router/issues/331),
[#371 selection](https://github.com/vllm-project/router/pull/371).

Baseline: `442b8bb74fba09052428cf3af0513916a4f92b43`.
No server changes, new dependencies, CLI switches or request-path activation.

## Problem and bounded contribution

The placement index describes residency, not whether its history is complete.
Missing a remove/clear event can leave positive matches for absent blocks. New
traffic establishes liveness, not completeness. Recovery must therefore be a
separate, explicit action backed by validated and applied evidence.

Existing `IngestionSignal` has source-level Advance, Gap, ReplayApplied and
IncarnationReset. Discovery attributes a source to an instance while the index
and routing target also distinguish DP rank. A direct source-only adapter would
lose information needed for rank isolation. A replay high-water sequence does
not prove continuous coverage, successful decoding or index application.

This prototype supplies a pure, deterministic state machine for contract review.
It does not replace #295's decoding, subscriber lifecycle or PD safety work,
nor #331/#371's load/scoring logic. It is intentionally not called by these paths.

## Objects and ownership

- `TrustKey`: complete cache compatibility namespace + instance ID + DP rank.
  The namespace must include model/weights/tokenizer compatibility, block size,
  hash configuration and any other index separation required by discovery. The
  prototype treats it as an opaque string, not a validated canonicalizer.
- `SubscriptionLease`: opaque local generation and authoritative incarnation.
  Each reconnect gets a new generation even if the worker did not restart.
  An allocation-identity fence also prevents equal numeric generations in two
  separate arbiter objects from accepting each other's leases or tickets.
- `RecoveryTicket`: current lease, local revision and monotonic issue time. Any
  subsequently applied event, invalidation or expiry fences the ticket.
- `RecoveryProof`: an assertion by a trusted, serialized recovery producer that
  a snapshot, clear barrier or contiguous replay has already been applied. This
  enum is NOT cryptographic proof or a validator of transport/index behavior.

The supervisor owns the arbiter and serializes lifecycle, applied events, proof
application and index access. Leases/tickets are process-local, not wire payloads.
Only the authoritative lifecycle owner may subscribe or retire; it must fence
old index mutations and clear worker-domain residency on restart, without
clearing independently owned shared-cache residency. This module does neither.

## State and transitions

| Event / condition | Per-target result | Exact KV credit contract |
| --- | --- | --- |
| Unknown/retired or subscribed without any applied state | FALLBACK | zero |
| First live event without bootstrap | DEGRADED | zero |
| Verified/applied snapshot or clear barrier, fresh ticket | EXACT | eligible |
| Contiguous applied event while fresh and exact | EXACT | eligible |
| Sequence gap, explicit invalidation or freshness expiry | DEGRADED | zero |
| New events after a gap/expiry | stays DEGRADED | zero |
| Continuous applied replay covering checkpoint+1 through high-water | EXACT | eligible |
| Duplicate/out-of-order event | no freshness refresh | unchanged or expired |
| Reconnect/restart | new generation, bootstrap required | zero |
| Any API error | caller must fail closed | zero |

These are **per-target** states. A model-level decision about all-source loss,
partial source availability and selecting cache_aware/P2C is not implemented.
Model, health, role and execution safety remain mandatory eligibility filters
even when a caller falls back. Load observations can remain usable independently
of exact overlap. A previously returned EXACT must not be cached as permission:
call `mode` at consumption under the same supervisor ordering contract.

## Recovery ordering and coverage

1. Establish the authoritative subscription lease.
2. Stop/serialize applied event mutations for the affected target.
3. Capture a ticket. If already expired, this latches degradation first.
4. Validate AND apply the snapshot/clear/replay to the same namespace and target.
5. Submit the proof using the ticket, before its TTL. If anything changed, reject
   it and re-establish the ordering barrier; do not keep routing on stale credit.
6. Resume event consumption and query the current mode at scoring time.

If exactness was established through sequence 100 and live events jump to 103,
the retained recovery lower bound is 101. A later gap at 108 does not move that
bound to 104. Replay must cover from <=101 through >=108; a high-water of 108
without verified intermediate application is not acceptable. Startup has no
trusted checkpoint, so replay alone cannot bootstrap this conservative version.

Snapshot and clear proofs must not roll back the applied high-water. Resetting
sequence numbers requires an explicit new subscription/incarnation. Sequences
are unsigned here; an upstream signed sequence must be validated before adapting.

Times are durations from a single monotonic process origin, not UTC. TTL expiry
uses `age >= max_age`. Before accepting a new event we check previous freshness:
an outage cannot self-heal simply because nobody called `mode` during the outage.
Duplicate/out-of-order traffic never refreshes the freshness clock. Quiet but
healthy streams conservatively degrade until a separate proof/heartbeat contract
is designed; this is a correctness bias, not a claimed optimal failure detector.

## API seams still requiring maintainer agreement

| Producer / consumer | Required contract | Explicitly deferred |
| --- | --- | --- |
| #295 lifecycle / discovery | namespace, instance, rank, authoritative incarnation, subscriber lease, retirement | same-source multi-rank native DP execution |
| strict decoder / index applier | publish Advance only after successful strict decode + complete application; reliable invalidation | ZMQ replay protocol and signal-channel overflow handling |
| recovery producer | validated contiguous coverage or ordered snapshot/clear; applied receipt | authenticating remote claims and reconciling shared tiers |
| #371 scorer | exact-credit gate and reason counters; safe fallback | scorer wiring, capacity filter and PD orchestration |
| model supervisor | aggregate target states and fallback selection | cross-Router HA and shared ownership |

Current source-only `IngestionSignal` cannot satisfy this contract. Do not expose
`recover(ReplayApplied { replay_seq })` as an adapter: it would forge the missing
coverage guarantee. The production design must also fence dropped control signals
and synchronize eligibility reads with index changes. This prototype does not
make asynchronous index reads race-free merely by returning EXACT.

## Tests and reproduction

The deterministic synthetic tests cover startup, contiguous/gapped sequences,
partial/full replay, staleness without polling, duplicate events, namespace /
instance / rank isolation, reconnect, restart, retirement, old proofs, monotonic
time, invalid intervals, integer boundaries and a synthetic 4-by-16 key layout.
The 64-key case is NOT a 64-rank engine integration test.

Local validation on 2026-10-13: 17 new trust tests passed; the targeted
`kv_index::` run passed 78 tests including those 17; the existing frontend L0
tests passed 7 and program-identity/hint parsing tests passed 9. `cargo check
--locked --offline --lib`, the standalone warning-free module test build,
targeted rustfmt and diff checks passed. The crate emitted existing atomic API
deprecation warnings in worker/retry code; these unrelated sites were not changed.
The cross-arbiter fence regression was observed failing before its fix and
passing afterward. This is a prototype defect caught locally, not a claimed
fix for an upstream issue. No full-suite, remote-CI or performance claim is made.

```sh
cargo test --locked --offline --lib kv_index::trust::tests
cargo check --locked --offline --lib
rustfmt --edition 2021 --check src/kv_index/trust.rs src/kv_index/trust_tests.rs
git diff --check
```

For an environment without cached project dependencies, the exact same pure
module can be tested without fetching/building the full dependency graph:

```sh
rustc --edition 2021 --test -D warnings src/kv_index/trust.rs -o /tmp/kv-trust-tests
/tmp/kv-trust-tests
```

No end-to-end routing, replay producer, real inference engine, throughput,
cross-Router HA, native-DP or physical KV action is validated by these tests.
Before publishing, obtain maintainer agreement on the seam and rebase on the
then-current upstream. The prototype may be revised substantially or retained
only as a contract-test fixture; it is not a claim of assigned M5 ownership.

## Suggested public sub-RFC scope (draft, not posted)

Title: `[RFC][#294 M5] Rank-scoped trust state and proof-driven recovery contract`

Request feedback on: rank-qualified ingestion signals; lifecycle generation
authority; replay coverage/applied receipts; per-target vs per-model fallback;
atomicity with index reads; stale/quiet-stream semantics and low-cardinality
observability. Coordinate lifecycle ownership with #295 and scoring with #371.
Do not claim a new standalone subscriber implementation or physical KV control.
