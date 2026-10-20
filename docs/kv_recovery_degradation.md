# KV routing recovery and degradation

This document is a sub-design for [Router #294](https://github.com/vllm-project/router/issues/294).
It extends the trusted KV library component in this branch. It is not an
accepted RFC, a claim that #294 is complete, or a substitute for real
multi-engine failure testing.

## Problem

An event-driven KV index is safe only while every routing decision knows
whether its evidence is complete and fresh. A subscriber can disconnect,
restart, skip a sequence, receive an obsolete incarnation, or fall behind a
shared storage tier. Treating one source failure as a whole-Router failure
throws away healthy exact evidence. Treating a recovered live stream as a new
baseline can instead credit blocks that were evicted during the gap.

The implementation therefore separates three questions:

1. Is a concrete resident copy still eligible for cache credit?
2. Which routable target should receive the request?
3. Which authoritative reconciliation proves that an untrusted source may
   return to exact mode?

## Identity and scope

The following identities remain distinct:

| Identity | Key | Purpose |
| --- | --- | --- |
| Cache identity | model, tokenizer/renderer hash, block size, cache salt/group | Prevents cross-model or cross-template credit |
| Routable target | `(instance_id, dp_rank)` | Selects an addressable worker/DP rank |
| Residency owner | source, incarnation, backend/object identity | Owns a Device, Host, Disk or External copy |
| Event source | target plus cache identity, incarnation and tier | Orders and fences lifecycle events |

A 4-instance, 16-DP deployment has 64 routable targets, not four opaque
instances. Each target has an independent source state and freshness
watermark. Shared Host/Disk copies may have additional owner identities; a
Device clear for one target must not erase a shared lower-tier copy.

The current library/runtime code remains single-group, Device-resident and
one-source-per-routable-target. The tiered owner model below is a required
contract beyond the completed prototype. The index provides an explicit
tier-scoped clear primitive so a Device barrier cannot erase a surviving Host
or shared-pool claim. The subscriber also accepts a backward-compatible
`AllBlocksCleared` extension with either `medium` or `ownership`: `medium`
clears one Worker tier and `ownership` clears one shared CacheOwner. An event
without either field retains upstream's current whole-worker semantics.
Conflicting, empty-owner and unknown-medium scopes are rejected. Upstream vLLM
does not emit these optional fields yet, so a real producer must adopt the
extension before tier-scoped clear can be claimed end to end. The trusted
Device runtime accepts only the whole-worker or Device-tier form and rejects
Host/shared-pool clears for its Device trust decision.

Instance-to-DP visibility must be explicit. A target may use only resident
copies reachable under the configured locality/access contract. A local
instance hit and a remote-instance hit are different tiers/paths even if both
eventually transfer into the same device.

## Source state machine

`KvTrustRegistry` exposes four bounded states:

```text
subscribe/reattach
       |
       v
  Recovering
       |
       | validated snapshot, clear barrier or complete replay
       v
    Healthy -------- stale/unavailable --------> Degraded
       |                                             |
       | gap/rollback/incarnation mismatch           |
       v                                             |
  Unreconciled <-------- live delta -----------------+
       |
       | validated snapshot/clear barrier/complete replay at or after watermark
       v
    Healthy
```

- `Recovering`: a source was attached, but no complete starting view exists.
- `Healthy`: a validated snapshot/clear barrier was followed by contiguous,
  fresh events.
- `Degraded`: the source is stale or unavailable, without evidence that the
  ordering identity changed.
- `Unreconciled`: a gap, rollback, obsolete incarnation or unproven recovery
  invalidated topology. Live deltas do not restore trust.

Only an ingestion owner may call `reconcile`. Request headers, Agent hints,
route receipts and client-provided sequence numbers never establish trust.
Reconciliation behind the highest observed sequence is rejected. Reattaching
the same incarnation preserves that watermark; only a new incarnation starts a
new sequence epoch. A source removal is represented by absence, not by a
fabricated healthy state.

### Bounded replay recovery

Discovery exposes an optional ZMQ ROUTER `replay_endpoint` per publisher. The
trusted runtime now uses it for two cases:

- late attachment: a first live event at sequence `N` requests replay from
  sequence 0 through `N-1`;
- live sequence gap: an event at sequence `N` after local watermark `L`
  requests replay from `L+1` through `N-1`.

Before the replay request, the target becomes `Unreconciled`; selection may
therefore fall back while network replay is running. The runtime does not hold
the state mutex while polling the DEALER socket. Each replayed batch is decoded,
rank-checked and applied under a short lock, but replay events are not allowed
to reconcile trust. Only contiguous replay through the entire requested range
restores `Healthy`; the triggering live event is applied afterward. A missing,
duplicated, out-of-order, wrong-topic, wrong-rank, corrupt or timed-out replay
clears partial residency, advances the incarnation and resets the checkpoint.
The source stays fenced until another complete replay or an authoritative clear
barrier succeeds.

Old but internally contiguous replay history is still useful for rebuilding
the index, but its publisher timestamp is retained. A replay that ends older
than the freshness window therefore reconciles to a stale, degraded source
rather than manufacturing fresh exact credit.

Replay is a recovery aid, not a portable snapshot. If the publisher no longer
retains the requested range, another Router replica needs a baseline beyond its
local replay horizon, or an engine restart changes the incarnation, the source
remains fenced unless the trusted ingestion owner supplies a portable
reconciliation snapshot.

### Portable reconciliation snapshots

`KvRuntime::import_portable_snapshot` accepts a complete format-v2 snapshot from
the trusted ingestion owner. It is a library boundary: Router request headers,
Agent hints, route receipts and arbitrary clients cannot call it or use it to
establish trust. The caller also supplies the expected cache identity, binding
the import to model, tokenizer/renderer ID, engine hash mode and block size. The
caller also supplies the expected `(owner_id, epoch)` fence from its trusted
control-plane integration; a snapshot's own fence declaration is never
self-authorizing.

The importer performs a two-phase replacement. It first validates the format,
identity, SHA-256 integrity value, source and block limits, timestamps,
incarnation/sequence watermarks, owner scope, per-residency block claims and
the complete prefix-parent graph. It then stages a new index, trust registry
and cursors. Only after every source succeeds is the cache identity's old pool
swapped. Any validation or staging failure leaves the previous residency,
trust and global block accounting untouched.

Snapshot import has these conservative rules:

- the snapshot must be marked complete and its source set must exactly equal
  the currently discovered `(instance_id, dp_rank)` set;
- the first successfully imported snapshot pins its full cache identity,
  including tokenizer/renderer ID, to the discovered pool; a later snapshot for
  the same model/hash/block core with a different tokenizer is rejected before
  it can replace routing state;
- an equal snapshot ID and checksum is an idempotent replay, while reusing an ID
  with different content is rejected across later snapshots for the same cache
  identity;
- an old incarnation or a lower sequence, creation time or source observation
  time cannot roll back state in the same incarnation;
- a different topology at the same `(incarnation, sequence)` is rejected even
  when a different snapshot ID has newer metadata; prefix and residency order
  are canonicalized before the source-content fingerprint is compared;
- a worker-owned residency must carry the source snapshot's incarnation; a
  mismatched ownership epoch is rejected before staging;
- the snapshot fence must equal the separately supplied trusted expectation and
  is covered by the snapshot checksum;
- a fencing epoch may only advance; a lower epoch is rejected, and one epoch
  cannot change owner. Applying the same owner and epoch is replayable, while a
  successful future epoch replaces the locally enforced fence;
- a snapshot discovered after its publisher replay horizon can establish the
  topology baseline, but its observation age is preserved; a snapshot older
  than the freshness window is degraded and receives no exact credit;
- shared-owner claims remain separate from worker-owned Device claims, including
  their tier and routing locality; and
- the replacement snapshot is charged against the global configured block
  budget before the pool swap.

The checksum binds the fence to the snapshot payload but detects only accidental
corruption; it does not authenticate the producer. The runtime compares the
declared fence with an independently supplied control-plane expectation. This
is not leader election, split-brain prevention, signature verification or an
authenticated transport. Authorization, durable storage and production
active-active ownership remain outside this CPU-side primitive. A runtime
built with `PortableSnapshotFenceAuthority` delegates the final atomic
epoch-validation and advancement step to an external adapter. The authority is
called only after snapshot validation, source-set checks, budget checks and
state staging succeed. This gives a durable control-plane implementation a
stable integration point that survives Router process restarts; the tests use
a shared reference authority, not a production database or leader-election
service.

The authority call does not hold the live KV state lock. Validation and staging
record the target pool's stable ID and mutation revision, plus the global fence
revision, then release the lock while the authority persists the fence. If
ingestion, discovery or removal changes that pool, the pool is deleted and
recreated, or another import advances the fence before the staged pool can be
installed, the import returns `ConcurrentSnapshotMutation` instead of
overwriting newer state. Traffic in unrelated cache pools does not force this
retry. The caller retries the same snapshot and fence; an authority must treat
exact-fence replay as idempotent. This bounded retry contract prevents a slow
database call from serializing event ingestion and routing selection, but it is
not a lease, leader election or proof against split brain.

Freshness is evaluated against the publisher observation age. A duplicate
event does not refresh it. Reading status is side-effect free, so metrics or
debug endpoints cannot keep stale evidence alive.

## Selection during degradation

The selector emits three bounded modes:

| Mode | Meaning | KV credit | Load behavior |
| --- | --- | --- | --- |
| `exact` | Every healthy candidate is fresh and reconciled | All eligible targets | Unified cache/load cost |
| `degraded` | Only part of the healthy pool is trusted | Trusted targets only | Untrusted targets compete by load with zero cache credit |
| `fallback` | No trusted evidence, empty query or no healthy target | None; provider is not queried | Deterministic normalized-load choice or no target |

Matches returned for an untrusted `(instance_id, dp_rank)` are filtered before
scoring. This prevents a stale provider entry from becoming exact credit after
its source has been fenced. One failed DP rank therefore does not force all 63
other targets into blind load balancing. Conversely, a degraded decision never
claims that untrusted targets were checked.

If no healthy target exists, the decision contains no target. The serving
layer must return a bounded `503` or apply an explicitly configured queue
policy; it must not retry a saturated pool forever. Retries must preserve the
prepared token IDs and re-evaluate current source state.

## P/D and multi-Router boundaries

P/D routing needs separate prefill and decode residency decisions. A decode
target is valid only when the selected prefill target's transfer path and
retention are known. `PdKvPairSelector` now supplies the CPU-tested atomic
selection boundary: both stages retain independent indexes, queries, trust
registries and scores; a pair is produced only when both stages select a
concrete `(instance_id, dp_rank)`. If either stage has no target, the pair is
incomplete. If one stage is degraded or falling back, the aggregate pair is
marked at least as degraded and never claims exact credit for the failed stage.

This selection result is not by itself an engine-side reservation.
`PdReservationCoordinator` supplies the next adapter-facing contract: reserve
prefill retention first, reserve decode admission second, and compensate the
prefill leg if decode is rejected. A successful coordinator returns both leases
as one guard; explicit release attempts decode first and prefill second, even if
one release call fails. The guard remains usable when a release is partial, so
the caller can retry the backend's idempotent release instead of relying only on
lease expiry.

Engines do not currently expose one distributed transaction across these two
stages. Adapters must therefore make `(attempt_id, stage)` reservations
idempotent and expire unreleased leases by `lease_ttl`. `attempt_id` identifies
one Router dispatch attempt and is deliberately separate from a client-visible
request ID. A retry of the same attempt reuses it; a new routing attempt that
may choose another target pair must use a fresh value. Otherwise a reused client
request ID could resolve to a stale lease for the previous target.

Decode idempotency is also bound to the exact prefill lease that authorized
it. A decode lease carries `prefill_lease_id`, while a prefill lease leaves the
field unset; the coordinator rejects any other value and releases both addressable
leases. This closes a partial-expiry window where the same attempt reacquires a
new prefill lease but a backend could otherwise reuse an older decode lease
whose TTL has not elapsed. A caller-supplied transfer ID does not replace this
lease-generation check.

The bounded TTL contains cancellation and ambiguous timeout windows in which a
remote stage may have accepted a request whose response was lost. The reference
adapter tests in `tests/kv_pd_reservation_conformance.rs` exercise
attempt-scoped idempotency, cross-attempt isolation, exact prefill/decode lease
binding, partial prefill expiry, idempotent release and TTL recovery after
cancellation of the whole coordinator future. The registered Router P/D mode
still rejects `trusted_kv_aware` because a real connector lifecycle, transfer
fencing and engine acknowledgment contract are not integrated yet. The
coordinator is a tested integration contract, not evidence that P/D
disaggregation is complete end to end.

Callers may configure `PdReservationCoordinator::with_operation_timeout` to
bound each reservation and release call. The timeout must be non-zero and
shorter than `lease_ttl`; otherwise the intent is rejected before touching a
backend. A timed-out reservation reports cleanup as `Ambiguous`, not
`NotAcquired`, because the remote stage may have committed before its response
was lost. An in-process timeout still cannot guarantee cleanup when the caller
drops the whole coordinator future, so backend TTL expiry remains mandatory.

When the caller omits `transfer_id`, the prefill connector may allocate one in
its lease. Decode must inherit that value from the prefill lease. If the caller
does provide a transfer identity, both leases must match it.

For multiple Router replicas, source state is Router-local unless an
authoritative shared snapshot is supplied. Replicating only live deltas is
insufficient because a newly started replica has no baseline. A replica can
enter exact mode only after its own snapshot/clear barrier, or after importing
a checksum-valid, identity-matching snapshot whose sequence is at least its
observed high-water mark. The importer enforces a trusted control-plane owner
and monotonic epoch, but authenticated snapshot transport, leader election,
durable epoch allocation and active-active split-brain prevention do not exist.
Shared hints may narrow or require retention, but cannot prove residency.

When a Router replica is isolated or cannot reconcile, it degrades to load
routing. It must not serve stale exact credit to make availability look
healthy. Active-active ownership integration, durable epoch allocation and
cross-replica load exchange remain unsolved parts of Roadmap Issue 5.

## Metrics and receipts

The following bounded metrics are proposed:

| Metric | Labels |
| --- | --- |
| `router_kv_source_state` | state, reason, instance, rank, tier |
| `router_kv_source_transitions_total` | from, to, reason |
| `router_kv_exact_fallback_total` | reason |
| `router_kv_stale_source_age_seconds` | tier, reason |
| `router_kv_recovery_duration_seconds` | initial_state, reason |
| `router_kv_degraded_duration_seconds` | cause |
| `router_kv_recovery_total` | result: replayed, failed |
| `router_kv_incorrect_exact_credit_total` | invariant category |

Route receipts contain mode, target/rank, source status counts and candidate
load/cache scores. They never contain prompt text or token IDs. High-cardinality
object/backend identifiers stay out of metric labels.

## Failure-injection matrix

The minimum software matrix covers a synthetic topology with four engine
instances and 16 DP ranks per instance:

| Scenario | Required result |
| --- | --- |
| One DP source disconnects | One target becomes degraded; exact credit remains valid for the other 63 targets |
| One instance loses all 16 sources | 48 targets retain their own states; no cross-instance owner is cleared |
| Subscriber gap on one rank | That rank is fenced during contiguous suffix replay; failure keeps it unreconciled |
| Late Router attachment | Replay history from 0 before applying live sequence; no exact credit during backfill |
| Replay endpoint unavailable or incomplete | Clear partial ownership and require a later complete replay or clear barrier |
| Worker restart/incarnation change | Old incarnation events and snapshots are rejected |
| Delayed/future timestamp | Source degrades; receipt proves observation age |
| Snapshot behind watermark | Rejected without lowering the reconciliation watermark |
| Valid clear barrier | Only the scoped owner/tier is cleared; the source may return to healthy |
| GPU clear with shared Host/Disk copy | Shared lower-tier copy survives, with its own owner and credit |
| All targeted sources fail | Bounded no-target/503 path; no unbounded retry |
| Router replica restart | No exact routing until snapshot/clear reconciliation |
| Partial Router isolation | Local degraded/failover policy; no stale exact claim |
| Duplicate/out-of-order events | Idempotence or fencing follows the owner/tier sequence |

The current tests exercise the state machine, incarnation-scoped cleanup,
partial-pool selection and tier-scoped clear at the index boundary, including
a synthetic 4x16 topology. A P/D pair selector keeps prefill and decode
evidence isolated, forms no pair when either stage has no target, and degrades
the aggregate receipt when only one stage is trusted. Reservation-contract tests
cover prefill rejection, decode rejection with prefill rollback, invalid backend
leases, connector-generated transfer identity, operation timeouts and release
retries. Real TCP tests also cover late attachment, a live
sequence gap, no exact credit while replay is in progress, failed replay
fencing and recovery on a later clear. Wire-level tests cover the optional
Device-tier and shared-owner clear forms while proving that Host/shared clears
cannot become trusted Device barriers. Portable-snapshot tests cover exact
recovery, stale topology baseline, checksum tampering, identity/source-set
mismatch, incarnation and timestamp rollback, ID reuse, same-sequence topology
conflicts, canonical prefix ordering, idempotent replay, global block capacity
and failure atomicity. Fencing tests cover trusted-token mismatch, same-epoch
owner conflict, epoch rollback, epoch advancement and failed-import
non-advancement. A real 4x16 failure matrix, a vLLM publisher that emits scoped
clears, engine restart, authenticated snapshot transport, physical shared-
storage ownership, P/D pairing and multi-Router partition still require later
integration environments and are not covered by CPU-only tests.

## Open questions

- What wire object identifies a shared storage owner independently from the
  publishing inference worker?
- Which authority produces and authenticates portable snapshots for Mooncake or
  other storage without replay?
- Which connector implementations can satisfy the adapter contract for
  idempotent paired reservation, explicit release and bounded lease expiry?
- Which control plane allocates durable Router epochs, authenticates snapshot
  ownership and bounds split-brain duplicate behavior?
- What is the release threshold for incorrect exact credit versus fallback
  availability?
