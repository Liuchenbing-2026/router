# Trusted KV selection building block

This implements a conservative single-group selection building block for the
exact-KV roadmap. It uses the existing `TieredMatchProvider` and concrete
`RoutableTarget { instance_id, dp_rank }` identities. It does not register a
`--policy kv_aware` CLI option or start discovery/event subscriptions.

## Lifecycle and selection

1. Create one bounded `KvTrustRegistry` per cache identity and register each
   engine-reported target and incarnation. Registration starts untrusted.
2. Apply validated KV events to the existing index. Call `advance` with the
   actual per-rank sequence; duplicates do not refresh freshness. Gaps,
   rollbacks, disconnects and epoch mismatches revoke trust.
3. Only call `reconcile` after independently validating a complete snapshot or
   clear/reconciliation barrier. Live events and replay deltas alone never
   establish or restore completeness. Old epochs and old watermarks are rejected.
4. Supply healthy candidates with fresh active-request counts to
   `TrustedKvSelector::select`. A missing, stale or untrusted healthy source
   makes the entire pool use deterministic least-load fallback, without
   querying or crediting the KV index. Unhealthy candidates are excluded.
5. When every eligible source is trusted and fresh, query the real index.
   Credit only contiguous prefix segments beginning at block zero. Transfer
   credit is discounted by storage tier and remote locality. Combine the
   uncached-prefix fraction with normalized active load using configurable
   weights. Return a structured receipt without prompt text or token IDs.

`cache_weight` and `transfer_penalty` must be finite values in `[0, 1]`.
Freshness and registry capacity must be positive. Registry removal releases
an entry; refusing a new entry at capacity preserves safe fallback. On equal
cost, ordering by instance ID and DP rank makes replay deterministic.

## Current boundary

The index and registry supplied to a selection must refer to the same model,
tokenizer, cache-salt/tenant domain and group. This API does not establish those
identities itself. Callers must derive identities and trust from engine-side
evidence, not client-provided headers or hints. Load observations must also be
fresh; the selector does not poll backend metrics.

This step is not a completed production exact-KV policy. Wiring discovery,
rank-scoped lifecycle signals, snapshot reconciliation, tokenizer/cache identity
validation, prepared request tokens, forwarding/rank headers, bounded index
topology and metrics remains separate work. Multi-group hybrid aggregation,
multi-segment transfer scoring, P/D bypass and device performance are not
implemented or claimed here.

## Targeted verification

```bash
cargo +1.95.0 test --locked --test kv_trusted_selection
cargo +1.95.0 test --locked --lib kv_index
cargo +1.95.0 fmt --all -- --check
```

Tests use actual `KvBlockIndexer` residency and removal plus source-lifecycle
faults. A provider that panics when queried proves fallback avoids untrusted
KV reads. These are CPU software checks, not engine/NPU or performance tests.
