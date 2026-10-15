# KV-aware routing and trusted selection

`kv_aware` is an opt-in policy for regular, co-located, normal-dense group-0
attention with worker-owned Device KV. It combines verified prefix overlap
with router-local active request load. The existing `cache_aware` policy keeps
its behavior.

## Running the policy

Build the current branch and select the policy explicitly:

```bash
cargo build --locked --bin vllm-router
./target/debug/vllm-router --host 127.0.0.1 --port 3001 \
  --worker-urls grpc://127.0.0.1:50051 \
  --policy kv_aware --kv-cache-weight 0.5 \
  --kv-max-event-age-secs 30 --kv-max-sources 1024 --kv-max-blocks 65536
```

These are example service commands; the CPU verification below does not start
a real inference engine. Rebuild the Rust extension before selecting
`PolicyType.KvAware` through the Python API or `--policy kv_aware` through the
Python launcher. The Python entry uses the same default limits.

For gRPC workers, discovery uses Control.GetServerInfo, GetModelInfo and
GetKvEventSources. Chat requests reuse the token IDs already produced by
EngineFrontend.prepare, including across retries. DP-aware gRPC workers receive
the selected rank through the existing gRPC routing metadata.

HTTP workers must expose `/get_server_info` with `instance_id`, `model_id` (or
`model_path`) and `kv_block_size`, plus `/kv_event_sources` keyed by DP rank.
Each enabled ZMQ publisher must have its own endpoint. Wildcard bind hosts are
resolved against the worker origin and connect endpoints are validated. The
configured API key is used for HTTP discovery. An engine without these
endpoints remains eligible for load fallback.

HTTP `/v1/completions` requests with a single token-ID prompt can use exact
selection. Text-only HTTP requests use load fallback, preserving transparent
proxy behavior. The selected DP rank overrides a client-supplied
`X-data-parallel-rank`; request tokens are forwarded unchanged.

## Completeness, identity and lifecycle

Registration starts untrusted. A publisher-emitted `AllBlocksCleared` observed
after subscription is the supported completeness barrier. Merely receiving a
live event, replay delta, or sequence zero does not establish a complete view.
The router does not clear engine caches itself. Operators must arrange a valid
barrier before expecting exact selection.

Event decoding, normal-dense shape admission, rank binding, sequence state,
index publication and selection are fenced by one runtime lock. A dropped
advisory signal cannot leave stale data visible as exact. Gaps, rollbacks,
invalid batches, wrong ranks, disconnect errors and stale timestamps revoke
trust. Repeated sequence numbers do not refresh freshness. Recovery requires
a new valid clear barrier; replay/snapshot reconciliation is not implemented
by this runtime. Publisher and router clocks must be synchronized within the
configured freshness window; timestamps over one second into the future are
rejected.

Worker metadata binds each source to an engine instance, model, block size and
addressable DP rank. Duplicate instances/endpoints and duplicate canonical
ranks are rejected. Removing or replacing a worker revokes its discovery lease
and stops its subscriber tasks; late discovery responses and queued events
cannot reattach the removed generation. Unhealthy/circuit-open workers are
excluded from selection.

The supported deployment must be normal-dense full attention in group zero.
Shape admission does not discover an engine's complete model architecture.
Hybrid/linear attention, sliding windows, LoRA, cache salts, shared external
ownership and multimodal requests are outside this runtime contract. Such
request namespaces or observed event extensions use safe fallback. This mode
must not be used to claim complete hybrid KV reuse. P/D, EPD and IGW combinations
are rejected at configuration validation.

## Selection, load and observability

All healthy candidates must share a cache identity and have fresh, reconciled
sources. Otherwise the entire eligible pool uses deterministic least-load
fallback without querying untrusted residency. Exact selection credits only
contiguous Device prefixes of complete blocks; the final prompt token is not
credited. It blends uncached-prefix fraction and normalized active request
load using `cache_weight`. This is a routing heuristic, not a calibrated
latency model or a measured performance gain.

One load guard covers each KV attempt. Errors, retries, cancellation before
headers, non-streaming completion and stream termination release that attempt
once. HTTP streaming polls upstream with downstream demand and does not create
an unbounded forwarding queue. The existing gRPC producer holder handles
terminal and dropped-body cleanup.

Responses expose `x-router-kv-mode` (`exact` or `fallback`) and
`x-router-kv-reason`. Debug route receipts contain target/rank and candidate
scores, never prompt text or token IDs. Prometheus counters are
`router_kv_decisions_total{mode,reason}` and
`router_kv_ingestion_failures_total{reason}`; reason labels are bounded constants.

Source count is capped globally. `max_blocks` conservatively charges stored
block announcements, including aliases and duplicates, because retained index
metadata can outlive residency. Removals do not refund this historical budget.
At exhaustion the affected identity's index is discarded and its sources lose
trust; new clear barriers are required. The last source's removal releases the
identity and its budget. This bounds index growth, not the entire process heap
or transport buffers.

## Library component

`kv_index::routing` still exposes KvTrustRegistry and TrustedKvSelector for
other engine-owned integrations. That selector can discount storage tiers and
remote locality; the registered runtime admits only Device events. Callers of
the library must independently prove complete reconciliation and cache identity.
Client headers/hints must never establish trust.

## CPU verification

```bash
cargo +1.95.0 test --locked --test kv_aware_runtime \
  --test kv_trusted_selection --test agent_hint_request_hints
cargo +1.95.0 test --locked --lib --bin vllm-router
cargo +1.95.0 clippy --locked --lib --bin vllm-router \
  --test kv_aware_runtime --test kv_trusted_selection \
  --test agent_hint_request_hints -- -D warnings
cargo +1.95.0 fmt --all -- --check
```

The integration tests use TCP ZMQ, HTTP and tonic services, actual index
residency/eviction, DP metadata and cancellation. They establish software
behavior; real model reuse, engine acceptance, NPU/GPU performance, hybrid
aggregation and HA remain separate acceptance work.
