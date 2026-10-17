# Token-verified bidirectional KV metadata for static pull P/D

The native Rust router can retain Decode's KV transfer metadata and attach it
to the next Prefill request. The actual KV remains in the engines and moves
through their native NIXL or Ascend Mooncake pull connector. This does not implement a new engine
connector or infer physical residency from a conversation identifier.

The feature is disabled by default. Enable it in the native binary:

```bash
vllm-router --vllm-pd-disaggregation --kv-connector nixl \
  --prefill http://prefill:8100 --decode http://decode:8200 \
  --bidirectional-kv-config '{"ttl_secs":450}'
```

Current scope is static HTTP NIXL **pull** or Ascend Mooncake V2 **pull**,
DP=1, without EPD, IGW or Program admission.
The Python launcher does not expose this option. Unsupported topologies fail
configuration validation instead of silently enabling an untested path.

Both engines must enable native bidirectional transfer, with a decoder TTL
longer than the router TTL. For example:

```json
{
  "kv_connector": "NixlConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "bidirectional_kv_xfer": true,
    "decoder_kv_blocks_ttl": 480,
    "kv_recompute_threshold": 64
  }
}
```

Use `kv_role: kv_consumer` on Decode. The recompute threshold and physical
block TTL belong to the engine, not the router. Follow the native
[NIXL usage guide](https://github.com/vllm-project/vllm/blob/cffc0cfa311f4997be58cfe8f5f6473ab4898269/docs/features/nixl_connector_usage.md)
for transport and platform requirements. A stock Ascend installation without
NIXL is not a validated substitute for CUDA device-buffer NIXL.

### Ascend NPU path on v0.30.0rc2

Use `--kv-connector mooncake-pull` with the companion
[Ascend adaptation](https://github.com/Liuchenbing-2026/vllm-ascend/tree/dda01353a89c8f1ab1b5b2a9322ec1f7be76f7b1),
based on Ascend `v0.30.0rc2` (`8fde6d4e335fe300e2b949aab0e31659cce0811d`).
The rc2 Dockerfile pairs it with core vLLM `v0.30.0`; record the actual image
digest and installed versions during validation.

The engine connector is `MooncakeConnectorV2`, and the example launcher is
`examples/disaggregated_prefill_v1/serve_bidirectional_multiturn.sh` in that
repository. Use TP/PP/DP/PCP/DCP=1, full attention, no speculation, and two free
NPUs. Both engines enable `bidirectional_kv_xfer`. This is direct READ reuse,
with a source-page claim before reverse DMA; it does not use a Mooncake Store.
The legacy `--kv-connector mooncake` selects a different push protocol and is
unchanged. A receiver crash after claiming can leave source pages pinned until
the source engine is restarted; this experimental extension has no distributed
lease recovery.

Run the same benchmark with `measure --transport mooncake`, which reads
P-side `vllm:mooncake_bytes_transferred_sum`. The gain and token-ID acceptance
criteria remain the same. CPU mock success does not establish physical NPU
transfer or TTFT improvement. Integrated device/model acceptance is pending.
On the companion rc2 image, 217 Mooncake CPU tests passed. The updated Router
passed 780 selected Rust tests and strict Clippy. A standalone NPU-buffer
diagnostic initially failed native teardown. Selecting the device before
importing Mooncake fixed it: two real-device round trips verified both 4 MiB
directions and exited normally. Restoring the old import order reproduced
both worker aborts. This is a diagnostic-entry fix; integrated engine/model
correctness and TTFT remain unmeasured. See the companion archive for the
exact environment and limitations.

## Client contract and prefix correctness

Send `conversation_id` and `return_token_ids: true` on every participating
request. The router consumes `conversation_id`; the worker never receives it.
The token-ID extension remains visible to the client, which must preserve the
complete generated history for reuse.

```json
{
  "model": "served-model",
  "conversation_id": "session-42",
  "prompt": [100, 200, 300],
  "max_tokens": 32,
  "return_token_ids": true,
  "stream": true
}
```

Token-ID `/v1/completions` requests can be checked directly. String completions
and plain-text chat use the selected Prefill engine's native `/tokenize` only
when there is a prior candidate. A tokenization failure becomes ordinary
prefill. The same chat-template options are forwarded to `/tokenize`.
The router uses actual `prompt_token_ids`, generated `token_ids`, and the
engine's `remote_num_tokens` to hash the computed prefix. It does not assume
that matching message text implies matching model tokens.

This matters for reasoning models: a chat template may strip earlier thinking
tokens. Such a rewritten prefix is a cache miss and is recomputed. Requests
without raw IDs, multiple outputs, truncation, embeddings, multimodal inputs,
LoRA/cache-salt overrides, and chat tool-definition rendering are conservatively
excluded. Agent harnesses can use pre-rendered token-ID completions to preserve
their exact tool/thinking history.

## Lifetime, bounds and observation

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_entries` | 1024 | Maximum tracked conversations, including active turns |
| `max_bytes` | 16777216 | Maximum retained transfer-metadata bytes |
| `max_entry_bytes` | 1048576 | Per-entry metadata and response-observation byte bound |
| `max_tokens` | 65536 | Per-turn token-proof bound |
| `ttl_secs` | 450 | Conservative local lifetime, shorter than the configured engine TTL |

The cache is scoped by credential, tenant header, model, endpoint and
conversation. It retains hashes and block metadata rather than prompt text.
Entries are consumed once. Concurrent turns invalidate prior generations;
late callbacks cannot republish them. Worker removal/replacement or unhealthy
sources cause a miss. An error, truncated stream, client cancellation, malformed
metadata or exhausted bound never publishes a reusable entry.

The native D-side `remote_host` is preserved: a NIXL side-channel address may
differ from the HTTP/API address. Only when it is omitted does the router use
the configured decoder's HTTP hostname. Worker metadata is trusted operator
control-plane input; client-supplied transfer parameters are not the cache source.

TTL starts before dispatch, not when the final metadata reaches the router.
Slow generation or transport therefore does not renew an engine lease. Native
`remote_blocks_expiry_time` is passed through unchanged: a remote monotonic
clock must not be converted into local wall time. Operators must keep router
TTL below the actual decoder TTL. A restart/source error is still subject to
native connector error handling; the router does not replay a partially emitted
response or guarantee residency after an engine restart.

Streaming observation tolerates arbitrary byte splits and CRLF/LF framing,
preserves response bytes, and follows downstream backpressure. Metadata is
committed only after a successful complete stream. No detached queue buffers
the response.

`x-router-kv-reuse` and the bounded-label counter
`vllm_router_bidirectional_kv_requests_total{reason="..."}` report routing
outcomes. **`candidate` means metadata was forwarded, not that an engine applied
it or that physical D→P transfer occurred.** Inspect P-side native transfer
metrics to prove the latter.

## Validation and real-device performance gate

```bash
cargo test --locked --lib bidirectional_kv
cargo test --locked --test bidirectional_kv_routing
cargo clippy --locked --lib --bin vllm-router --test bidirectional_kv_routing -- -D warnings
cargo fmt --all -- --check
```

The tests use TCP mock workers. They cover metadata forwarding, rendered-token
validation, rewritten history, single-use consumption, replacement, expiry,
generation fencing, cancellation, stream fragmentation and capacity bounds.
They do not establish GPU/Ascend transfer correctness or a TTFT speedup.

### Software results, 2026-10-17

The aarch64 CPU run used Rust/Cargo 1.95.0 and Python 3.12.13 in
`quay.nju.edu.cn/ascend/vllm-omni:v0.30.0`, with image digest
`quay.io/ascend/vllm-omni@sha256:274d65369c912c14cc241f53c9c900d95e844b91652ca604cc71f07a9e1b3aaf`.
It reused installed tools and dependency caches; it was not a clean image build.

| Check | Result |
| --- | --- |
| Library tests, including nine new lifetime/protocol tests | 767 passed |
| Native CLI unit tests | 4 passed |
| Bidirectional configuration tests | 2 passed |
| Bidirectional TCP tests | 3 passed |
| Existing P/D integration tests | 4 passed |
| Total selected Rust tests | 780 passed, zero failed/ignored |
| Clippy | all targets, all features, `-D warnings` passed |
| Format | passed |
| Integrated engine KV/model correctness and TTFT gain | not measured |

Separate real Rust-binary/TCP-mock probes observed no D metadata on the second
P request in the earlier delivered binary, and the expected D metadata in the
new binary. The earlier binary is not an exact build of this task's base:
the base also includes a prefill `min_tokens` fix. No timing comparison is
claimed between those software probes. These are local checks, not upstream CI.

The example launcher and benchmark require an existing CUDA/vLLM/NIXL
environment and two free devices. They do not provision machines:
The launcher uses two TP=1 engines on one host; it does not reproduce a
two-host, TP=8 AWS/EFA topology. Measuring requires `httpx`; offline comparison
uses the Python standard library.

```bash
export MODEL=/absolute/path/to/fixed-model-snapshot
export ROUTER_BIN=/absolute/path/to/target/release/vllm-router

# Terminal 1, OFF. P prefix caching is deliberately disabled to model eviction.
LEG=off RUN_DIR=/absolute/path/run-pair1-off \
  bash examples/bidirectional_kv/serve_pd_multiturn.sh

# Terminal 2, after all services are healthy.
python examples/bidirectional_kv/benchmark.py measure \
  --leg off --pair 1 --url http://127.0.0.1:29112 \
  --prefill http://127.0.0.1:29110 --model bidir \
  --output /absolute/path/results-pair1-off

# Stop the task-owned OFF launcher with Ctrl-C; restart BOTH engines and router.
LEG=on RUN_DIR=/absolute/path/run-pair1-on \
  bash examples/bidirectional_kv/serve_pd_multiturn.sh
python examples/bidirectional_kv/benchmark.py measure \
  --leg on --pair 1 --url http://127.0.0.1:29112 \
  --prefill http://127.0.0.1:29110 --model bidir \
  --output /absolute/path/results-pair1-on

python examples/bidirectional_kv/benchmark.py compare \
  --off /absolute/path/results-pair1-off/results.json \
  --on /absolute/path/results-pair1-on/results.json \
  --output /absolute/path/comparison-pair1.json
```

The strict comparison requires identical inputs and output token IDs, later-turn
candidate receipts, zero P-side NIXL read bytes in OFF, positive P-side read bytes
in ON, and at least 1.05× median later-turn TTFT speedup at every tested context
length. Missing metrics, quality mismatch, missing samples or no gain fail the
gate. The failure is preserved, not converted into a performance success.

Repeat with fresh engines in AB/BA/AB order before a broader claim. Keep the
model hashes, exact source/package revisions, image tag/digest, hardware,
launch logs, raw requests, P metrics, failures and timing intervals. This is a
prefix-eviction workload; also evaluate P prefix caching enabled for a normal
operating baseline. It is not a production-goodput or complete-Agent benchmark.
No real-device bidirectional speedup is claimed by these software tests or by
the unexecuted example commands.
