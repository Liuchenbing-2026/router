# Ascend tokenizer-cache backend integration

| Delivery | Status |
| --- | --- |
| Backend startup and offline verification helpers | Implemented and locally checked |
| Native gRPC prompt encoding | L0 request-chain wiring implemented and locally tested; HTTP forwarding unchanged |
| End-to-end integration and hardware acceptance | Pending |

## Component integration

After starting compatible workers, use the existing HTTP forwarding path:

~~~bash
vllm-router --host 127.0.0.1 --port 8080 \
  --policy power_of_two --worker-urls http://127.0.0.1:8000
~~~

For replica tests, add independently reachable worker URLs. Keep normal JSON
requests on this path so the Ascend renderer receives text. In the pinned
source, src/tokenizer/cache.rs already provides CachedTokenizer. This branch
now reuses its bounded cache in the native gRPC frontend; that path is separate
from the Ascend segment cache. HTTP forwarding still sends text to the backend.
This delivery does not implement an exact-KV policy or change policy selection.

Exact-input reuse for future KV-aware routing must consume the shared
request-token contract. Exact history is a bounded routing hint and must remain
separate from physical KV residency. The proposed initial correctness scope
is independently addressable DP=1 workers. Shared-endpoint native DP, unified
cost selection, and complete recovery need their own implementation and tests.

## Backend contract

| Item | Value |
| --- | --- |
| vLLM-Ascend source | 4ea839b813fecbd1aba3affa63e3c896e75b123b |
| Paired vLLM reference | ced6857afa0ea7b2e3f0846a62e1394e90f15607 |
| Cache option | additional_config.tokenizer_cache_gb; default 0 |
| Local checks | Python 3.9.6, stdlib; no image or inference service used |
| Hardware, throughput, quality | Not run |
| Task branch | xxx_tokenizer_cache |
| Component base SHA | 91b65ee4bdfbe98863a32a2205dced5dbb249ffe |
| Delivery SHA | Commit containing this example; inspect with git log -1 -- . |

Install the paired sources in an Ascend environment with its compatible
CANN, torch-npu and model dependencies. source_contract.json records fingerprints
of the relevant installed patch/config/renderer files. The launcher verifies
these files before execution. This detects this specific interface mismatch;
it does not certify the complete environment or all model templates.

Source: [Ascend frontend cache](https://github.com/vllm-project/vllm-ascend/pull/16298).
The patch remains an unmerged dependency. Engine CLI arguments belong to the
backend startup command; do not insert them into an OpenAI request body.

## Commands

Dry-run command construction without installing the backend:

~~~bash
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 4 --dry-run \
  -- --served-model-name local-model --max-model-len 8192
python3 -m unittest -v test_example.py
~~~

After preparing the pinned backend, start OFF and ON servers in separate
terminals, with separately allocated resources:

~~~bash
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 0 --port 8000 \
  -- --served-model-name local-model --max-model-len 8192
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 4 --port 8001 \
  -- --served-model-name local-model --max-model-len 8192
python3 verify_tokens.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 \
  --output token-conformance.json
~~~

The service commands above are provided for hardware execution and were not
run locally. The comparison requires complete token-ID equality across OFF,
ON and repeated synthetic inputs. Report a failure on different IDs,
nondeterminism or malformed responses. Its per-call timing is diagnostic;
sequential /tokenize calls are not an inference load benchmark.

For workload evaluation, keep the same model, template, API-process count,
hardware, arrival trace and KV-routing policy. Measure tokenizer CPU time,
process RSS, TTFT, completion latency and success-task cost in the existing
benchmark harness. Include tools, truncation and template changes in model
conformance before deployment. No performance result is supplied here.

## Boundaries

The cache stores text-segment-to-token-ID results in host memory. Its budget
is an estimate per tokenizer instance rather than an RSS limit. API-process
count and DP-rank count are separate. A cache hit does not establish physical
KV residency, reduce logical request token count, or enable cross-model KV
reuse. Cache-OFF preserves the budget switch; wrappers still require the
paired renderer interfaces.

Public offline checks cover launcher argument preservation, invalid budgets,
installed-file mismatch and complete-ID comparison failure. These checks do
not substitute for native Router builds, real backend conformance, cluster
tests, generation/eviction recovery or P/D acceptance.

## Native Rust L0 request-chain wiring

Ported the fixed source of upstream [frontend L0 integration](https://github.com/vllm-project/router/pull/340)
from `de07603c11b3be0c22c45ea3036702d46ad650d5` onto component base
`91b65ee4bdfbe98863a32a2205dced5dbb249ffe`. That upstream contribution is
still open/unmerged at the 2026-10-10 check; this task branch contains the
implementation, not a claim that the upstream PR has merged. Its redundant
manifest dependency addition was excluded; existing dependency versions stay.

Set `VLLM_ROUTER_L0_CACHE=1` on a Router using its existing gRPC backend
configuration. The default is disabled. Capacity knobs are
`VLLM_ROUTER_L0_MAX_ENTRIES`, `VLLM_ROUTER_L0_MAX_BYTES`, and
`VLLM_ROUTER_L0_MAX_ENTRY_BYTES`. See `docs/backend/grpc.md` for the native
worker protocol and setup; ordinary HTTP workers do not use this frontend.

Flow: chat template/render -> PromptEncoder -> bounded L0 -> PreparedChat ->
selection/retry -> GenerateRequest. A hit requires the same tokenizer instance,
special-token option, cache salt and **whole rendered prompt**. Growing-prefix
reuse is not implemented. Encoded inputs bypass text encoding. Retry reuses
the already prepared IDs. Oversized entries are returned without retention;
unknown or randomized tokenizer configurations bypass caching.

Actual local commands, Rust 1.99.0 on macOS arm64 (no inference container):

```bash
cargo test --lib backend::l0::tests -- --test-threads=1
cargo test --lib tokenizer::cache::tests -- --test-threads=1
cargo test --test grpc_frontend_l0 -- --test-threads=1
```

Results: 7 native L0 units, 20 existing cache regressions and 1 native HTTP /
tonic gRPC integration test passed. The integration uses a real tokenizer
fixture and simulated workers, including retry/streaming/salt/budget checks.
No NPU model, physical KV index, shared-endpoint DP16, or inference benchmark
was run. Existing compiler deprecation warnings are preserved in the archive.

The broader backend regression suite passed 30 tests (including the 7 L0
tests above; do not count those twice). Formatting passed. Strict clippy
with the repository's CI-pinned Rust 1.95.0 passed:

```bash
cargo +1.95.0 clippy --all-targets --all-features -- -D warnings
```

This was a local check, not a CI run. The same strict command on Rust 1.99.0
reported inherited diagnostics in five unchanged source files; the failed log
is retained with the successful pinned-toolchain log.

## Native render and serving conformance (2026-10-10)

`verify_serving.py` supplements `/tokenize` checks with the actual chat-render
protocol and, with `--serve`, generation request ingress. It compares complete
OFF/ON token IDs, repetition, growing conversations, Unicode, tenant salts and
tools. Serving must return the same `prompt_token_ids` and prompt usage.

```bash
python3 -m unittest -v test_example.py test_serving.py
python3 verify_serving.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 --serve \
  --output render-serving-conformance.json
```

`BACKEND_API_KEY`, if required, is read from the environment. Never commit it.
The six harness tests use local HTTP fixture servers, not an inference model.
The serving command requires prepared OFF/ON backends and is **not executed**
on NPU in this delivery. Output timing is diagnostic only; the harness does not
measure generation quality, inference throughput, or full Agent success.
Generation text hashes are recorded without treating nondeterministic text as
a tokenizer-equivalence failure. Real workload evaluation must use held-out
Agent tasks and include retries, failures, complete latency and total cost.
