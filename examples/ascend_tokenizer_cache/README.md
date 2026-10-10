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

## Verification plan — 2026-10-11

| Evidence boundary | Current scope |
| --- | --- |
| This change | Repository-specific acceptance plan; no new production-routing implementation |
| Existing software checks | Historical native/fixture results above; not a real model or NPU qualification |
| Live acceptance | Pending; no hardware has been allocated for this plan |
| Release | Correctness → native ingress → fault recovery → isolated performance → full-task evaluation |

先验收 **token ID 一致性**，再验收 **事件索引真实性**，最后验收 **任务收益**。三者不能互相替代。
Host 分词缓存、Router whole-prompt L0、引擎物理 KV residency 是三个独立对象；逻辑 token 数不因编码缓存命中而下降。

### Installation prerequisites and immutable experiment record

| Field | Required record |
| --- | --- |
| Repository | Base SHA from this README; task branch `xxx_tokenizer_cache`; tested HEAD from `git rev-parse HEAD` |
| Paired engine | Ascend `4ea839b813fecbd1aba3affa63e3c896e75b123b`; vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; check whether each dependency has merged at execution time |
| Container | Full registry/tag and actually pulled digest; additional dependencies, OS/architecture, Python/Rust/Go versions; no invented image or digest |
| Model | Weight revision/digest, tokenizer files, chat-template digest, LoRA revision, dtype/quantization and context length |
| Hardware | Allocated physical cards, device model, driver/runtime versions, interconnect, TP/PP/DP topology, API-process count |
| Runtime | Complete launcher command and additional-config, routing config digest, connector/version, event endpoints, seeds and TTL/capacity |
| Workload | Dataset revision/hash, task/repository/time split, prompt/output lengths, arrivals/concurrency, warm-up, timeout and retry budget |

Run the installed-file contract check in `launch_backend.py` before starting workers. Keep credentials in environment variables. Public artifacts contain synthetic inputs and aggregate evidence; keep restricted raw inputs and results local. Model serving, load tests and NPU occupancy need explicitly assigned resources; none are started by this document.

### Shared backend conformance commands

Run from this example directory. All real-service commands below are **plans, not completed tests**. OFF and ON need separately allocated comparable resources; a sequential crossover on the same cards must restart/reset between conditions and alternate order. Keep KV policy, API-process count, weights and template identical.

~~~bash
python3 -m unittest -v test_example.py test_serving.py
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 0 --port 8000 \
  -- --served-model-name local-model --max-model-len 8192
python3 launch_backend.py --model Qwen/Qwen3-0.6B --cache-gb 4 --port 8001 \
  -- --served-model-name local-model --max-model-len 8192
python3 verify_tokens.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 \
  --output token-conformance.json
python3 verify_serving.py --baseline http://127.0.0.1:8000 \
  --cached http://127.0.0.1:8001 --model local-model --rounds 3 --serve \
  --output render-serving-conformance.json
~~~

`Qwen/Qwen3-0.6B` is an inexpensive text smoke-test example, not DeepSeek/DP16 certification. Use the actual production weights/template for final conformance. `verify_serving.py` calls backend render/generate, **not the entire upstream router/Agent**. Capture actual upstream ingress separately under the cases below. Check every complete token ID, not only count; reject malformed IDs, changed `usage.prompt_tokens`, and render/serving mismatch. Extend the live corpus to empty/truncated inputs, tools and unsupported modalities without silently skipping failures. Output text hashes are diagnostic; generation quality requires an independent task judge.

### Repository-specific cases

| ID | Test object | Input / failure / execution | Independent oracle and pass condition | Current state |
| --- | --- | --- | --- | --- |
| VR-01 | gRPC L0编码链 | L0 OFF/ON；whole prompt重复/增长、encoded输入、salt、LRU/oversize、retry | 完整token IDs不变；重试只用一次准备结果；HTTP不算gRPC L0收益 | 7项L0+20项cache+1项native fixture历史通过 |
| VR-02 | 事件索引语义 | 重放Store/Remove/Clear、重复token位置、共享pool、group和Rank隔离 | 位置敏感；移除中间块截断连续前缀；保留子树；Worker clear不删pool；claim幂等 | 已有indexer源码单测；本轮复测记录独立 |
| VR-03 | 严格发现与解码 | 未知medium/hash、畸形msgpack、block数不齐、负token、rank越界、地址不合法 | 无静默zip截断/Advance；缺/未知契约不默认Device或exact；源降级有证据 | 源码缺口；#352仍open，不当已合入 |
| VR-04 | 事件序号与信任 | 重复、乱序、可恢复gap、超replay窗口、restart、静默源、启动无snapshot | 只有连续且验证后的事件提升watermark；未对账源credit=0；live流不足恢复EXACT | trust arbiter未接入；待实现 |
| VR-05 | 4实例×DP16 | 64个(instance,rank)，仅实例2/rank7有长前缀；伪造客户端rank头；引入热rank过载 | 先选正确target再覆盖rank头；实际worker日志一致；健康/过载优先；其余rank不冒算命中 | 目标RFC能力，当前kv_aware未接线 |
| VR-06 | Regular/P-D生命周期 | cancel、timeout、client disconnect、retry、P2xx但坏handoff、D失败 | slot仅释放一次；P/D分开commit；旧generation无迟到提交；stream开始后不重播 | 待M5/M7/M8和generation fence |
| VR-07 | 四策略与成本消融 | P2C、consistent_hash、cache_aware、kv_aware；另以RR作sanity；overlap权重0、temperature0 | 所有策略同trace/容量；报告错误credit、实际reuse、尾延迟和goodput；不把热命中作性能结论 | M12待开发，kv_aware命令尚不可执行 |
| VR-08 | 异构与HA边界 | 多模型/LoRA/salt/group/host/disk/sharedpool；后续多Router和checkpoint | 缓存身份隔离；下层tier不假装Device；v1只承诺单Router，HA/混合检查点后置 | F5/F8后置；禁止扩大当前支持 |

### Failure reproduction and result recording

Each case has five fields in the actual run record: **phenomenon**, **steps** (fixed versions/input/flags), **expected versus observed**, **confirmed cause versus hypothesis**, **fix and rerun result**. Preserve the first failing trace and every unresolved/skipped condition. Assign the request/attempt ID before ingress; attach route ID, model/template identity, worker incarnation, endpoint/rank, event sequence and engine response to the same trace. A recovered request does not establish a repaired root cause.

### Performance and quality gates (proposed; not achieved)

| Gate | Required comparison / measurement |
| --- | --- |
| G0 — correctness | Zero full-token-ID mismatches, unauthorized cross-identity credit, unsafe handoff and resource leaks in the declared deterministic corpus; explicitly count unknown/skipped cases |
| G1 — native ingress | Actual router/Agent ingress reaches the registered model/endpoint/rank with unchanged semantics; standalone backend probes cannot pass this gate |
| G2 — recovery | Scripted gaps, clear/restart, cancellation, saturation and malformed events revoke trust correctly; no exact claim before validated reconciliation |
| G3 — isolated benefit | Cache OFF/ON × routing policy factorial; fixed hardware/trace/quality; cold/warm/churn separately; measure CPU encode, RSS, actual KV reuse, TTFT P99, completion P95, SLO goodput |
| G4 — task value | Fixed strong, fixed main and simple rules as controls; held-out tasks; total spend including failed attempts, retries, classifier and fusion divided by successful tasks; quality and latency constraints decided before running |

Use at least five paired seeds/arrival traces for the initial comparison and report paired bootstrap 95% confidence intervals. This repeat count and all performance thresholds are **project proposals**, not an upstream guarantee; compute the necessary sample size from variance and the business non-inferiority margin. Zero observed failures does not prove impossibility outside the tested corpus. Do not select the best run or tune on the held-out set. A busy server with no completions must report zero goodput and undefined successful-task cost rather than omit the run.

Before a load test, inspect the **pinned** engine benchmark entrypoint with `vllm bench serve --help`; record the exact supported flags. If that revision lacks this CLI, use its checked-in serving benchmark entrypoint and record its SHA. Do not copy speculative `kv_aware` flags into a runnable command: at Router base `91b65ee4`, that policy is not wired. A trace runner, task judge and fault injector are still required beyond the current conformance helpers.

The paired vLLM revision above was inspected at `vllm/entrypoints/cli/benchmark/serve.py`, `vllm/benchmarks/serve.py` and `vllm/benchmarks/datasets/datasets.py`; the following flag names exist there. This is a **planned synthetic cold-load example**, not a measured workload, warmed-prefix replay or Agent evaluation. Run only after G0/G1 pass and resources are assigned. The rate/length/concurrency values are illustrative and must be replaced by the preregistered workload; `127.0.0.1:8080` must be the actual reachable upstream ingress. Use the installed tokenizer matching production weights, not the model alias as a tokenizer path.

~~~bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8080 --endpoint /v1/chat/completions \
  --model local-model --tokenizer Qwen/Qwen3-0.6B \
  --dataset-name random --random-input-len 4096 --random-output-len 128 \
  --num-prompts 1000 --request-rate 4 --max-concurrency 32 --seed 1 \
  --percentile-metrics ttft,e2el --metric-percentiles 50,95,99 \
  --save-result --result-dir benchmark-results
~~~

Repeat across the fixed policy/cache conditions with separate result directories, alternating run order and identical generated inputs/arrival seeds. `e2el` is end-to-end request latency, not full Agent-task time. Random prompts do not establish prefix-reuse accuracy; warm/churn/PD and model-selection quality require the dedicated traces and oracles in the table. This CLI does not measure classifier/Agent loop costs unless those are actually on the tested path.

Full RFC implementation and shared validation design:
https://github.com/Liuchenbing-2026/Model_test/blob/router/agent-routing-evaluation/项目交付/任务列表/任务13%20llm-d%20Router调研与可复现指南/详细任务/路由方案/README.md#validation-plan-20261011

### Native verification entrypoints

Run from repository root with the CI-pinned toolchain for new changes; these commands exist at the fixed source:

~~~bash
cargo +1.95.0 test --lib kv_index -- --test-threads=1
cargo +1.95.0 test --lib kv_hash -- --test-threads=1
cargo +1.95.0 test --lib backend::l0::tests -- --test-threads=1
cargo +1.95.0 test --lib tokenizer::cache::tests -- --test-threads=1
cargo +1.95.0 test --test grpc_frontend_l0 -- --test-threads=1
cargo +1.95.0 fmt --all -- --check
cargo +1.95.0 clippy --all-targets --all-features -- -D warnings
~~~

`kv_index` is a library foundation at this base; `KvIndexSupervisor` has no production policy consumer in `server.rs`/HTTP Router. Discovery currently calls `/get_server_info` and `/kv_event_sources`, unlike RFC shorthand `/server_info`; probe actual engine endpoints and fail closed when metadata is missing. The latter endpoint is a deployment compatibility requirement, not proved available by the Router unit tests. `src/kv_hash.rs` conformance tests cover a narrow `sha256_cbor` sequence-hash utility, not all salts/MM/group cases or physical hit accuracy. Upstream #352 is still open at this check and must not be presented as shipped strict decoding.

First make VR-03/04 and generation fencing pass before allowing positive event-index credit in a native `kv_aware` policy. Existing character history or session hints may influence eligible targets but must never be written into the physical KV index or labeled reusable KV tokens.
