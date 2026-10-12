# Llama 3.1 8B - vllm-router Native Load Balancing

Kubernetes deployment for Llama 3.1 8B with vllm-router handling native load balancing across 2 backend pods.

## Overview

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Docker Image**: vllm/vllm-openai:latest
- **Parallelism**: TP8 (Tensor Parallel 8)
- **Architecture**: 2 independent vLLM backends, vllm-router load balances
- **Replicas**: 2 decode pods (no prefill/decode separation)
- **Resources per pod**: 8 GPUs, 800Gi memory, 180 CPUs
- **Namespace**: vllm-router-llama31

## Architecture

```
Client → vllm-router → [Pod-0:8200 (TP8/DP1), Pod-1:8200 (TP8/DP1)]
```

Both backends handle full inference (prefill + decode). Kubernetes pod discovery
registers each ready pod IP as a separate worker. Tensor parallel ranks cooperate
inside one engine; each pod has one DP rank. The router does not expand a shared
headless service name into fictitious DP workers.

## Prerequisites

1. **Kubernetes Cluster** with 16 GPUs
2. **Tools**: kubectl, helm, helmfile
3. **HuggingFace Token**: Set as environment variable before deploying
4. **Router image**: Replace the historical private image in
   `router-deployment.yaml` with an accessible image built from the current
   checkout, supporting `--service-discovery`. The YAML is not evidence that
   this historical image was built or validated against the current CLI.
5. **Image versions**: Pin the engine image and record its registry digest for
   actual runs; the existing `latest` placeholder does not freeze a benchmark.

## Deployment Steps

### 1. Set HuggingFace Token

Export your HuggingFace token as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

Example:
```bash
export HF_TOKEN=your_actual_token
```

The `deploy.sh` script will automatically create the Kubernetes secret from this environment variable.

### 2. Deploy Model Services and Router

```bash
cd scripts/k8s/llama3/vllm-router/native
./deploy.sh
```

This single command deploys:
- Namespace creation (if needed)
- HuggingFace token secret (from HF_TOKEN environment variable)
- 2 vLLM backend pods via helmfile (8 GPUs TP8 / DP1 each)
- Backend Kubernetes Services (headless service for pod discovery)
- vllm-router deployment
- vllm-router service

Backend readiness is required before the router manifests are applied. A timeout
exits with an error instead of proceeding with unavailable model pods.

### 3. Verify Deployment

```bash
kubectl get pods -n vllm-router-llama31
kubectl logs -n vllm-router-llama31 -l app=vllm-router -f
```

## Running Benchmarks

```bash
./run-benchmark.sh                # 100 prompts, 16 concurrency
./run-benchmark.sh 200 32         # 200 prompts, 32 concurrency
```

Or manually:
```bash
kubectl port-forward -n vllm-router-llama31 svc/vllm-router-llama31 10001:10001

# Then from a pod with vllm installed
vllm bench serve \
  --dataset-name random \
  --num-prompts 100 \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --random-input-len 1000 \
  --random-output-len 1000 \
  --endpoint /v1/completions \
  --max-concurrency 16 \
  --host localhost \
  --port 10001
```

## Configuration

### values.yaml

Key settings:
```yaml
decode:
  replicas: 2  # Number of backend pods
  containers:
  - name: "vllm"
    image: vllm/vllm-openai:latest
    args:
      - "--tensor-parallel-size"
      - "8"  # TP8 per pod
```

### router-deployment.yaml

Router watches ready backend pods in the deployment namespace and connects to
their individual engine ports. The manifest includes a ServiceAccount and a
namespace-scoped Role granting only `get`, `list`, and `watch` on pods:
```yaml
command:
  - vllm-router
  - --service-discovery
  - --service-discovery-namespace
  - vllm-router-llama31
  - --service-discovery-port
  - "8200"
  - --selector
  - llm-d.ai/role=decode
  - llm-d.ai/inferenceServing=true
  - --intra-node-data-parallel-size
  - "1"
```

With two ready backend pods, this creates two workers, each addressing one
TP8/DP1 engine. Pod replacement is handled through discovery rather than relying
on HTTP connection pooling to randomly resolve a shared DNS name. Native traffic
goes directly to port 8200; the headless service and routing sidecar are not used
as worker identities in this path.

The engine argument is `--tensor-parallel-size 8`. Its DP size defaults to 1.
`--intra-node-data-parallel-size` is a router argument, not a vLLM engine argument.
To experiment with TP1/DP8, change the chart, engine arguments and router DP size
together, then validate each addressed rank; that is a different deployment.

## Configuration Regression Checks

```bash
pip install -e '.[dev]'
python -m pytest -q py_test/test_native_k8s_example.py
bash -n scripts/k8s/llama3/vllm-router/native/deploy.sh
```

Run these commands from the repository root. The CPU tests check engine/chart
parallelism, GPU allocation, pod-specific routing, read-only discovery RBAC, and
failure when backend readiness times out using mocked command-line tools. They
run in the existing Python CI suite. They do not deploy Kubernetes, allocate GPUs,
or establish throughput and generation quality. Validate those separately with
the benchmark and evaluation commands above and record the actual images,
worker addresses, successful requests and model results.

## Monitoring

```bash
# Backend logs
kubectl logs -n vllm-router-llama31 -l llm-d.ai/role=decode -f

# Router logs
kubectl logs -n vllm-router-llama31 -l app=vllm-router -f

# Router metrics
kubectl port-forward -n vllm-router-llama31 svc/vllm-router-llama31 29000:29000
curl http://localhost:29000/metrics
```

## Cleanup

```bash
./cleanup.sh
kubectl delete namespace vllm-router-llama31
```

## Comparison Use Case

This setup is for comparing routing mechanisms:
- **vllm-router/native**: Native load balancing (this directory)
- **vllm-router/pd-disagg**: P-D disaggregation with vllm-router
- **llm-d/native**: llm-d native load balancing
- **llm-d/pd-disagg**: llm-d P-D disaggregation
