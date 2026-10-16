#!/usr/bin/env bash
# Run in an already-provisioned CUDA/vLLM/NIXL environment; creates no cloud resources.
set -euo pipefail
: "${MODEL:?Set MODEL to fixed local weights or a pinned Hugging Face model}"
: "${ROUTER_BIN:?Set ROUTER_BIN to the tested Rust vllm-router binary}"
: "${RUN_DIR:?Set RUN_DIR to a new task-owned directory}"
LEG=${LEG:-off}
if [[ "$LEG" != off && "$LEG" != on ]]; then exit 2; fi
mkdir "$RUN_DIR"
P_PORT=${P_PORT:-29110}
D_PORT=${D_PORT:-29111}
ROUTER_PORT=${ROUTER_PORT:-29112}
P_DEVICE=${P_DEVICE:-0}
D_DEVICE=${D_DEVICE:-1}
if [[ "$P_DEVICE" == "$D_DEVICE" ]]; then
  echo 'Use distinct free devices for the performance comparison.' >&2; exit 2
fi
BIDIR=false
ROUTER_EXTRA=()
if [[ "$LEG" == on ]]; then
  BIDIR=true
  ROUTER_EXTRA=(--bidirectional-kv-config '{"ttl_secs":450}')
fi
python - "$RUN_DIR" "$MODEL" "$LEG" "$P_DEVICE" "$D_DEVICE" "$P_PORT" "$D_PORT" "$ROUTER_PORT" <<'PY'
import importlib.metadata, json, pathlib, socket, subprocess, sys
gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
gpus = dict(row.strip().split(", ") for row in gpu_rows.splitlines())
selected = {gpus[sys.argv[4]], gpus[sys.argv[5]]}
active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
if any(row.split(",")[0].strip() in selected for row in active.splitlines()):
    raise SystemExit("Selected devices have existing workloads; nothing was started")
for port in [*map(int, sys.argv[6:9]), 29120, 29121]:
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", port))
info = {"model": sys.argv[2], "leg": sys.argv[3], "python": sys.version,
        "packages": {n: importlib.metadata.version(n) for n in ["vllm", "nixl", "torch"]},
        "devices": subprocess.check_output(["nvidia-smi"], text=True)}
pathlib.Path(sys.argv[1], "environment.json").write_text(json.dumps(info, indent=2))
PY
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM
CUDA_VISIBLE_DEVICES="$P_DEVICE" VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=29120 \
  vllm serve "$MODEL" --served-model-name bidir --dtype bfloat16 --enforce-eager \
  --max-model-len 32768 --max-num-seqs 8 --tensor-parallel-size 1 \
  --no-enable-prefix-caching --port "$P_PORT" \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"engine_id\":\"bidir-prefill\",\"kv_connector_extra_config\":{\"bidirectional_kv_xfer\":$BIDIR,\"decoder_kv_blocks_ttl\":480,\"kv_recompute_threshold\":64}}" \
  > "$RUN_DIR/prefill.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES="$D_DEVICE" VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=29121 \
  vllm serve "$MODEL" --served-model-name bidir --dtype bfloat16 --enforce-eager \
  --max-model-len 32768 --max-num-seqs 8 --tensor-parallel-size 1 --port "$D_PORT" \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_consumer\",\"engine_id\":\"bidir-decode\",\"kv_connector_extra_config\":{\"bidirectional_kv_xfer\":$BIDIR,\"decoder_kv_blocks_ttl\":480,\"kv_recompute_threshold\":64}}" \
  > "$RUN_DIR/decode.log" 2>&1 &
pids+=("$!")
python - "$P_PORT" "$D_PORT" <<'PY'
import sys,time,urllib.request
for port in sys.argv[1:]:
    deadline=time.monotonic()+1200
    while True:
        try:
            with urllib.request.urlopen("http://127.0.0.1:"+port+"/health", timeout=2): break
        except Exception:
            if time.monotonic()>deadline: raise
            time.sleep(2)
PY
"$ROUTER_BIN" --host 127.0.0.1 --port "$ROUTER_PORT" \
  --vllm-pd-disaggregation --kv-connector nixl \
  --prefill "http://127.0.0.1:$P_PORT" --decode "http://127.0.0.1:$D_PORT" \
  --policy round_robin "${ROUTER_EXTRA[@]}" > "$RUN_DIR/router.log" 2>&1 &
pids+=("$!")
printf '%s\n' "P=http://127.0.0.1:$P_PORT D=http://127.0.0.1:$D_PORT Router=http://127.0.0.1:$ROUTER_PORT; Ctrl-C stops only this script's child processes."
wait
