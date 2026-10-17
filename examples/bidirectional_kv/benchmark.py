#!/usr/bin/env python3
"""Measure one real-engine arm, or strictly compare two saved arms. No synthetic gains."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

def metric_sum(text, name):
    values = []
    for line in text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            values.append(float(line.split()[-1]))
    return sum(values) if values else None


def _measure(args, directory, records):
    import httpx

    key = os.environ.get("OPENAI_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    with httpx.Client(timeout=300, headers=headers) as client:
        before = client.get(args.prefill.rstrip("/") + "/metrics")
        before.raise_for_status()
        (directory / "prefill-before.prom").write_text(before.text)
        for length in args.lengths:
            for conversation in range(args.conversations):
                # Same deterministic prompt for a given pair across OFF and ON.
                text = (f"Experiment {args.pair} context {length} case {conversation}. "
                        + "A careful engineer checks the input and records the result. "
                        * (length + 16))
                tokenized = client.post(args.prefill.rstrip("/") + "/tokenize",
                                        json={"model": args.model, "prompt": text})
                tokenized.raise_for_status()
                prompt = tokenized.json()["tokens"][:length]
                if len(prompt) != length:
                    raise RuntimeError("Tokenizer returned too few tokens")
                for turn in range(args.turns):
                    body = {"model": args.model, "prompt": prompt,
                            "max_tokens": args.output_tokens, "ignore_eos": True,
                            "temperature": 0, "seed": 0, "stream": True,
                            "return_token_ids": True}
                    if args.leg == "on":
                        body["conversation_id"] = f"pair-{args.pair}-{length}-{conversation}"
                    started = time.perf_counter()
                    first_token = None
                    output = []
                    done = False
                    finished = False
                    record = {"pair": args.pair, "length": length,
                              "conversation": conversation, "turn": turn,
                              "input_tokens": len(prompt),
                              "input_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                              "output_token_ids": output, "success": False}
                    records.append(record)
                    with client.stream("POST", args.url.rstrip("/") + "/v1/completions",
                                       json=body) as response:
                        response.raise_for_status()
                        receipt = response.headers.get("x-router-kv-reuse")
                        for line in response.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                done = True
                                continue
                            item = json.loads(data)
                            if "error" in item:
                                raise RuntimeError(str(item["error"]))
                            for choice in item.get("choices", []):
                                delta = choice.get("token_ids") or []
                                if delta and first_token is None:
                                    first_token = time.perf_counter() - started
                                output.extend(delta)
                                finished |= choice.get("finish_reason") in ("stop", "length")
                    if not done or not finished or first_token is None or len(output) != args.output_tokens:
                        raise RuntimeError("Incomplete stream or missing raw output token IDs")
                    record.update({"ttft_seconds": first_token,
                                   "latency_seconds": time.perf_counter() - started,
                                   "reuse_receipt": receipt, "success": True})
                    with (directory / "requests.jsonl").open("a") as log:
                        log.write(json.dumps(record) + "\n")
                    # Preserve the full raw generated history, including thinking/special tokens.
                    prompt = prompt + output + [prompt[-1]] * args.append_tokens
                    time.sleep(args.tool_gap)
        time.sleep(args.metrics_settle)
        after = client.get(args.prefill.rstrip("/") + "/metrics")
        after.raise_for_status()
        (directory / "prefill-after.prom").write_text(after.text)
    name = f"vllm:{args.transport}_bytes_transferred_sum"
    a, b = metric_sum(before.text, name), metric_sum(after.text, name)
    manifest = {"leg": args.leg, "pair": args.pair, "arguments": vars(args), "complete": True,
                "client_versions": {"httpx": httpx.__version__},
                "prefill_read_bytes_delta": None if a is None or b is None else b - a,
                "requests": records,
                "scope": "Real engines only; completed later turns are the TTFT comparison unit."}
    (directory / "results.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"requests": len(records), "prefill_read_bytes_delta": manifest["prefill_read_bytes_delta"]}))


def measure(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        _measure(args, directory, records)
    except BaseException as error:
        (directory / "results.json").write_text(json.dumps({
            "leg": args.leg, "pair": args.pair, "arguments": vars(args),
            "complete": False, "error": repr(error), "requests": records,
        }, indent=2))
        raise


def compare(args):
    off = json.loads(Path(args.off).read_text())
    on = json.loads(Path(args.on).read_text())
    if not off.get("complete") or not on.get("complete"):
        raise RuntimeError("An arm failed or was interrupted; preserve it and rerun")
    if off["leg"] != "off" or on["leg"] != "on" or off["pair"] != on["pair"]:
        raise RuntimeError("Leg or pair mismatch")
    if off["arguments"].get("transport", "nixl") != on["arguments"].get("transport", "nixl"):
        raise RuntimeError("Transport mismatch")
    for option in ("lengths", "conversations", "turns", "output_tokens", "append_tokens", "tool_gap", "model"):
        if off["arguments"][option] != on["arguments"][option]:
            raise RuntimeError("Workload mismatch: " + option)
    if len(off["requests"]) != len(on["requests"]):
        raise RuntimeError("Different completed-request sets")
    later = []
    for a, b in zip(off["requests"], on["requests"]):
        if not a.get("success") or not b.get("success"):
            raise RuntimeError("A request failed or was interrupted")
        for field in ("pair", "length", "conversation", "turn", "input_tokens", "input_sha256", "output_token_ids"):
            if a[field] != b[field]:
                raise RuntimeError("Quality/workload mismatch: " + field)
        if a["turn"]:
            if b["reuse_receipt"] != "candidate":
                raise RuntimeError("ON did not exercise the remote candidate path")
            later.append((a, b))
    if not later:
        raise RuntimeError("No later-turn samples")
    off_bytes, on_bytes = off["prefill_read_bytes_delta"], on["prefill_read_bytes_delta"]
    if off_bytes is None or on_bytes is None or not math.isfinite(on_bytes) or off_bytes != 0 or on_bytes <= 0:
        raise RuntimeError("Missing isolated P-side proof of physical D-to-P transfer")
    results = []
    for length in off["arguments"]["lengths"]:
        group = [(a, b) for a, b in later if a["length"] == length]
        baseline = statistics.median(a["ttft_seconds"] for a, _ in group)
        optimized = statistics.median(b["ttft_seconds"] for _, b in group)
        results.append({"length": length, "later_turns": len(group),
                        "off_ttft_median_s": baseline, "on_ttft_median_s": optimized,
                        "speedup": baseline / optimized})
    passed = all(r["speedup"] >= args.minimum_speedup for r in results)
    report = {"passed": passed, "minimum_speedup": args.minimum_speedup,
              "results": results, "token_ids_equal": True,
              "prefill_read_bytes_delta_on": on_bytes,
              "scope": "One paired run; repeat AB/BA/AB with fresh engines before a general claim."}
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("TTFT gain gate failed; keep the measurements and investigate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure_parser = sub.add_parser("measure")
    measure_parser.add_argument("--url", required=True)
    measure_parser.add_argument("--prefill", required=True)
    measure_parser.add_argument("--model", required=True)
    measure_parser.add_argument("--transport", choices=["nixl", "mooncake"], default="nixl")
    measure_parser.add_argument("--leg", choices=["off", "on"], required=True)
    measure_parser.add_argument("--pair", type=int, required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    measure_parser.add_argument("--conversations", type=int, default=2)
    measure_parser.add_argument("--turns", type=int, default=3)
    measure_parser.add_argument("--output-tokens", type=int, default=32)
    measure_parser.add_argument("--append-tokens", type=int, default=64)
    measure_parser.add_argument("--tool-gap", type=float, default=0.1)
    measure_parser.add_argument("--metrics-settle", type=float, default=10)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--off", required=True)
    compare_parser.add_argument("--on", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--minimum-speedup", type=float, default=1.05)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args)
    else:
        compare(args)
