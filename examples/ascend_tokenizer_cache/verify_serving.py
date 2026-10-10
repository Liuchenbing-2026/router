"""Compare native render and generation IDs on pinned OFF/ON backends."""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

MIN_ROUNDS = 2


def post(base, path, body):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("BACKEND_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["BACKEND_API_KEY"]
    request = urllib.request.Request(
        base.rstrip("/") + path,
        json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def token_ids(value):
    if not isinstance(value, list) or not value:
        raise ValueError("missing/non-list/empty token_ids")
    if any(type(x) is not int or x < 0 for x in value):
        raise ValueError("invalid token ID")
    return value


def cases(model):
    messages = [
        {"role": "system", "content": "Answer briefly. Preserve code and Unicode."},
        {"role": "user", "content": "Explain `a + b` in Chinese. 你好 👋"},
    ]
    yield "unicode", {"model": model, "messages": messages}
    yield "salt-a", {"model": model, "messages": messages, "cache_salt": "tenant-a"}
    yield "salt-b", {"model": model, "messages": messages, "cache_salt": "tenant-b"}
    yield "growing", {
        "model": model,
        "messages": [
            *messages,
            {"role": "assistant", "content": "Addition combines two values."},
            {"role": "user", "content": "Continue with Python examples."},
        ],
    }
    yield "tools", {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "none",
    }


def verify(baseline, cached, model, rounds=3, serve=False):
    if rounds < MIN_ROUNDS:
        raise ValueError("rounds must be >= 2 to check repeated requests")
    results = []
    for name, source in cases(model):
        payload = dict(source, max_tokens=1, temperature=0, stream=False)
        expected = None
        for iteration in range(rounds):
            for label, base in [("off", baseline), ("on", cached)]:
                started = time.perf_counter()
                rendered = post(base, "/v1/chat/completions/render", payload)
                ids = token_ids(rendered.get("token_ids"))
                if expected is None:
                    expected = ids
                if ids != expected:
                    raise ValueError(
                        f"{name}: full render IDs differ ({label}/{iteration})"
                    )
                record = {
                    "case": name,
                    "mode": label,
                    "iteration": iteration,
                    "prompt_tokens": len(ids),
                    "render_ms": (time.perf_counter() - started) * 1000,
                }
                if serve:
                    output = post(
                        base,
                        "/v1/chat/completions",
                        dict(payload, return_token_ids=True),
                    )
                    if token_ids(output.get("prompt_token_ids")) != ids:
                        raise ValueError(f"{name}: serving/render IDs differ ({label})")
                    if output.get("usage", {}).get("prompt_tokens") != len(ids):
                        raise ValueError(
                            f"{name}: serving token count differs ({label})"
                        )
                    if not output.get("choices"):
                        raise ValueError(f"{name}: missing serving choices ({label})")
                    encoded = json.dumps(output["choices"], sort_keys=True).encode()
                    record["output_sha256"] = hashlib.sha256(encoded).hexdigest()
                results.append(record)
    return {
        "scope": "native_render_and_serving" if serve else "native_render_only",
        "generation_quality": "not_evaluated",
        "performance": "diagnostic_only",
        "full_prompt_ids_equal": True,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cached", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.baseline, args.cached, args.model, args.rounds, args.serve)
    except Exception as error:
        args.output.write_text(json.dumps({"passed": False, "error": str(error)}))
        raise
    args.output.write_text(json.dumps(dict(result, passed=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
