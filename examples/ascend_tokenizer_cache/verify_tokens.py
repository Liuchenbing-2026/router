#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare complete /tokenize IDs from cache-OFF and cache-ON backends."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib import request
from urllib.parse import urlsplit

MAX_ROUNDS = 100
MAX_TIMEOUT_SECONDS = 300


def fetch_tokens(base_url, payload, timeout):
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("VLLM_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    started = time.perf_counter()
    with request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    elapsed = (time.perf_counter() - started) * 1000
    tokens = body.get("tokens") if isinstance(body, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokenize response must contain a nonempty tokens array")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("tokenize response contains invalid token IDs")
    if "count" in body and (
        type(body["count"]) is not int or body["count"] != len(tokens)
    ):
        raise ValueError("tokenize count differs from returned IDs")
    return tokens, elapsed


def compare(baseline, cached, model, rounds, timeout):
    prompts = [
        {"prompt": "Repeated history. " * 128, "add_special_tokens": False},
        {
            "prompt": "Unicode: \u4f60\u597d caf\u00e9 \U0001f680\n<|im_start|> literal marker",
            "add_special_tokens": False,
        },
        {
            "messages": [{"role": "user", "content": "Return the sum of 17 and 25."}],
            "add_generation_prompt": True,
        },
        {
            "messages": [
                {"role": "user", "content": "Explain a prefix."},
                {"role": "assistant", "content": "A shared beginning."},
                {"role": "user", "content": "Give one example."},
            ],
            "add_generation_prompt": True,
        },
    ]
    rows = []
    references = {}
    for iteration in range(rounds):
        for index, case in enumerate(prompts):
            payload = dict(case, model=model)
            off, off_ms = fetch_tokens(baseline, payload, timeout)
            on, on_ms = fetch_tokens(cached, payload, timeout)
            if off != on or (index in references and references[index] != off):
                raise ValueError(f"token IDs differ at round {iteration} case {index}")
            references[index] = off
            rows.append(
                {
                    "round": iteration,
                    "case": index,
                    "tokens": len(off),
                    "ids_sha256": hashlib.sha256(json.dumps(off).encode()).hexdigest(),
                    "off_ms": off_ms,
                    "on_ms": on_ms,
                }
            )
    return {
        "tokens_equal": True,
        "cases": len(prompts),
        "rounds": rounds,
        "scope": "synthetic /tokenize conformance; no inference or KV-hit claim",
        "observations": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cached", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= MAX_ROUNDS:
        parser.error("--rounds must be between 1 and 100")
    if args.baseline.rstrip("/") == args.cached.rstrip("/"):
        parser.error("OFF and ON endpoints must be distinct")
    if not 0 < args.timeout <= MAX_TIMEOUT_SECONDS:
        parser.error("--timeout must be between 0 and 300 seconds")
    try:
        result = compare(
            args.baseline, args.cached, args.model, args.rounds, args.timeout
        )
    except (ValueError, OSError) as exc:
        print(
            json.dumps({"tokens_equal": False, "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        return 1
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
