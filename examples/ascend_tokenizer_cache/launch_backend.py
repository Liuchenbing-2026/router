#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the pinned Ascend backend with an explicit tokenizer-cache budget."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

MAX_PORT = 65535


def inspect_backend(contract):
    for package, files in contract["package_files"].items():
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("backend package is unavailable: " + package)
        root = Path(next(iter(spec.submodule_search_locations)))
        for relative, expected in files.items():
            path = root / relative
            if not path.is_file():
                raise ValueError(
                    "backend contract file is unavailable: " + package + "/" + relative
                )
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "backend contract differs: " + package + "/" + relative
                )


def parse_command(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-gb", type=float, default=0)
    parser.add_argument("--additional-config", default="{}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cache_gb) or args.cache_gb < 0:
        parser.error("--cache-gb must be finite and non-negative")
    if not 1 <= args.port <= MAX_PORT:
        parser.error("--port must be between 1 and 65535")
    try:
        config = json.loads(args.additional_config)
    except ValueError:
        parser.error("--additional-config must be a JSON object")
    if not isinstance(config, dict):
        parser.error("--additional-config must be a JSON object")
    if "tokenizer_cache_gb" in config:
        parser.error("set the cache budget through --cache-gb")
    extra = args.server_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    reserved = (
        "--additional-config",
        "--additional_config",
        "--model",
        "--host",
        "--port",
        "--api-key",
        "--api_key",
    )
    if any(word.split("=", 1)[0].split(".", 1)[0] in reserved for word in extra):
        parser.error(
            "server args must not override launcher fields; use VLLM_API_KEY for auth"
        )
    if extra and not extra[0].startswith("--"):
        parser.error("server args must start with an option")
    config["tokenizer_cache_gb"] = args.cache_gb
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--additional-config",
        json.dumps(config, separators=(",", ":")),
    ]
    return args, command + extra


def main(argv=None):
    args, command = parse_command(sys.argv[1:] if argv is None else argv)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "argv": command}))
        return 0
    contract = json.loads(Path(__file__).with_name("source_contract.json").read_text())
    try:
        inspect_backend(contract)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
