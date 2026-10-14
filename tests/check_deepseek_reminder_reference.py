#!/usr/bin/env python3
"""Check the grpc_custom_roles.rs golden prompts against a vLLM checkout.

Run with a virtualenv Python; needs only the standard library, not model weights.
The reference checkout for this integration is PR #61446 at 8bbb174bc8a3cddf.
"""

import argparse
import runpy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_checkout", type=Path)
    args = parser.parse_args()
    for dialect in ("deepseek_v4", "deepseek_v41"):
        reference = args.vllm_checkout / "vllm" / "tokenizers" / f"{dialect}_encoding.py"
        encode = runpy.run_path(str(reference))["encode_messages"]
        for history in (False, True):
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "latest_reminder", "content": "Be brief 世界"},
            ]
            expected = (
                "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
                "<｜latest_reminder｜>Be brief 世界"
            )
            if history:
                messages.append({"role": "assistant", "content": "Hi there"})
                expected += "Hi there<｜end▁of▁sentence｜>"
            actual = encode(messages, thinking_mode="chat")
            if actual != expected:
                raise AssertionError((dialect, history, actual, expected))
            print(f"PASS {dialect} assistant_history={history}: {actual!r}")


if __name__ == "__main__":
    main()
