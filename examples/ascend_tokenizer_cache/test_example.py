# SPDX-License-Identifier: Apache-2.0
"""Offline checks for launch boundaries and ID comparison."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import launch_backend
import verify_tokens


class ExampleTests(unittest.TestCase):
    def test_http_response_rejects_invalid_ids_or_counts(self):
        bodies = (
            {"tokens": []},
            {"tokens": [True]},
            {"tokens": [-1]},
            {"tokens": [1], "count": 2},
            {"tokens": [1], "count": True},
        )
        for body in bodies:
            with self.subTest(body=body):
                stream = io.BytesIO(json.dumps(body).encode())
                with patch(
                    "verify_tokens.request.urlopen", return_value=stream
                ), self.assertRaises(ValueError):
                    verify_tokens.fetch_tokens("http://test", {}, 1)
        stream = io.BytesIO(b'{"tokens":[1,2],"count":2}')
        with patch("verify_tokens.request.urlopen", return_value=stream):
            tokens, elapsed = verify_tokens.fetch_tokens("http://test", {}, 1)
            self.assertEqual(tokens, [1, 2])
            self.assertGreaterEqual(elapsed, 0)

    def test_budget_off_and_on_preserve_other_config_and_arguments(self):
        for budget in (0, 4):
            args, command = launch_backend.parse_command(
                [
                    "--model",
                    "local-model",
                    "--cache-gb",
                    str(budget),
                    "--dry-run",
                    "--additional-config",
                    '{"kv_transfer_config":{"role":"prefill"}}',
                    "--",
                    "--tensor-parallel-size",
                    "2",
                ]
            )
            config = json.loads(command[command.index("--additional-config") + 1])
            self.assertEqual(
                config,
                {
                    "kv_transfer_config": {"role": "prefill"},
                    "tokenizer_cache_gb": 0,
                },
            )
            self.assertEqual(
                float(command[command.index("--tokenizer-cache-gb") + 1]), budget
            )
            self.assertEqual(command[-2:], ["--tensor-parallel-size", "2"])
            self.assertTrue(args.dry_run)

    def test_invalid_budget_and_duplicate_config_do_not_launch(self):
        for extra in (
            ["--cache-gb", "-1"],
            ["--cache-gb", "nan"],
            ["--cache-gb", "inf"],
            ["--additional-config", "[]"],
            ["--additional-config", '{"tokenizer_cache_gb":4}'],
            ["--", "--additional-config={}"],
            ["--", "--tokenizer-cache-gb=8"],
            ["--", "--tokenizer_cache_gb", "8"],
            ["--port", "0"],
            ["--", "--additional-config.tokenizer_cache_gb=8"],
        ):
            with self.subTest(extra=extra), patch("sys.stderr"), self.assertRaises(
                SystemExit
            ):
                launch_backend.parse_command(["--model", "local-model", *extra])

    def test_installed_contract_rejects_changed_or_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "renderer.py").write_bytes(b"known contract")
            spec = importlib.util.spec_from_file_location(
                "fake", root / "__init__.py", submodule_search_locations=[str(root)]
            )
            contract = {
                "package_files": {
                    "fake": {
                        "renderer.py": hashlib.sha256(b"known contract").hexdigest()
                    }
                }
            }
            with patch("launch_backend.importlib.util.find_spec", return_value=spec):
                launch_backend.inspect_backend(contract)
                (root / "renderer.py").write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    launch_backend.inspect_backend(contract)
                (root / "renderer.py").unlink()
                with self.assertRaises(ValueError):
                    launch_backend.inspect_backend(contract)

    def test_comparison_rejects_single_wrong_id_and_nondeterminism(self):
        with patch(
            "verify_tokens.fetch_tokens", side_effect=[([1, 2], 1), ([1, 3], 1)]
        ), self.assertRaises(ValueError):
            verify_tokens.compare("http://off", "http://on", "local", 1, 1)
        matching = [([1, 2], 1)] * 8
        with patch(
            "verify_tokens.fetch_tokens",
            side_effect=[*matching, ([2, 1], 1), ([2, 1], 1)],
        ), self.assertRaises(ValueError):
            verify_tokens.compare("http://off", "http://on", "local", 2, 1)
        with patch("verify_tokens.fetch_tokens", side_effect=matching):
            result = verify_tokens.compare("http://off", "http://on", "local", 1, 1)
            self.assertTrue(result["tokens_equal"])
            self.assertEqual(len(result["observations"]), 4)


if __name__ == "__main__":
    unittest.main()
