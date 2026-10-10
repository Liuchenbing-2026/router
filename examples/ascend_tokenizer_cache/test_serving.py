"""Exercise the acceptance harness over HTTP; no model inference is performed."""

import json
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from verify_serving import verify

SECOND_CALL = 2


class ServingHarnessTests(unittest.TestCase):
    def setUp(self):
        self.fault = None
        self.paths = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                owner.paths.append(self.path)
                if owner.fault == "http":
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                ids = [1, len(json.dumps(payload["messages"])), 3]
                if owner.fault == "render" and len(owner.paths) == SECOND_CALL:
                    ids = [1, 7, 3]
                result = {"token_ids": ids}
                if not self.path.endswith("/render"):
                    result = {
                        "prompt_token_ids": ids,
                        "usage": {"prompt_tokens": len(ids)},
                        "choices": [{"message": {"content": "fixture"}}],
                    }
                    if owner.fault == "generation":
                        result["prompt_token_ids"] = [9, 9, 9]
                    if owner.fault == "usage":
                        result["usage"]["prompt_tokens"] = 4
                if owner.fault == "malformed":
                    result = {"token_ids": [True, 2, 3]}
                self.wfile.write(json.dumps(result).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_transport_reaches_render_and_serving(self):
        result = verify(self.url, self.url, "fixture", rounds=2, serve=True)
        self.assertEqual(len(result["results"]), 20)
        self.assertEqual(result["generation_quality"], "not_evaluated")
        self.assertIn("/v1/chat/completions", self.paths)

    def test_full_ids_mismatch_with_equal_counts_fails(self):
        self.fault = "render"
        with self.assertRaisesRegex(ValueError, "full render IDs differ"):
            verify(self.url, self.url, "fixture", rounds=2)

    def test_serving_mismatch_fails(self):
        self.fault = "generation"
        with self.assertRaisesRegex(ValueError, "serving/render IDs differ"):
            verify(self.url, self.url, "fixture", rounds=2, serve=True)

    def test_inconsistent_usage_fails(self):
        self.fault = "usage"
        with self.assertRaisesRegex(ValueError, "token count differs"):
            verify(self.url, self.url, "fixture", rounds=2, serve=True)

    def test_http_failure_fails(self):
        self.fault = "http"
        with self.assertRaises(urllib.error.HTTPError):
            verify(self.url, self.url, "fixture", rounds=2)

    def test_invalid_token_type_fails(self):
        self.fault = "malformed"
        with self.assertRaisesRegex(ValueError, "invalid token ID"):
            verify(self.url, self.url, "fixture", rounds=2)


if __name__ == "__main__":
    unittest.main()
