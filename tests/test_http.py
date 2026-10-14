"""HTTP backend round-trip tests against a local reference server."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from s1core import ChoiceSpec, NoulSpec, System1Harness
from s1core.backends import BackendError, SystemOneHTTPDriver


class FakeSystemOneHandler(BaseHTTPRequestHandler):
    captured: list[dict] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        FakeSystemOneHandler.captured.append(body)
        if body.get("model") == "boom":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "internal"}')
            return
        decisions = {}
        for question in body["questions"]:
            if question["type"] == "noul":
                decisions[question["question_id"]] = {"true": 0.92, "false": 0.08}
            else:
                decisions[question["question_id"]] = {
                    key: 0.9 if key == "fast" else 0.1 / max(len(question["criteria"]) - 1, 1)
                    for key in question["criteria"]
                }
        response = json.dumps({"decisions": decisions}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):  # silence test output
        pass


class SystemOneHTTPDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSystemOneHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_round_trip_through_harness(self):
        FakeSystemOneHandler.captured = []
        driver = SystemOneHTTPDriver(self.base_url, model_id="fake-jev-1", api_key="k-test")
        harness = System1Harness(driver)
        lane = ChoiceSpec(
            question_id="lane",
            instructions="Which handling lane does this ticket belong in?",
            options={
                "fast": "Automated fast-track handling with no human contact needed.",
                "slow": "Human agent reviews everything before any action is taken.",
            },
        )
        callback = NoulSpec(
            question_id="callback",
            instructions="Should the fast lane trigger a courtesy callback to the customer?",
            true_criteria="The customer explicitly asked to be called back by phone.",
            false_criteria="The customer did not ask for any callback contact.",
            depends_on=("lane", "fast"),
        )
        verdicts = harness.evaluate(
            {"ticket": "callback requested"}, [lane, callback],
            reference_time_iso="2026-10-14T12:00:00+00:00",
        )
        self.assertEqual(verdicts["lane"].status.name, "RATIFIED")
        self.assertEqual(verdicts["lane"].value, "fast")
        self.assertEqual(verdicts["callback"].status.name, "RATIFIED")
        self.assertIs(verdicts["callback"].value, True)

        # two waves -> two HTTP calls; each carries the canonical envelope
        # (raw state + deterministic facts) and the payload schema
        self.assertEqual(len(FakeSystemOneHandler.captured), 2)
        first = FakeSystemOneHandler.captured[0]
        self.assertEqual(first["model"], "fake-jev-1")
        self.assertEqual(
            first["state"],
            {"raw_state": {"ticket": "callback requested"}, "@deterministic_facts": {}},
        )
        self.assertEqual(first["questions"][0]["type"], "choice")
        self.assertIn("__abstain_insufficient_evidence__", first["questions"][0]["criteria"])

    def test_http_error_maps_to_backend_error(self):
        driver = SystemOneHTTPDriver(self.base_url, model_id="boom")
        harness = System1Harness(driver)
        spec = ChoiceSpec(
            question_id="lane",
            instructions="Which handling lane does this ticket belong in?",
            options={
                "fast": "Automated fast-track handling with no human contact needed.",
                "slow": "Human agent reviews everything before any action is taken.",
            },
        )
        with self.assertRaises(BackendError):
            harness.evaluate({}, [spec])


if __name__ == "__main__":
    unittest.main()
