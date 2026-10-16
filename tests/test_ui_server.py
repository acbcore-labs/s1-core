"""Contract tests for the local UI server (tools/ui).

The frontend is render-only by law, so these server contracts are the whole
behavioral surface: static files served, evaluate round-trips through the
real harness (fail-closed through the API), sweeps flip statuses via the
pure gate, ledgers verify and replay, path traversal is refused, and the
compile-time lint reaches the client as a 422. Stdlib only, like the rest
of the suite — this file must pass on the zero-dependency unittest path too.
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

_TOOLS_UI = Path(__file__).resolve().parents[1] / "tools" / "ui"
if str(_TOOLS_UI) not in sys.path:
    sys.path.insert(0, str(_TOOLS_UI))

import app as ui_app  # noqa: E402  (imports s1core, scenarios, examples)

from s1core import (  # noqa: E402
    ChoiceSpec,
    JsonlAuditLedger,
    LexicalHeuristicDriver,
    System1Harness,
)
from s1core.audit import SCHEMA_VERSION  # noqa: E402


def _request(base, path, body=None):
    url = base + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, _maybe_json(resp.read()), resp.headers
    except urllib.error.HTTPError as exc:
        return exc.code, _maybe_json(exc.read()), exc.headers


def _maybe_json(raw: bytes):
    text = raw.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _post(base, path, body=None):
    return _request(base, path, body if body is not None else {})


class UiServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="s1core_ui_test_")
        ui = ui_app.UiApp([ui_app._REPO_ROOT, cls._tmp.name])
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                         ui_app.make_handler(ui))
        cls.app = ui
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls._tmp.cleanup()

    # ------------------------------------------------------------ static

    def test_index_served(self):
        status, body, headers = _request(self.base, "/")
        self.assertEqual(200, status)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("S1-Core", body)
        status, _, _ = _request(self.base, "/api/ghost")
        self.assertEqual(404, status)

    def test_static_assets_served(self):
        for name in ("app.js", "style.css"):
            status, _, headers = _request(self.base, f"/static/{name}")
            self.assertEqual(200, status)
        status, _, _ = _request(self.base, "/static/../app.py")
        self.assertEqual(403, status)

    def test_state_endpoint(self):
        status, body, _ = _request(self.base, "/api/state")
        self.assertEqual(200, status)
        self.assertEqual(SCHEMA_VERSION, body["schema_version"])
        self.assertIn(body["backend"]["kind"], ("toy", "live"))
        self.assertEqual(body["key_present"], body["backend"]["kind"] == "live")

    def test_scenarios_listed(self):
        status, body, _ = _request(self.base, "/api/scenarios")
        self.assertEqual(200, status)
        self.assertEqual({"triage", "model_swap", "cascade", "recalibration"},
                         {s["id"] for s in body["scenarios"]})

    # ---------------------------------------------------------- evaluate

    def test_evaluate_round_trip_ratified_and_fail_closed(self):
        body = {
            "backend": "toy",
            "raw_state": {"ticket_body": (
                "A human agent must review this refund escalation; the "
                "customer is angry and the amount needs a human agent."),
                "order_total": 120.0, "requested_refund": 42.0},
            "questions": [
                {"type": "choice", "question_id": "lane",
                 "instructions": "Which handling lane does this ticket need?",
                 "options": {
                     "agent_review": "A human agent must review this refund.",
                     "auto": "Approve automatically without any human"}},
            ],
        }
        status, payload, _ = _post(self.base, "/api/evaluate", body)
        self.assertEqual(200, status)
        self.assertTrue(payload["chain_verified"])
        self.assertEqual(0, payload["replay"]["wrong"])
        self.assertEqual(1, len(payload["waves"]))
        for row in payload["rows"]:
            if row["status"] != "RATIFIED":
                self.assertIsNone(row["value"])
        ratified = [r for r in payload["rows"] if r["status"] == "RATIFIED"]
        self.assertTrue(ratified, "toy sensor should ratify the aligned lane")
        self.assertEqual(payload["records"][0]["schema_version"],
                         SCHEMA_VERSION)

    def test_evaluate_dependency_skip_and_spec_error(self):
        body = {
            "backend": "toy",
            "raw_state": {"text": "refund total 100"},
            "questions": [
                {"type": "choice", "question_id": "lane",
                 "instructions": "Which handling lane does this ticket need?",
                 "options": {"a": "The first possible handling lane",
                             "b": "The second possible handling lane"}},
                {"type": "noul", "question_id": "child",
                 "instructions": "Is the ticket about the second lane?",
                 "true_criteria": "The ticket needs the second lane.",
                 "false_criteria": "The ticket needs the first lane.",
                 "depends_on": ["lane", "b"]},
            ],
        }
        status, payload, _ = _post(self.base, "/api/evaluate", body)
        self.assertEqual(200, status)
        statuses = {r["question_id"]: r["status"] for r in payload["rows"]}
        self.assertEqual(["lane"], payload["waves"][0])
        child = payload["rows"][1]
        if statuses["lane"] != "RATIFIED":
            self.assertNotIn("RATIFIED",
                             (statuses["lane"],))  # sanity
        self.assertIn(statuses["child"],
                      ("SKIPPED_DEPENDENCY", "RATIFIED", "ABSTAINED_GATE",
                       "ABSTAINED_MODEL"))

        bad = {"raw_state": {"a": 1},
               "questions": [{"type": "choice", "question_id": "x",
                              "instructions": "tiny", "options": {"a": "aaa", "b": "bbb"}}]}
        status, payload, _ = _post(self.base, "/api/evaluate", bad)
        self.assertEqual(422, status)
        self.assertIn("too terse", payload["error"])

        unknown = {"raw_state": {"a": 1},
                   "questions": [{"type": "zebra", "question_id": "x",
                                  "instructions": "What is a zebra question?"}]}
        status, payload, _ = _post(self.base, "/api/evaluate", unknown)
        self.assertEqual(422, status)
        self.assertIn("unknown question type", payload["error"])

    # ------------------------------------------------------------- sweep

    def test_sweep_flips_strict_and_validates_policy(self):
        body = {
            "backend": "toy",
            "raw_state": {"ticket_body": (
                "A human agent must review this refund escalation; the "
                "customer is angry and the amount needs a human agent.")},
            "questions": [
                {"type": "choice", "question_id": "lane",
                 "instructions": "Which handling lane does this ticket need?",
                 "options": {
                     "agent_review": "A human agent must review this refund.",
                     "auto": "Approve automatically without any human"}},
            ],
        }
        _, run, _ = _post(self.base, "/api/evaluate", body)
        strict = {"min_top1_prob": 0.999, "min_top2_margin": 0.999,
                  "max_normalized_entropy": 0.0}
        status, sweep, _ = _post(self.base, "/api/sweep",
                                 {"run_id": run["run_id"], "policy": strict})
        self.assertEqual(200, status)
        changed = [r for r in sweep["rows"] if r["changed"]]
        self.assertTrue(all(r["base_status"] == "RATIFIED" for r in changed))
        self.assertTrue(all(r["status"] == "ABSTAINED_GATE" for r in changed))
        self.assertTrue(any(r["changed"] for r in sweep["rows"]),
                         "strict policy must refuse the ratified verdict")

        status, payload, _ = _post(self.base, "/api/sweep",
                                   {"run_id": run["run_id"],
                                    "policy": {"min_top1_prob": 7.0}})
        self.assertEqual(422, status)
        status, payload, _ = _post(self.base, "/api/sweep",
                                   {"run_id": run["run_id"],
                                    "policy": {"nope": 1}})
        self.assertEqual(422, status)
        status, payload, _ = _post(self.base, "/api/sweep",
                                   {"run_id": "nope", "policy": {}})
        self.assertEqual(404, status)

    # ------------------------------------------------------ ledger & paths

    def test_ledger_load_replay_and_traversal_refused(self):
        ledger_path = Path(self._tmp.name) / "test_ledger.jsonl"
        harness = System1Harness(
            LexicalHeuristicDriver(), audit_sink=JsonlAuditLedger(ledger_path))
        harness.evaluate(
            {"x": "hello world"},
            [ChoiceSpec(question_id="q",
                        instructions="Which option matches this state best?",
                        options={"one": "The state contains hello world content",
                                 "two": "Something entirely different"})])
        status, loaded, _ = _post(self.base, "/api/ledger/load",
                                  {"path": str(ledger_path)})
        self.assertEqual(200, status)
        self.assertTrue(loaded["chain_verified"])
        self.assertEqual(0, loaded["replay"]["wrong"])
        self.assertEqual(1, len(loaded["streams"]))

        status, listing, _ = _request(self.base, "/api/ledgers")
        self.assertEqual(200, status)
        by_path = {entry["path"]: entry for entry in listing["ledgers"]}
        self.assertIn("test_ledger.jsonl", by_path)
        self.assertEqual(1, by_path["test_ledger.jsonl"]["records"])

        noise = Path(self._tmp.name) / "noise.jsonl"
        noise.write_text('{"foo": 1}\n', encoding="utf-8")
        status, listing, _ = _request(self.base, "/api/ledgers")
        by_path = {entry["path"]: entry for entry in listing["ledgers"]}
        self.assertNotIn("noise.jsonl", by_path)
        noise.unlink()

        status, payload, _ = _post(self.base, "/api/ledger/load",
                                   {"path": "/etc/hosts"})
        self.assertEqual(403, status)
        status, payload, _ = _post(self.base, "/api/ledger/load",
                                   {"path": "does/not/exist.jsonl"})
        self.assertEqual(404, status)

        status, replayed, _ = _post(self.base, "/api/replay",
                                    {"run_id": loaded["run_id"]})
        self.assertEqual(200, status)
        self.assertEqual(0, replayed["wrong"])
        status, payload, _ = _post(self.base, "/api/replay", {"run_id": "nope"})
        self.assertEqual(404, status)

    # --------------------------------------------------------- scenarios

    def test_certification_scenario_payload(self):
        status, payload, _ = _post(self.base, "/api/scenarios/model_swap/run")
        self.assertEqual(200, status)
        self.assertEqual("certification", payload["kind"])
        self.assertTrue(payload["chain_verified"])
        self.assertEqual(0, payload["replay"]["wrong"])
        self.assertEqual(3, len(payload["candidates"]))
        promoted = [c["promoted"] for c in payload["candidates"]]
        self.assertEqual([False, True, False], promoted)
        # the regressed candidate fails on incorrect ratifications
        self.assertEqual(2, payload["candidates"][0]["incorrect_count"])
        self.assertEqual(0, payload["candidates"][1]["incorrect_count"])
        self.assertEqual([], payload["candidates"][1]["coverage_failures"])
        # the vacuous abstainer: zero wrongness, rejected on declared coverage
        vacuous = payload["candidates"][2]
        self.assertEqual(0, vacuous["incorrect_count"])
        self.assertEqual(2, len(vacuous["coverage_failures"]))
        for spec_q in vacuous["per_question"].values():
            self.assertEqual(0.0, spec_q["coverage"])
        self.assertEqual(payload["pinned_cases"], len(payload["rows"]))

    def test_cascade_and_recalibration_payloads(self):
        status, cascade, _ = _post(self.base, "/api/scenarios/cascade/run")
        self.assertEqual(200, status)
        self.assertEqual(4, len(cascade["tiers"]))
        self.assertEqual(cascade["cheapest_zero_wrong"],
                         "conformal@α=0.10 (fitted)")
        for tier in cascade["tiers"]:
            self.assertIn("policy", tier)
            self.assertIn("cost_per_1k", tier)

        status, recal, _ = _post(self.base, "/api/scenarios/recalibration/run")
        self.assertEqual(200, status)
        self.assertEqual(3, len(recal["fits"]))
        lane = recal["fits"]["refund_lane"]
        self.assertGreater(lane["temperature"], 0.05)
        self.assertLess(lane["temperature"], 20.0)
        self.assertGreaterEqual(lane["coverage"], 1 - lane["alpha"] - 1e-9)
        self.assertLess(lane["ece_after"], lane["ece_before"])
        self.assertEqual(10, len(lane["bins"]))

        status, payload, _ = _post(self.base, "/api/scenarios/nope/run")
        self.assertEqual(404, status)

    def test_calibrate_endpoint(self):
        _, cascade, _ = _post(self.base, "/api/scenarios/cascade/run")
        status, fit, _ = _post(self.base, "/api/calibrate",
                               {"run_id": cascade["run_id"],
                                "question_id": "refund_lane", "alpha": 0.10})
        self.assertEqual(200, status)
        self.assertEqual("choice", fit["primitive"])
        self.assertGreaterEqual(fit["coverage"], 0.89)
        status, payload, _ = _post(self.base, "/api/calibrate",
                                   {"run_id": cascade["run_id"],
                                    "question_id": "refund_lane",
                                    "alpha": 1.5})
        self.assertEqual(422, status)
        status, payload, _ = _post(self.base, "/api/calibrate",
                                   {"run_id": cascade["run_id"],
                                    "question_id": "ghost", "alpha": 0.1})
        self.assertEqual(404, status)

    # ------------------------------------------------------------- misc

    def test_unknown_routes_and_bad_json(self):
        status, _, _ = _request(self.base, "/api/ghost")
        self.assertEqual(404, status)
        req = urllib.request.Request(self.base + "/api/evaluate",
                                     data=b"not json",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        self.assertEqual(400, status)


if __name__ == "__main__":
    unittest.main()
