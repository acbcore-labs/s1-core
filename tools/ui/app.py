"""S1-Core local UI — a stdlib HTTP server putting a render-only frontend in
front of the harness.

Laws this server enforces:

- The gate never lives in the browser. Every verdict, sweep, fit, drift
  figure, and chain check is computed here by s1core; the client only draws.
  A malicious client could draw nonsense; it cannot extract a value the
  harness refused.
- Local only: the server binds 127.0.0.1. Ledger loading is restricted to
  the repository root plus directories given via ``--allow`` (resolved paths
  only — ``..`` traversal is rejected after resolution, not string-checked).
- The API key is read from the environment/.env by s1core.config and is
  never echoed back to the page — only the model id and whether a live
  backend exists.

Run:  uv run python tools/ui/app.py [--port 8765] [--allow DIR] [--no-open]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

_TOOLS_UI = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_UI.parents[1]
for _p in (_TOOLS_UI, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import s1core  # noqa: E402
import scenarios  # noqa: E402
from scenarios import ApiError, SCENARIOS, build_run, check_replay  # noqa: E402

from s1core import (  # noqa: E402
    ChoiceSpec,
    EpistemicStatus,
    GateInputs,
    GatePolicy,
    InMemoryAuditLedger,
    NoulSpec,
    ScoreSpec,
    SpecError,
    StateError,
    System1Harness,
    compile_waves,
    evaluate,
)
from s1core.audit import SCHEMA_VERSION  # noqa: E402

STATIC = _TOOLS_UI / "static"
MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_LEDGER_RECORDS = 5000
MAX_LEDGER_SCAN = 500

# never walked when listing available ledgers
_LEDGER_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".zcode",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
})

# GatePolicy field -> coercion (floats validated later by gates.evaluate)
_BOOL_POLICY_FIELDS = frozenset({"debiased_permutation"})
_FLOAT_POLICY_FIELDS = frozenset({
    "min_top1_prob", "min_top2_margin", "max_normalized_entropy",
    "max_score_mode_aad", "temperature", "conformal_tau",
    "max_permutation_divergence",
})
_KNOWN_POLICY_FIELDS = _BOOL_POLICY_FIELDS | _FLOAT_POLICY_FIELDS

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def coerce_policy(overrides: dict) -> dict:
    """Validate a partial GatePolicy override from the client; unknown fields
    are a 422, not a silent ignore — a dial that silently does nothing is a
    lie, and lying dials are how gates get 'tuned'."""
    kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in _KNOWN_POLICY_FIELDS:
            raise ApiError(422, f"unknown GatePolicy field: {key!r}")
        if key in _BOOL_POLICY_FIELDS:
            kwargs[key] = bool(value)
        elif value is None:
            kwargs[key] = None
        else:
            try:
                kwargs[key] = float(value)
            except (TypeError, ValueError):
                raise ApiError(422, f"policy field {key!r} must be a number")
    return kwargs


def spec_from_json(data: Any) -> ChoiceSpec | ScoreSpec | NoulSpec:
    """Parse one client-side question declaration into a spec. Everything the
    compile-time lint will catch is reported, never swallowed."""
    if not isinstance(data, dict):
        raise ApiError(422, "each question must be a JSON object")
    question_id = data.get("question_id")
    if not isinstance(question_id, str) or not question_id:
        raise ApiError(422, "each question needs a non-empty question_id")
    policy_kwargs = coerce_policy(data.get("policy") or {})
    policy = GatePolicy(**policy_kwargs)
    deps = data.get("depends_on")
    if deps is not None:
        if (isinstance(deps, list) and len(deps) == 2
                and all(isinstance(x, str) for x in deps)):
            deps = (deps[0], deps[1])
        else:
            try:
                deps = [(parent, required) for parent, required in deps]
            except (TypeError, ValueError):
                raise ApiError(422,
                               f"[{question_id}] depends_on must be "
                               f"[parent, value] or a list of such pairs")
    common = {"question_id": question_id, "policy": policy, "depends_on": deps,
              "instructions": str(data.get("instructions", ""))}
    qtype = data.get("type")
    if qtype == "choice":
        options = data.get("options")
        if not isinstance(options, dict) or not options:
            raise ApiError(422, f"[{question_id}] choice options must be a "
                                f"non-empty object of key -> criteria prose")
        return ChoiceSpec(options={str(k): str(v) for k, v in options.items()},
                          inject_escape_hatch=bool(
                              data.get("inject_escape_hatch", True)),
                          **common)
    if qtype == "score":
        levels = data.get("rubric_levels")
        if not isinstance(levels, dict) or not levels:
            raise ApiError(422, f"[{question_id}] rubric_levels must be a "
                                f"non-empty object of level -> criteria prose")
        try:
            rubric = {int(k): str(v) for k, v in levels.items()}
        except (TypeError, ValueError):
            raise ApiError(422, f"[{question_id}] rubric level keys must be "
                                f"integers")
        return ScoreSpec(rubric_levels=rubric, **common)
    if qtype == "noul":
        try:
            true_threshold = float(data.get("true_threshold", 0.75))
            false_threshold = float(data.get("false_threshold", 0.35))
        except (TypeError, ValueError):
            raise ApiError(422, f"[{question_id}] noul thresholds must be "
                                f"numbers")
        return NoulSpec(true_criteria=str(data.get("true_criteria", "")),
                        false_criteria=str(data.get("false_criteria", "")),
                        true_threshold=true_threshold,
                        false_threshold=false_threshold,
                        **common)
    raise ApiError(422, f"[{question_id}] unknown question type: {qtype!r} "
                        f"(expected 'choice', 'score', or 'noul')")


def _walk_jsonl(root: Path) -> Iterator[Path]:
    """Yield *.jsonl files under root, pruning dependency/tool directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _LEDGER_SKIP_DIRS]
        for name in filenames:
            if name.endswith(".jsonl"):
                yield Path(dirpath) / name


def _peek_ledger(path: Path) -> dict | None:
    """Classify one file: a ledger-like .jsonl returns its record count and
    mtime; anything else returns None. Only the first record is parsed."""
    try:
        stat = path.stat()
        with path.open(encoding="utf-8") as fh:
            first = fh.readline().strip()
            if not first:
                return None
            try:
                record = json.loads(first)
            except json.JSONDecodeError:
                return None
            if not isinstance(record, dict) or "record_digest" not in record:
                return None
            records = 1 + sum(1 for line in fh if line.strip())
    except OSError:
        return None
    return {"records": records, "size_bytes": stat.st_size,
            "mtime": stat.st_mtime}


def apply_postprocessing(verdict, record: dict, policy: GatePolicy):
    """Reapply the recorded post-gate control events (mirror of
    audit.replay_verdict's law) so a sweep stays faithful to what the live
    harness did after the pure gate."""
    events = record.get("postprocessing") or {}
    divergence = events.get("permutation_divergence")
    if divergence is not None:
        if not isinstance(divergence, (int, float)) or not 0.0 <= divergence <= 1.0:
            raise ApiError(422, "recorded permutation divergence is invalid")
        if divergence > policy.max_permutation_divergence:
            verdict = dataclasses.replace(verdict,
                                          status=EpistemicStatus.ABSTAINED_GATE,
                                          value=None)
    if events.get("loop_guard_tripped") and verdict.status in (
            EpistemicStatus.ABSTAINED_MODEL, EpistemicStatus.ABSTAINED_GATE):
        verdict = dataclasses.replace(
            verdict, status=EpistemicStatus.LOOP_GUARD_TRIPPED, value=None)
    return verdict


class UiApp:
    """Run registry + path policy. One instance per server (or test)."""

    def __init__(self, allowed_roots):
        self.allowed_roots = [Path(r).resolve() for r in allowed_roots]
        self.runs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def store(self, payload: dict) -> dict:
        with self._lock:
            self._counter += 1
            payload["run_id"] = f"r{self._counter}"
            self.runs[payload["run_id"]] = payload
            return payload

    def get_run(self, run_id: str) -> dict:
        run = self.runs.get(str(run_id))
        if run is None:
            raise ApiError(404, f"unknown run: {run_id}")
        return run

    def resolve_ledger_path(self, raw: str) -> Path:
        if not raw:
            raise ApiError(422, "path is required")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.allowed_roots[0] / candidate
        resolved = candidate.resolve()
        for root in self.allowed_roots:
            if resolved == root or root in resolved.parents:
                break
        else:
            raise ApiError(403, f"path outside the allowed roots: {raw}")
        if not resolved.is_file():
            raise ApiError(404, f"no such file: {resolved}")
        return resolved

    def list_ledgers(self) -> list[dict]:
        """Enumerate plausible audit ledgers under the allowed roots.

        A file qualifies only when its first line parses as a JSON object
        carrying ``record_digest`` — the dropdown lists real ledgers, not
        any stray .jsonl. Newest first; paths shown relative to the repo
        root when possible.
        """
        found = []
        scanned = 0
        for root in self.allowed_roots:
            if not root.is_dir():
                continue
            for path in _walk_jsonl(root):
                scanned += 1
                if scanned > MAX_LEDGER_SCAN:
                    break
                info = _peek_ledger(path)
                if info is None:
                    continue
                resolved = path.resolve()
                display = str(resolved)
                for root in self.allowed_roots:
                    try:
                        display = str(resolved.relative_to(root))
                        break
                    except ValueError:
                        continue
                found.append({"path": display, **info})
        found.sort(key=lambda entry: (-entry["mtime"], entry["path"]))
        for entry in found:
            del entry["mtime"]
        return found

    # ------------------------------------------------------------- actions

    def do_evaluate(self, body: dict) -> dict:
        raw_state = body.get("raw_state")
        if not isinstance(raw_state, dict):
            raise ApiError(422, "raw_state must be a JSON object")
        question_specs = [spec_from_json(q) for q in body.get("questions", [])]
        if not question_specs:
            raise ApiError(422, "questions must be a non-empty list")
        driver, backend_kind = scenarios_backend(body.get("backend", "auto"))
        waves = compile_waves(question_specs)
        ledger = InMemoryAuditLedger()
        harness = System1Harness(driver, audit_sink=ledger)
        reference = body.get("reference_time_iso") or None
        try:
            verdicts = harness.evaluate(
                raw_state, question_specs, reference_time_iso=reference,
                stream_id=str(body.get("stream_id") or "ui-composer"))
        except (SpecError, StateError, TypeError, ValueError) as exc:
            raise ApiError(422, str(exc))
        records = [record.to_jsonable() for record in ledger.records]
        return build_run(
            kind="evaluate", driver=driver, backend_kind=backend_kind,
            records=records, labels={},
            reference_time_iso=reference,
            verdicts={qid: v.to_jsonable() for qid, v in verdicts.items()},
            waves=[[spec.question_id for spec in wave] for wave in waves],
            questions_meta=[
                {"question_id": spec.question_id,
                 "depends_on": [list(cond) for cond in spec.dependencies()]}
                for spec in question_specs
            ],
        )

    def do_ledger_load(self, body: dict) -> dict:
        path = self.resolve_ledger_path(str(body.get("path", "")))
        records = []
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if len(records) >= MAX_LEDGER_RECORDS:
                raise ApiError(422, f"ledger exceeds {MAX_LEDGER_RECORDS} "
                                    f"records; refusing to load")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ApiError(422, f"{path.name}:{lineno}: invalid JSON "
                                    f"({exc.msg})")
        if not records:
            raise ApiError(422, "ledger contains no records")
        streams = sorted({str(r.get("stream_id", "default")) for r in records})
        model_ids = sorted({str(r.get("model_id", "")) for r in records})
        return build_run(
            kind="ledger",
            driver=_DriverInfo(sorted(model_ids)[0] if model_ids else "?"),
            backend_kind="ledger", records=records, labels={},
            path=str(path), streams=streams, model_ids=model_ids,
        )

    def do_sweep(self, body: dict) -> dict:
        run = self.get_run(body.get("run_id", ""))
        kwargs = coerce_policy(body.get("policy") or {})
        rows = []
        for idx, record in enumerate(run["records"]):
            base_status = (record.get("verdict") or {}).get("status", "?")
            compiled = (record.get("compiled_question") or {})
            raw = record.get("raw_probabilities") or {}
            if not raw:
                rows.append({"record_idx": idx,
                             "question_id": compiled.get("question_id", "?"),
                             "base_status": base_status,
                             "status": base_status, "value": None,
                             "changed": False})
                continue
            gate_inputs = GateInputs.from_jsonable(record["gate_inputs"])
            policy = dataclasses.replace(gate_inputs.policy, **kwargs)
            try:
                verdict = evaluate(dataclasses.replace(gate_inputs,
                                                       policy=policy),
                                   raw, "sweep")
            except ValueError as exc:
                raise ApiError(422, f"invalid sweep policy: {exc}")
            verdict = apply_postprocessing(verdict, record, policy)
            rows.append({
                "record_idx": idx,
                "question_id": compiled.get("question_id", "?"),
                "base_status": base_status,
                "status": verdict.status.value,
                "value": verdict.value,
                "changed": verdict.status.value != base_status,
            })
        return {"run_id": run["run_id"], "rows": rows}

    def do_calibrate(self, body: dict) -> dict:
        run = self.get_run(body.get("run_id", ""))
        question_id = body.get("question_id")
        if not question_id:
            raise ApiError(422, "question_id is required")
        try:
            alpha = float(body.get("alpha", 0.10))
        except (TypeError, ValueError):
            raise ApiError(422, "alpha must be a number")
        if not 0.0 < alpha < 1.0:
            raise ApiError(422, "alpha must be strictly between 0 and 1")
        primitive = None
        for record in run["records"]:
            if (record.get("compiled_question") or {}).get("question_id") == question_id:
                primitive = (record.get("gate_inputs") or {}).get("primitive")
                break
        if primitive is None:
            raise ApiError(404, f"question {question_id!r} not in this run")
        samples = scenarios.collect_samples(
            run["records"],
            {int(k): v for k, v in (run.get("labels") or {}).items()},
            question_id)
        return scenarios.fit_samples(samples, primitive, alpha)

    def do_replay(self, body: dict) -> dict:
        run = self.get_run(body.get("run_id", ""))
        return {"run_id": run["run_id"], **check_replay(run["records"])}


class _DriverInfo:
    """Stand-in for ledger runs: only a model_id is needed for the payload."""

    def __init__(self, model_id: str):
        self.model_id = model_id


_backend_cache: dict[str, tuple[Any, str]] = {}


def scenarios_backend(preference: str) -> tuple[Any, str]:
    if preference not in _backend_cache:
        _backend_cache[preference] = scenarios.resolve_backend(preference)
    return _backend_cache[preference]


def make_handler(app: UiApp) -> type[BaseHTTPRequestHandler]:
    class UiHandler(BaseHTTPRequestHandler):
        server_version = "s1core-ui/0.1"

        def log_message(self, *args):  # quiet — the UI is its own log
            pass

        # ---------------------------------------------------------- plumbing

        def _send(self, status: int, content: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"),
                       "application/json")

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                raise ApiError(413, "request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise ApiError(400, f"invalid JSON body: {exc.msg}")
            if not isinstance(data, dict):
                raise ApiError(400, "request body must be a JSON object")
            return data

        # ------------------------------------------------------------- GET

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self._serve_static("index.html")
                elif path.startswith("/static/"):
                    self._serve_static(path[len("/static/"):])
                elif path == "/api/state":
                    live, _toy = scenarios_backend("live")
                    self._json(200, {
                        "version": s1core.__version__,
                        "schema_version": SCHEMA_VERSION,
                        "backend": {
                            "kind": "live" if live else "toy",
                            "model_id": (live or scenarios_backend("toy")[0]).model_id,
                        },
                        "key_present": live is not None,
                    })
                elif path == "/api/scenarios":
                    self._json(200, {"scenarios": SCENARIOS})
                elif path == "/api/ledgers":
                    self._json(200, {"ledgers": app.list_ledgers()})
                elif match := re.fullmatch(r"/api/runs/([A-Za-z0-9_]+)", path):
                    self._json(200, app.get_run(match.group(1)))
                else:
                    self._error(404, f"no route: {path}")
            except ApiError as exc:
                self._error(exc.status, str(exc))
            except Exception:
                traceback.print_exc()
                self._error(500, "internal server error")

        def _serve_static(self, name: str) -> None:
            target = (STATIC / name).resolve()
            if not str(target).startswith(str(STATIC.resolve()) + "/") \
                    and target != STATIC.resolve():
                self._error(403, "static path escapes the static dir")
                return
            if not target.is_file():
                self._error(404, f"no such static file: {name}")
                return
            self._send(200, target.read_bytes(),
                       _MIME.get(target.suffix, "application/octet-stream"))

        # ------------------------------------------------------------ POST

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                body = self._read_json()
                if path == "/api/evaluate":
                    self._json(200, app.store(app.do_evaluate(body)))
                elif path == "/api/ledger/load":
                    self._json(200, app.store(app.do_ledger_load(body)))
                elif path == "/api/sweep":
                    self._json(200, app.do_sweep(body))
                elif path == "/api/calibrate":
                    self._json(200, app.do_calibrate(body))
                elif path == "/api/replay":
                    self._json(200, app.do_replay(body))
                elif match := re.fullmatch(
                        r"/api/scenarios/([A-Za-z0-9_-]+)/run", path):
                    scenario_id = match.group(1)
                    if not any(s["id"] == scenario_id for s in SCENARIOS):
                        raise ApiError(404, f"unknown scenario: {scenario_id}")
                    payload = scenarios.run_scenario(
                        scenario_id, str(body.get("backend", "auto")))
                    self._json(200, app.store(payload))
                else:
                    self._error(404, f"no route: {path}")
            except ApiError as exc:
                self._error(exc.status, str(exc))
            except (SpecError, StateError) as exc:
                self._error(422, str(exc))
            except Exception:
                traceback.print_exc()
                self._error(500, "internal server error")

    return UiHandler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="S1-Core local UI (127.0.0.1 only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow", action="append", default=[],
        help="additional directory ledger files may be loaded from "
             "(repeatable); the repo root is always allowed")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open a browser tab")
    args = parser.parse_args(argv)

    roots = [_REPO_ROOT] + [Path(p) for p in args.allow]
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 make_handler(UiApp(roots)))
    url = f"http://127.0.0.1:{args.port}"
    print(f"S1-Core UI  →  {url}   (Ctrl-C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
