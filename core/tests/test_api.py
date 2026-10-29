"""Day 10 canonical API: request ids, state revisions, bounded bodies, headless run."""

from __future__ import annotations

import http.client
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constellab_core.__main__ import main  # noqa: E402
from constellab_core.config import MAX_CONFIG_BYTES  # noqa: E402
from constellab_core.contract import SimulationSnapshot  # noqa: E402
from constellab_core.service import CoreHandler, serve  # noqa: E402


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._log = CoreHandler.log_message
        CoreHandler.log_message = lambda *a, **k: None
        cls.server = serve("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        CoreHandler.log_message = cls._log
        cls.server.shutdown()
        cls.server.server_close()

    def _request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        request_headers = {"Content-Type": "application/json"} if body is not None else {}
        request_headers.update(headers or {})
        conn.request(method, path, body=body, headers=request_headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return res, json.loads(data) if data else None

    def _reset(self):
        res, _ = self._request("POST", "/api/scenario", json.dumps({"key": "visual-orbit-demo"}))
        self.assertEqual(res.status, 200)


class TestRequestIds(ApiTestCase):
    def test_supplied_request_id_is_echoed_on_the_header_and_body(self):
        res, payload = self._request("GET", "/api/state", headers={"X-Request-Id": "probe-42"})
        self.assertEqual(res.status, 200)
        self.assertEqual(res.getheader("X-Request-Id"), "probe-42")
        self.assertEqual(payload["request_id"], "probe-42")

    def test_missing_request_id_is_generated(self):
        res, payload = self._request("GET", "/api/status")
        generated = res.getheader("X-Request-Id")
        self.assertEqual(res.status, 200)
        self.assertRegex(generated, r"^[0-9a-f]{32}$")
        self.assertEqual(payload["name"], "constellab-core")

    def test_invalid_request_id_is_rejected_predictably(self):
        res, payload = self._request(
            "POST", "/api/scenario", json.dumps({"key": "empty"}),
            headers={"X-Request-Id": "bad id with spaces"},
        )
        self.assertEqual(res.status, 400)
        self.assertEqual(payload["error"], "invalid_request_id")
        self.assertIn("message", payload)
        # The reply still carries a freshly generated id so it can be traced.
        self.assertRegex(res.getheader("X-Request-Id"), r"^[0-9a-f]{32}$")

    def test_post_success_carries_the_request_id(self):
        res, payload = self._request(
            "POST", "/api/clock/pause", json.dumps({}), headers={"X-Request-Id": "pause-1"},
        )
        self.assertEqual(res.status, 200)
        self.assertEqual(payload["request_id"], "pause-1")
        self.assertIn("revision", payload)
        self._reset()


class TestStateStream(ApiTestCase):
    def test_state_payload_is_an_envelope_around_the_contract_snapshot(self):
        res, payload = self._request("GET", "/api/state")
        self.assertEqual(res.status, 200)
        self.assertEqual(res.getheader("X-Revision"), str(payload["revision"]))
        self.assertIsInstance(payload["revision"], int)
        self.assertEqual(payload["scenario_id"], "visual-orbit-demo")
        self.assertEqual(payload["clock"], payload["snapshot"]["clock"])
        snapshot = SimulationSnapshot.from_dict(payload["snapshot"])
        self.assertEqual(len(snapshot.satellites), 4)

    def test_snapshot_endpoint_stays_a_pure_contract_payload(self):
        res, payload = self._request("GET", "/api/snapshot")
        self.assertEqual(res.status, 200)
        self.assertNotIn("revision", payload)
        SimulationSnapshot.from_dict(payload)

    def test_revision_increases_while_the_clock_is_running(self):
        self._reset()
        _, first = self._request("GET", "/api/state")
        first_revision = first["revision"]
        later = first
        for _ in range(20):
            # The advance loop ticks every 100 ms; give it room to bump.
            time.sleep(0.06)
            _, later = self._request("GET", "/api/state")
            if later["revision"] > first_revision:
                break
        self.assertGreater(later["revision"], first_revision)
        self.assertGreaterEqual(later["snapshot"]["clock"]["simulation_time_s"],
                                first["snapshot"]["clock"]["simulation_time_s"])

    def test_every_state_change_bumps_the_revision(self):
        self._reset()
        _, before = self._request("GET", "/api/state")
        res, paused = self._request("POST", "/api/clock/pause", json.dumps({}))
        self.assertEqual(res.status, 200)
        self.assertGreater(paused["revision"], before["revision"])
        res, _ = self._request("POST", "/api/clock/pause", json.dumps({}))
        _, after = self._request("GET", "/api/state")
        self.assertGreater(after["revision"], before["revision"])
        self._reset()

    def test_scenario_load_reports_a_new_revision(self):
        self._reset()
        _, before = self._request("GET", "/api/state")
        res, loaded = self._request("POST", "/api/scenario", json.dumps({"key": "swarm-48"}))
        self.assertEqual(res.status, 200)
        self.assertGreater(loaded["revision"], before["revision"])
        _, after = self._request("GET", "/api/state")
        self.assertEqual(after["scenario_id"], "swarm-48")
        self.assertGreaterEqual(after["revision"], loaded["revision"])
        self._reset()


class TestBoundedRequests(ApiTestCase):
    def test_oversized_body_is_rejected(self):
        body = b"{" + b" " * (MAX_CONFIG_BYTES + 1) + b"}"
        res, payload = self._request("POST", "/api/scenario", body)
        self.assertEqual(res.status, 413)
        self.assertEqual(payload["error"], "payload_too_large")
        self.assertIn("request_id", payload)

    def test_non_json_content_type_is_rejected(self):
        res, payload = self._request(
            "POST", "/api/clock/pause", "x", headers={"Content-Type": "text/plain"},
        )
        self.assertEqual(res.status, 415)
        self.assertEqual(payload["error"], "unsupported_media_type")

    def test_unknown_route_is_a_structured_404(self):
        res, payload = self._request("GET", "/api/nope")
        self.assertEqual(res.status, 404)
        self.assertEqual(payload["error"], "not_found")
        self.assertIn("request_id", payload)
        self.assertIn("message", payload)

    def test_negative_time_scale_is_rejected(self):
        res, payload = self._request("POST", "/api/clock/time_scale", json.dumps({"time_scale": -2}))
        self.assertEqual(res.status, 400)
        self.assertEqual(payload["error"], "invalid_time_scale")
        self._reset()


class TestHeadlessRun(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_run_streams_records_without_a_server(self):
        code, out, err = self._run(["run", "swarm-48", "--duration", "60", "--step", "20"])
        self.assertEqual(code, 0)
        records = [json.loads(line) for line in out.splitlines()]
        self.assertEqual([record["t"] for record in records], [0.0, 20.0, 40.0, 60.0])
        self.assertEqual(len(records[0]["satellites"]), 48)
        self.assertFalse(records[0]["clock"]["paused"], "a headless run is not the paused epoch")
        self.assertEqual(records[0]["satellites"][0]["id"], "sat-swarm-01")
        # Motion is real: the same satellite moved between records.
        self.assertNotEqual(records[0]["satellites"][0]["position_eci_km"],
                            records[-1]["satellites"][0]["position_eci_km"])
        summary = json.loads(err.splitlines()[-1])
        self.assertEqual(summary["steps"], 4)
        self.assertEqual(summary["satellites"], 48)

    def test_run_writes_a_file_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "states.ndjson"
            code, out, err = self._run([
                "run", "visual-orbit-demo", "--duration", "30", "--step", "10",
                "--out", str(target),
            ])
            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            lines = target.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            self.assertTrue(json.loads(err.splitlines()[-1])["out"].endswith("states.ndjson"))
            os.unlink(target)

    def test_invalid_step_is_a_usage_error(self):
        code, _, err = self._run(["run", "empty", "--step", "0"])
        self.assertEqual(code, 3)
        self.assertIn("step must be > 0", err)

    def test_empty_scenario_runs_with_zero_satellites(self):
        code, out, _ = self._run(["run", "empty", "--duration", "0", "--step", "1"])
        self.assertEqual(code, 0)
        record = json.loads(out.splitlines()[0])
        self.assertEqual(record["satellites"], [])


class TestSwarmRegistries(ApiTestCase):
    def test_swarm_endpoint_exposes_the_day_25_to_30_registries(self):
        res, payload = self._request("GET", "/api/swarm")
        self.assertEqual(res.status, 200)
        self.assertEqual(payload["targets"]["kinds"],
                         ["hotspot", "flood-region", "change-detection"])
        self.assertEqual({item["key"] for item in payload["sensors"]["models"]},
                         {"sensor.conical", "sensor.agile"})
        self.assertEqual({item["key"] for item in payload["tasks"]["types"]},
                         {"task.imaging", "task.change-detection"})
        self.assertEqual({item["key"] for item in payload["policies"]["policies"]},
                         {"local-greedy", "fixed-priority", "leases-only",
                          "distributed-greedy", "oracle-upper-bound"})
        self.assertEqual({item["key"] for item in payload["contacts"]["models"]},
                         {"link.uhf", "link.optical"})
        self.assertIn("overflow_policy", payload["transport"]["defaults"])
        # The route is advertised in status alongside every other endpoint.
        _, status = self._request("GET", "/api/status")
        self.assertIn("GET /api/swarm", status["endpoints"])


class TestG2Endpoint(ApiTestCase):
    def test_g2_route_serves_the_cached_gate_report(self):
        res, payload = self._request("GET", "/api/g2")
        self.assertEqual(res.status, 200)
        self.assertEqual(payload["request_id"], res.getheader("X-Request-Id"))
        self.assertEqual(payload["revision"], int(res.getheader("X-Revision")))
        self.assertTrue(payload["checks"]["g2_pass"])
        self.assertTrue(payload["checks"]["replay_reproduces_hash"])
        self.assertEqual(payload["scenario_id"], "g2-partition")
        self.assertEqual(payload["fault"]["metrics"]["conflicts"], 2)
        self.assertEqual(payload["control"]["metrics"]["conflicts"], 0)
        self.assertEqual(payload["first_violation"]["rule"], "duplicate-task-claim")
        self.assertEqual(payload["timeline"][0]["time_s"], 0.0)

        # The route is discoverable through status, so a stale core is obvious.
        _, status = self._request("GET", "/api/status")
        self.assertIn("GET /api/g2", status["endpoints"])

    def test_ml_route_serves_the_compact_evidence_payload(self):
        res, payload = self._request("GET", "/api/ml")
        self.assertEqual(res.status, 200)
        self.assertEqual(payload["version"], "ml-report-1")
        self.assertIn("gate", payload)
        self.assertIn("generalization", payload)
        self.assertIn("benchmark", payload)
        self.assertIn("shield", payload)
        self.assertIn("compatibility", payload)
        self.assertIn("ablations", payload)
        self.assertIn("training", payload)
        self.assertIn("pipeline", payload)
        self.assertIn("manifest", payload["pipeline"])
        self.assertIn("request_id", payload)
        _, status = self._request("GET", "/api/status")
        self.assertIn("GET /api/ml", status["endpoints"])

    def test_g2_report_is_stable_across_requests(self):
        _, first = self._request("GET", "/api/g2")
        _, second = self._request("GET", "/api/g2")
        self.assertEqual(first["fault"]["trace_hash"], second["fault"]["trace_hash"])
        self.assertEqual(first["control"]["trace_hash"], second["control"]["trace_hash"])


if __name__ == "__main__":
    unittest.main()
