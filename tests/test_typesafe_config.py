"""TypeSafe wire-adapter and .env configuration tests (all offline)."""

import os
import unittest
from pathlib import Path

from s1core import ChoiceSpec, ScoreSpec, NoulSpec  # noqa: E402
from s1core.backends.typesafe import (  # noqa: E402
    estimate_cost_usd,
    typesafe_request_builder,
    typesafe_response_parser,
)
from s1core.config import load_dotenv, typesafe_driver_from_env  # noqa: E402


def compiled_batch():
    return {
        qid: spec.compile()
        for qid, spec in [
            ("lane", ChoiceSpec(
                question_id="lane",
                instructions="Which lane applies to this ticket?",
                options={"fast": "one meaningful criteria line here",
                         "slow": "another meaningful line of prose"},
            )),
            ("urgency", ScoreSpec(
                question_id="urgency",
                instructions="Rate the urgency on the rubric provided.",
                rubric_levels={1: "Routine with no impact.", 2: "Elevated impact today.",
                               3: "Critical impact right now."},
            )),
            ("gate", NoulSpec(
                question_id="gate",
                instructions="Does the ticket exceed the stated threshold?",
                true_criteria="It exceeds the stated threshold clearly.",
                false_criteria="It stays within the stated threshold.",
            )),
        ]
    }


class TypeSafeRequestBuilderTests(unittest.TestCase):
    def test_shape_and_score_array(self):
        batch = compiled_batch()
        payloads = [q.to_payload() for q in batch.values()]
        request = typesafe_request_builder("jev-latest", "/v1/systemone", {"s": 1}, payloads)
        self.assertEqual(set(request), {"model", "state", "questions"})
        self.assertEqual(sorted(request["questions"]), ["gate", "lane", "urgency"])
        # question_id is the map key, not repeated inside the object
        self.assertNotIn("question_id", request["questions"]["lane"])
        # score levels travel as an ordered array of descriptions
        self.assertEqual(
            request["questions"]["urgency"]["criteria"],
            ["Routine with no impact.", "Elevated impact today.", "Critical impact right now."],
        )
        # choice/noul criteria stay maps
        self.assertEqual(request["questions"]["lane"]["criteria"]["fast"],
                         "one meaningful criteria line here")
        self.assertEqual(request["questions"]["gate"]["criteria"]["true"],
                         "It exceeds the stated threshold clearly.")


class TypeSafeResponseParserTests(unittest.TestCase):
    def test_full_round_trip(self):
        batch = compiled_batch()
        body = {
            "answers": {
                "lane": {"type": "choice", "choice": "fast",
                         "probabilities": {"fast": 0.9, "slow": 0.1},
                         "confidence": 0.8},
                # 0-indexed wire levels for a 3-level rubric
                "urgency": {"type": "score", "score": 2.0,
                            "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
                            "confidence": 0.8},
                "gate": {"type": "noul", "noul": 0.95},
            },
            "usage": {"input_tokens": 296, "output_tokens": 20},
        }
        parsed = typesafe_response_parser(body, batch)
        self.assertAlmostEqual(parsed["lane"]["fast"], 0.9)
        # full simplex: declared criteria keys (incl. injected escape hatches) fill at 0.0
        self.assertEqual(
            set(parsed["lane"]),
            {"fast", "slow", "__abstain_insufficient_evidence__", "__abstain_out_of_domain__"},
        )
        # wire "0","1","2" restored to harness levels "1","2","3"
        self.assertAlmostEqual(parsed["urgency"]["1"], 0.0)
        self.assertAlmostEqual(parsed["urgency"]["2"], 0.2)
        self.assertAlmostEqual(parsed["urgency"]["3"], 0.8)
        self.assertAlmostEqual(parsed["gate"]["true"], 0.95)
        self.assertAlmostEqual(parsed["gate"]["false"], 0.05, places=12)

    def test_missing_question_raises(self):
        batch = compiled_batch()
        with self.assertRaises(Exception):
            typesafe_response_parser({"answers": {}}, batch)

    def test_cost_estimate(self):
        self.assertAlmostEqual(estimate_cost_usd(1_000_000), 0.042)
        self.assertAlmostEqual(estimate_cost_usd(1000), 0.000042)


class DotEnvTests(unittest.TestCase):
    def test_load_dotenv_parses_and_does_not_override(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                '# comment\nS1_TEST_KEY=" abc123 "\nS1_TEST_EMPTY=""\n'
                "S1_TEST_QUOTED='value'\nbroken line\n\n",
                encoding="utf-8",
            )
            os.environ["S1_TEST_EXISTING"] = "original"
            os.environ["S1_TEST_KEY"] = "original"
            try:
                applied = load_dotenv(path)
                self.assertEqual(os.environ["S1_TEST_EXISTING"], "original")  # untouched
                self.assertEqual(os.environ["S1_TEST_KEY"], "original")       # no override
                self.assertEqual(os.environ["S1_TEST_EMPTY"], "")
                self.assertEqual(os.environ["S1_TEST_QUOTED"], "value")
                # pre-existing keys are not reported as applied
                self.assertNotIn("S1_TEST_KEY", applied)
                self.assertIn("S1_TEST_QUOTED", applied)
            finally:
                for key in ("S1_TEST_KEY", "S1_TEST_EMPTY", "S1_TEST_QUOTED", "S1_TEST_EXISTING"):
                    os.environ.pop(key, None)

    def test_driver_from_env_none_without_key(self):
        saved = os.environ.pop("TYPESAFE_API_KEY", None)
        try:
            self.assertIsNone(typesafe_driver_from_env({}))
            self.assertIsNone(typesafe_driver_from_env({"TYPESAFE_API_KEY": "   "}))
        finally:
            if saved is not None:
                os.environ["TYPESAFE_API_KEY"] = saved

    def test_driver_from_env_builds_typesafe_driver(self):
        driver = typesafe_driver_from_env({"TYPESAFE_API_KEY": "k-test"})
        self.assertIsNotNone(driver)
        self.assertEqual(driver.model_id, "jev-latest")
        self.assertEqual(driver.base_url, "https://api.typesafe.ai")
        self.assertEqual(driver.endpoint, "/v1/systemone")


class TypeSafeParserValidationTests(unittest.TestCase):
    def test_missing_answers_map(self):
        from s1core.backends import BackendError

        with self.assertRaises(BackendError):
            typesafe_response_parser({"unexpected": True}, compiled_batch())

    def test_score_level_count_mismatch(self):
        from s1core.backends import BackendError

        batch = {"urgency": compiled_batch()["urgency"]}
        body = {"answers": {"urgency": {"type": "score",
                                        "probabilities": {"0": 0.5, "1": 0.5}}}}
        with self.assertRaises(BackendError):
            typesafe_response_parser(body, batch)

    def test_unknown_answer_type(self):
        from s1core.backends import BackendError

        batch = {"lane": compiled_batch()["lane"]}
        body = {"answers": {"lane": {"type": "gossip", "probabilities": {"fast": 1.0}}}}
        with self.assertRaises(BackendError):
            typesafe_response_parser(body, batch)


class DotEnvEdgeCaseTests(unittest.TestCase):
    def test_missing_file_is_empty(self):
        self.assertEqual(load_dotenv("/nonexistent/path/.env"), {})

    def test_lines_without_keys_are_skipped(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as fh:
            fh.write("=value\n")
            path = fh.name
        try:
            self.assertEqual(load_dotenv(path), {})
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
