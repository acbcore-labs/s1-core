"""Stage 2 deterministic preprocessing tests."""

import unittest
from datetime import datetime, timezone

from s1core import Augmentor, IsoDateDeltaFacts, RatioFacts, CallableExtractor, digest_of


class IsoDateDeltaFactsTests(unittest.TestCase):
    def test_auto_detection(self):
        augmentor = Augmentor((IsoDateDeltaFacts(),))
        envelope = augmentor.augment(
            {"event_date": "2026-10-07T00:00:00+00:00", "note": "not a date"},
            datetime(2026, 10, 14, tzinfo=timezone.utc),
        )
        self.assertEqual(envelope["@deterministic_facts"]["event_date__days_from_reference"], -7)

    def test_explicit_keys_override_heuristic(self):
        augmentor = Augmentor((IsoDateDeltaFacts(["deadline"], suffix="__days_left"),))
        envelope = augmentor.augment(
            {"deadline": "2026-10-20"}, datetime(2026, 10, 14, tzinfo=timezone.utc)
        )
        facts = envelope["@deterministic_facts"]
        self.assertEqual(facts["deadline__days_left"], 6)

    def test_unparseable_skipped(self):
        augmentor = Augmentor((IsoDateDeltaFacts(["event_date"]),))
        envelope = augmentor.augment(
            {"event_date": "sometime soon"}, datetime(2026, 10, 14, tzinfo=timezone.utc)
        )
        self.assertEqual(envelope["@deterministic_facts"], {})

    def test_naive_treated_as_utc(self):
        augmentor = Augmentor((IsoDateDeltaFacts(),))
        envelope = augmentor.augment(
            {"closed_at": "2026-10-13T23:00:00"}, datetime(2026, 10, 14, tzinfo=timezone.utc)
        )
        self.assertEqual(envelope["@deterministic_facts"]["closed_at__days_from_reference"], -1)


class RatioFactsTests(unittest.TestCase):
    def test_ratio_and_exceeds_flag(self):
        augmentor = Augmentor((RatioFacts("requested_refund", "order_total"),))
        envelope = augmentor.augment(
            {"requested_refund": 150.0, "order_total": 120.0},
            datetime(2026, 10, 14, tzinfo=timezone.utc),
        )
        facts = envelope["@deterministic_facts"]
        self.assertAlmostEqual(facts["requested_refund_over_order_total"], 1.25)
        self.assertIs(facts["requested_refund_over_order_total__exceeds_1"], True)

    def test_missing_keys_noop(self):
        augmentor = Augmentor((RatioFacts("a", "b"),))
        envelope = augmentor.augment({"a": 1}, datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(envelope["@deterministic_facts"], {})


class CallableExtractorTests(unittest.TestCase):
    def test_custom_extractor(self):
        def word_count(state, facts, ctx):
            if isinstance(state.get("body"), str):
                facts["body_word_count"] = len(state["body"].split())

        augmentor = Augmentor((CallableExtractor(word_count),))
        envelope = augmentor.augment(
            {"body": "four words only here now"}, datetime(2026, 10, 14, tzinfo=timezone.utc)
        )
        self.assertEqual(envelope["@deterministic_facts"]["body_word_count"], 5)


class CanonicalizationTests(unittest.TestCase):
    def test_envelope_is_canonical_and_stable(self):
        augmentor = Augmentor((RatioFacts("a", "b"),))
        e1 = augmentor.augment({"a": 1.0, "b": 4.0}, datetime(2026, 10, 14, tzinfo=timezone.utc))
        e2 = augmentor.augment({"b": 4.0, "a": 1.0}, datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(digest_of(e1), digest_of(e2))

    def test_string_state_passes_through(self):
        augmentor = Augmentor((IsoDateDeltaFacts(),))
        envelope = augmentor.augment("plain text state", datetime(2026, 10, 14, tzinfo=timezone.utc))
        self.assertEqual(envelope["raw_state"], "plain text state")
        self.assertEqual(envelope["@deterministic_facts"], {})


if __name__ == "__main__":
    unittest.main()
