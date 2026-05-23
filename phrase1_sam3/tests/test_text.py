import unittest
from types import SimpleNamespace

from phrase1_sam3.text import canonicalize, extract_candidates_from_record


class TextExtractionTest(unittest.TestCase):
    def test_canonicalize_synonyms(self):
        self.assertEqual(canonicalize("men"), "person")
        self.assertEqual(canonicalize("bike"), "bicycle")
        self.assertEqual(canonicalize("cell phone"), "phone")

    def test_extracts_question_and_answer_objects(self):
        record = SimpleNamespace(
            captions=["A red cup sits beside a laptop."],
            questions=[{"question_id": 1, "question": "What is the man holding?"}],
            answers=[{"question_id": 1, "multiple_choice_answer": "tennis racket", "answers": []}],
            labels=[],
        )
        candidates, rejected = extract_candidates_from_record(record, "D")
        names = {item.canonical_name for item in candidates}
        self.assertIn("cup", names)
        self.assertIn("laptop", names)
        self.assertIn("person", names)
        self.assertIn("tennis racket", names)
        self.assertIsInstance(rejected, list)

    def test_extracts_unknown_caption_objects_with_lower_score(self):
        record = SimpleNamespace(
            captions=["A stroller is parked near the bench."],
            questions=[],
            answers=[],
            labels=[],
        )
        candidates, _ = extract_candidates_from_record(record, "A")
        by_name = {item.canonical_name: item for item in candidates}
        self.assertIn("stroller", by_name)
        self.assertIn("bench", by_name)
        self.assertGreater(by_name["bench"].score, by_name["stroller"].score)

    def test_lemmatizes_irregular_plural_caption_object(self):
        record = SimpleNamespace(
            captions=["Three mice are on the table."],
            questions=[],
            answers=[],
            labels=[],
        )
        candidates, _ = extract_candidates_from_record(record, "A")
        names = {item.canonical_name for item in candidates}
        self.assertIn("mouse", names)
        self.assertNotIn("mice", names)

    def test_rejects_abstract_caption_heads(self):
        record = SimpleNamespace(
            captions=["The reason for the delay is unknown."],
            questions=[],
            answers=[],
            labels=[],
        )
        candidates, rejected = extract_candidates_from_record(record, "A")
        names = {item.canonical_name for item in candidates}
        rejected_names = {item["phrase"] for item in rejected}
        self.assertIn("reason", rejected_names)
        self.assertNotIn("delay", names)
        self.assertNotIn("unknown", names)


if __name__ == "__main__":
    unittest.main()
