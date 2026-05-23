from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import spacy


_nlp: spacy.Language | None = None


def _get_nlp(model_name: str = "en_core_web_sm") -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(model_name)
    return _nlp


@dataclass
class Candidate:
    canonical_name: str
    candidate_phrases: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    support_count: int = 0
    score: float = 0.0
    status: str = "accepted"

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "candidate_phrases": sorted(self.candidate_phrases),
            "sources": sorted(self.sources),
            "support_count": self.support_count,
            "score": round(self.score, 3),
            "status": self.status,
        }


COCO_OBJECTS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "television",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}

COMMON_OBJECTS = COCO_OBJECTS | {
    "table", "desk", "door", "window", "wall", "floor", "ceiling", "shirt", "pants", "shorts",
    "shoe", "shoes", "hat", "helmet", "jacket", "dress", "plate", "glass", "sign", "pole",
    "building", "house", "road", "street", "sidewalk", "room", "kitchen", "bathroom", "flower",
    "flowers", "plant", "tree", "food", "fruit", "vegetable", "monitor", "computer", "phone",
    "paper", "bag", "box", "cart", "net", "racket", "board", "surf board", "bike", "plane",
    "animal", "animals", "child", "children", "man", "woman", "boy", "girl", "people",
}

SYNONYMS = {
    "people": "person",
    "men": "person",
    "women": "person",
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "child": "person",
    "children": "person",
    "bike": "bicycle",
    "bikes": "bicycle",
    "tv": "television",
    "t.v.": "television",
    "cellphone": "phone",
    "cell phone": "phone",
    "mobile phone": "phone",
    "couches": "couch",
    "sofa": "couch",
    "racket": "tennis racket",
    "surf board": "surfboard",
    "plane": "airplane",
}

REJECT_TERMS = {
    "scene", "background", "foreground", "view", "image", "picture", "photo", "side", "thing",
    "object", "area", "kind", "type", "color", "shape", "number", "time", "brand", "sport",
    "meal", "material", "condition", "chance", "reason", "way", "sense", "mood", "vibe",
    "productivity", "letter", "word", "name", "company", "motto", "country", "city", "delay",
}

RISKY_TERMS = {"sky", "grass", "water", "road", "street", "room", "crowd", "floor", "wall"}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been", "am", "do", "does",
    "did", "can", "could", "will", "would", "should", "there", "this", "that", "these",
    "those", "what", "which", "who", "where", "when", "why", "how", "many", "much", "in",
    "on", "at", "of", "for", "to", "from", "with", "without", "and", "or", "but", "as",
    "it", "its", "his", "her", "their", "they", "them", "he", "she", "you", "your", "any",
    "all", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "only", "most", "likely", "possibly", "pictured", "shown", "seen", "visible",
}

ATTRIBUTE_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "orange", "brown", "gray", "grey",
    "pink", "purple", "wooden", "metal", "plastic", "glass", "striped", "round", "square",
    "tall", "short", "small", "large", "big", "old", "young", "dark", "light",
}

YES_NO = {"yes", "no", "yeah", "nope"}


def extract_candidates_from_record(record: Any, source_variant: str) -> tuple[list[Candidate], list[dict[str, Any]]]:
    merged: dict[str, Candidate] = {}
    rejected: list[dict[str, Any]] = []

    include_caption = source_variant in {"A", "D", "E"}
    include_vqa = source_variant in {"B", "D", "E"}
    include_labels = source_variant in {"C", "E"}

    if include_caption:
        for caption in record.captions:
            _add_text_candidates(merged, rejected, caption, "caption")

    if include_vqa:
        for question in record.questions:
            _add_text_candidates(merged, rejected, str(question.get("question", "")), "vqa_text")
        _add_answer_candidates(merged, rejected, record)

    if include_labels:
        for label in record.labels:
            _add_phrase(merged, rejected, label, "dataset_label")

    ranked = sorted(_score_candidates(merged.values()), key=lambda item: (-item.score, item.canonical_name))
    return ranked, rejected


def _add_answer_candidates(
    merged: dict[str, Candidate],
    rejected: list[dict[str, Any]],
    record: Any,
) -> None:
    question_by_id = {
        str(question.get("question_id", question.get("qid", ""))): str(question.get("question", ""))
        for question in record.questions
    }

    for answer_row in record.answers:
        answers = _answer_strings(answer_row)
        question_text = question_by_id.get(str(answer_row.get("question_id", answer_row.get("qid", ""))), "")
        question_objects = [
            phrase for phrase in _extract_phrases_open(question_text)
            if phrase not in REJECT_TERMS and phrase not in ATTRIBUTE_WORDS
        ]
        for answer in answers:
            normalized = answer.lower().strip()
            if not normalized or normalized in YES_NO or _is_number(normalized):
                continue
            if normalized in ATTRIBUTE_WORDS and question_objects:
                _add_phrase(merged, rejected, f"{normalized} {question_objects[0]}", "vqa_text")
            else:
                _add_text_candidates(merged, rejected, normalized, "vqa_text")


def _answer_strings(answer_row: dict[str, Any]) -> list[str]:
    values = []
    if answer_row.get("multiple_choice_answer"):
        values.append(str(answer_row["multiple_choice_answer"]))
    for item in answer_row.get("answers", []):
        if isinstance(item, dict) and item.get("answer"):
            values.append(str(item["answer"]))
    return sorted(set(values))


def _add_text_candidates(
    merged: dict[str, Candidate],
    rejected: list[dict[str, Any]],
    text: str,
    source: str,
) -> None:
    _add_rejected_noun_heads(rejected, text, source)
    for phrase in _extract_phrases_open(text):
        _add_phrase(merged, rejected, phrase, source)


def _extract_phrases_open(text: str) -> list[str]:
    doc = _get_nlp()(text)
    phrases: set[str] = set()

    for chunk in doc.noun_chunks:
        head_lemma = chunk.root.lemma_.lower()
        if chunk.root.pos_ in {"PRON", "DET"}:
            continue
        if head_lemma in REJECT_TERMS or head_lemma in STOP_WORDS:
            continue

        clean_tokens = [
            token.lemma_.lower()
            for token in chunk
            if token.pos_ in {"ADJ", "NOUN"} and token.lemma_.lower() not in STOP_WORDS
        ]
        clean_phrase = " ".join(clean_tokens)
        if clean_phrase:
            phrases.add(clean_phrase)
        if head_lemma not in STOP_WORDS and len(head_lemma) > 2:
            phrases.add(head_lemma)

    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    normalized_text = " ".join(tokens)
    for phrase in sorted(COMMON_OBJECTS, key=lambda item: -len(item)):
        if " " in phrase and re.search(rf"\b{re.escape(phrase)}\b", normalized_text):
            phrases.add(phrase)

    return sorted(phrases)


def _add_rejected_noun_heads(rejected: list[dict[str, Any]], text: str, source: str) -> None:
    for chunk in _get_nlp()(text).noun_chunks:
        head_lemma = chunk.root.lemma_.lower()
        if head_lemma in REJECT_TERMS:
            rejected.append({"phrase": head_lemma, "source": source, "reason": "generic_or_non_groundable"})


def _add_phrase(
    merged: dict[str, Candidate],
    rejected: list[dict[str, Any]],
    phrase: str,
    source: str,
) -> None:
    clean = _clean_phrase(phrase)
    if not clean:
        return
    canonical = canonicalize(clean)
    if canonical in REJECT_TERMS:
        rejected.append({"phrase": clean, "source": source, "reason": "generic_or_non_groundable"})
        return

    candidate = merged.setdefault(canonical, Candidate(canonical_name=canonical))
    candidate.candidate_phrases.add(clean)
    candidate.candidate_phrases.add(canonical)
    candidate.sources.add(source)
    candidate.support_count += 1


def _score_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    source_weights = {"dataset_label": 4.0, "vqa_text": 3.0, "caption": 2.0, "generated": 1.0}
    for candidate in candidates:
        score = candidate.support_count
        score += sum(source_weights.get(source, 0.5) for source in candidate.sources)
        if candidate.canonical_name in COCO_OBJECTS:
            score += 1.5
        elif candidate.canonical_name in COMMON_OBJECTS:
            score += 0.75
        else:
            score -= 1.0
            if candidate.support_count <= 1:
                score -= 1.5
        if candidate.canonical_name in RISKY_TERMS:
            score -= 1.0
            candidate.status = "risky"
        if len(candidate.canonical_name.split()) == 1:
            score += 0.25
        candidate.score = score
    return list(candidates)


def canonicalize(phrase: str) -> str:
    phrase = _clean_phrase(phrase)
    if phrase in SYNONYMS:
        return SYNONYMS[phrase]
    if phrase in COCO_OBJECTS:
        return phrase

    words = phrase.split()
    if len(words) > 1 and words[-1] in SYNONYMS:
        return SYNONYMS[words[-1]]
    if not words:
        return phrase

    head_surface = words[-1]
    nlp = _get_nlp()
    head_in_vocab = head_surface in nlp.vocab
    doc = nlp(phrase)
    head_lemma = doc[-1].lemma_.lower() if doc else head_surface
    if head_lemma in SYNONYMS:
        return SYNONYMS[head_lemma]
    if head_lemma and head_lemma != head_surface:
        return head_lemma

    if not head_in_vocab and head_surface.endswith("ies") and len(head_surface) > 4:
        return head_surface[:-3] + "y"
    if not head_in_vocab and head_surface.endswith("es") and len(head_surface) > 4:
        singular = head_surface[:-2]
        if singular in COMMON_OBJECTS:
            return singular
    if not head_in_vocab and head_surface.endswith("s") and len(head_surface) > 3:
        singular = head_surface[:-1]
        if singular in COMMON_OBJECTS or singular in SYNONYMS:
            return SYNONYMS.get(singular, singular)
    return head_surface


def _clean_phrase(phrase: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", phrase.lower())
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens[:4])


def _looks_like_groundable_phrase(phrase: str) -> bool:
    tokens = phrase.split()
    if not tokens:
        return False
    if len(tokens) == 1:
        return tokens[0] not in ATTRIBUTE_WORDS and len(tokens[0]) > 2
    return tokens[-1] not in ATTRIBUTE_WORDS and any(token in ATTRIBUTE_WORDS for token in tokens[:-1])


def _is_number(value: str) -> bool:
    return value.isdigit() or value in {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
