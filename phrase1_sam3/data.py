from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .io_utils import iter_jsonl, read_json


@dataclass
class ImageRecord:
    image_id: str
    image_path: Path
    split: str
    questions: list[dict[str, Any]] = field(default_factory=list)
    answers: list[dict[str, Any]] = field(default_factory=list)
    captions: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def inventory(config: Config) -> dict[str, Any]:
    dataset_dir = config.datasetlink
    images_dir = config.imageslink
    questions_path = dataset_dir / config.questions_file
    annotations_path = dataset_dir / config.annotations_file
    captions_path = _optional_path(config.captions_file)
    labels_path = _optional_path(config.dataset_labels_file)

    report = {
        "dataset_dir": str(dataset_dir),
        "images_dir": str(images_dir),
        "dataset_dir_exists": dataset_dir.exists(),
        "images_dir_exists": images_dir.exists(),
        "questions_file": _file_report(questions_path),
        "annotations_file": _file_report(annotations_path),
        "captions_file": _file_report(captions_path) if captions_path else None,
        "dataset_labels_file": _file_report(labels_path) if labels_path else None,
        "image_splits": {},
        "jsonl_files": [],
    }

    if images_dir.exists():
        for child in sorted(images_dir.iterdir()):
            if child.is_dir():
                report["image_splits"][child.name] = _count_images(child, limit=1000000)

    if dataset_dir.exists():
        for path in sorted(dataset_dir.glob("*.jsonl")):
            report["jsonl_files"].append({"name": path.name, "size_bytes": path.stat().st_size})

    if questions_path.exists():
        questions = load_questions(questions_path)
        image_ids = {str(item["image_id"]) for item in questions if "image_id" in item}
        report["questions_file"]["question_count"] = len(questions)
        report["questions_file"]["image_count"] = len(image_ids)

    if annotations_path.exists():
        annotations = load_annotations(annotations_path)
        report["annotations_file"]["annotation_count"] = len(annotations)

    return report


def load_records(config: Config) -> list[ImageRecord]:
    questions_path = config.datasetlink / config.questions_file
    annotations_path = config.datasetlink / config.annotations_file

    questions = load_questions(questions_path)
    answers_by_qid = _answers_by_question(load_annotations(annotations_path)) if annotations_path.exists() else {}
    captions_by_image = load_text_mapping(_optional_path(config.captions_file), "caption")
    labels_by_image = load_text_mapping(_optional_path(config.dataset_labels_file), "label")

    by_image: dict[str, ImageRecord] = {}
    for question in questions:
        image_id = str(question.get("image_id", ""))
        if not image_id:
            continue
        record = by_image.setdefault(
            image_id,
            ImageRecord(
                image_id=image_id,
                image_path=resolve_image_path(config.imageslink, image_id, question.get("image_path"), config.split),
                split=config.split,
            ),
        )
        record.questions.append(question)
        question_id = str(question.get("question_id", question.get("qid", "")))
        if question_id in answers_by_qid:
            record.answers.append(answers_by_qid[question_id])

    for image_id, captions in captions_by_image.items():
        if image_id in by_image:
            by_image[image_id].captions.extend(captions)
    for image_id, labels in labels_by_image.items():
        if image_id in by_image:
            by_image[image_id].labels.extend(labels)

    return list(by_image.values())


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("questions"), list):
        return data["questions"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported questions format: {path}")


def load_annotations(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        return data["annotations"]
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("question_id", key)
                rows.append(row)
        if rows:
            return rows
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported annotations format: {path}")


def load_text_mapping(path: Path | None, value_key: str) -> dict[str, list[str]]:
    if not path or not path.exists():
        return {}

    by_image: dict[str, list[str]] = defaultdict(list)
    if path.suffix == ".jsonl":
        rows = iter_jsonl(path)
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        data = read_json(path)
        if isinstance(data, dict):
            rows = data.get("items") or data.get("annotations") or data.get("captions") or data
            if isinstance(rows, dict):
                for image_id, values in rows.items():
                    if isinstance(values, str):
                        by_image[str(image_id)].append(values)
                    elif isinstance(values, list):
                        by_image[str(image_id)].extend(str(item) for item in values)
                return dict(by_image)
        else:
            rows = data

    for row in rows:
        if not isinstance(row, dict):
            continue
        image_id = str(row.get("image_id", row.get("id", "")))
        value = row.get(value_key, row.get("text", row.get("name", "")))
        if image_id and value:
            by_image[image_id].append(str(value))
    return dict(by_image)


def resolve_image_path(images_dir: Path, image_id: str, row_path: Any, split: str) -> Path:
    if row_path:
        candidate = Path(str(row_path))
        if candidate.is_absolute():
            return candidate
        if (images_dir / candidate).exists():
            return images_dir / candidate

    numeric = _numeric_image_id(image_id)
    split_dir = "val2014" if "val" in split else "train2014"
    return images_dir / split_dir / f"COCO_{split_dir}_{numeric:012d}.jpg"


def _answers_by_question(annotations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in annotations:
        question_id = str(row.get("question_id", row.get("qid", "")))
        if question_id:
            result[question_id] = row
    return result


def _optional_path(value: str) -> Path | None:
    return Path(value).expanduser() if value else None


def _file_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _count_images(path: Path, limit: int) -> int:
    count = 0
    for child in path.iterdir():
        if child.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            count += 1
            if count >= limit:
                break
    return count


def _numeric_image_id(image_id: str) -> int:
    match = re.search(r"(\d+)$", image_id)
    return int(match.group(1)) if match else int(image_id)
