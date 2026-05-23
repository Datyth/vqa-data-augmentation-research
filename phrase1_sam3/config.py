from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    datasetlink: Path
    imageslink: Path
    output_dir: Path = Path("phrase1_sam3/outputs")
    output_layout: str = "grouped_json"
    split: str = "rest_val"
    questions_file: str = "v2_OpenEnded_mscoco_rest_val2014_questions.json"
    annotations_file: str = "v2_mscoco_rest_val2014_annotations.json"
    captions_file: str = "phrase1_sam3/outputs/captions.json"
    caption_models: str = "blip2,florence2"
    caption_output_file: str = "phrase1_sam3/outputs/captions.json"
    caption_device: str = "auto"
    caption_torch_dtype: str = "auto"
    caption_max_new_tokens: int = 64
    caption_num_beams: int = 3
    blip2_model_id: str = "Salesforce/blip2-opt-2.7b"
    florence2_model_id: str = "microsoft/Florence-2-base"
    dataset_labels_file: str = ""
    source_variant: str = "D"
    top_k: int = 10
    max_images: int = 30
    mask_threshold: float = 0.6
    min_area_fraction: float = 0.0005
    max_area_fraction: float = 0.85
    duplicate_iou: float = 0.8
    sam_backend: str = "dryrun"


DEFAULTS = {
    "output_dir": "phrase1_sam3/outputs",
    "output_layout": "grouped_json",
    "split": "rest_val",
    "questions_file": "v2_OpenEnded_mscoco_rest_val2014_questions.json",
    "annotations_file": "v2_mscoco_rest_val2014_annotations.json",
    "captions_file": "phrase1_sam3/outputs/captions.json",
    "caption_models": "blip2,florence2",
    "caption_output_file": "phrase1_sam3/outputs/captions.json",
    "caption_device": "auto",
    "caption_torch_dtype": "auto",
    "caption_max_new_tokens": 64,
    "caption_num_beams": 3,
    "blip2_model_id": "Salesforce/blip2-opt-2.7b",
    "florence2_model_id": "microsoft/Florence-2-base",
    "dataset_labels_file": "",
    "source_variant": "D",
    "top_k": 10,
    "max_images": 30,
    "mask_threshold": 0.6,
    "min_area_fraction": 0.0005,
    "max_area_fraction": 0.85,
    "duplicate_iou": 0.8,
    "sam_backend": "dryrun",
}


def load_config(path: str | Path) -> Config:
    raw = dict(DEFAULTS)
    raw.update(_parse_simple_config(Path(path)))

    if "datasetlink" not in raw or "imageslink" not in raw:
        raise ValueError("Config must define datasetlink and imageslink")

    return Config(
        datasetlink=Path(str(raw["datasetlink"])).expanduser(),
        imageslink=Path(str(raw["imageslink"])).expanduser(),
        output_dir=Path(str(raw["output_dir"])).expanduser(),
        output_layout=str(raw["output_layout"]),
        split=str(raw["split"]),
        questions_file=str(raw["questions_file"]),
        annotations_file=str(raw["annotations_file"]),
        captions_file=str(raw["captions_file"]),
        caption_models=str(raw["caption_models"]),
        caption_output_file=str(raw["caption_output_file"]),
        caption_device=str(raw["caption_device"]),
        caption_torch_dtype=str(raw["caption_torch_dtype"]),
        caption_max_new_tokens=int(raw["caption_max_new_tokens"]),
        caption_num_beams=int(raw["caption_num_beams"]),
        blip2_model_id=str(raw["blip2_model_id"]),
        florence2_model_id=str(raw["florence2_model_id"]),
        dataset_labels_file=str(raw["dataset_labels_file"]),
        source_variant=str(raw["source_variant"]).upper(),
        top_k=int(raw["top_k"]),
        max_images=int(raw["max_images"]),
        mask_threshold=float(raw["mask_threshold"]),
        min_area_fraction=float(raw["min_area_fraction"]),
        max_area_fraction=float(raw["max_area_fraction"]),
        duplicate_iou=float(raw["duplicate_iou"]),
        sam_backend=str(raw["sam_backend"]),
    )


def _parse_simple_config(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        if "=" in clean:
            key, value = clean.split("=", 1)
        elif ":" in clean:
            key, value = clean.split(":", 1)
        else:
            raise ValueError(f"Invalid config line {line_number}: {line}")
        values[key.strip()] = _parse_value(value.strip())
    return values


def _parse_value(value: str) -> Any:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    try:
        if any(part in value for part in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value
