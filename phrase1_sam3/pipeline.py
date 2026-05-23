from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .data import ImageRecord, load_records
from .io_utils import write_json
from .sam_backend import SamBackend, make_sam_backend
from .text import Candidate, extract_candidates_from_record


def run_pipeline(config: Config, max_images: int | None = None) -> dict[str, Any]:
    backend = make_sam_backend(config.sam_backend, config.output_dir)
    _clear_sam3_mask_pngs(config)
    records = load_records(config)
    limit = max_images if max_images is not None else config.max_images
    selected_records = records[:limit]

    summary = {
        "config": {
            "source_variant": config.source_variant,
            "top_k": config.top_k,
            "mask_threshold": config.mask_threshold,
            "sam_backend": config.sam_backend,
            "output_layout": config.output_layout,
        },
        "image_count": 0,
        "accepted_object_count": 0,
        "prompt_count": 0,
        "output_dir": str(config.output_dir),
    }
    grouped = {"artifacts": {}, "candidates": {}, "raw_sam": {}}
    combined = {"summary": summary, **grouped}

    for record in _sam3_progress(selected_records, config):
        artifact = process_record(record, config, backend)
        image_id = record.image_id
        summary["image_count"] += 1
        summary["accepted_object_count"] += len(artifact["final"]["objects"])
        summary["prompt_count"] += len(artifact["final"]["sam3_prompt_log"])

        if config.output_layout == "per_image":
            write_json(config.output_dir / "artifacts" / f"{image_id}.json", artifact["final"])
            write_json(config.output_dir / "candidates" / f"{image_id}.json", artifact["candidates"])
            write_json(config.output_dir / "raw_sam" / f"{image_id}.json", artifact["raw_sam"])
        elif config.output_layout in {"grouped_json", "single_json"}:
            grouped["artifacts"][image_id] = artifact["final"]
            grouped["candidates"][image_id] = artifact["candidates"]
            grouped["raw_sam"][image_id] = artifact["raw_sam"]
        else:
            raise ValueError(f"Unsupported output_layout: {config.output_layout}")

    if config.output_layout == "per_image":
        write_json(config.output_dir / "summary.json", summary)
    elif config.output_layout == "grouped_json":
        write_json(config.output_dir / "summary.json", summary)
        write_json(config.output_dir / "artifacts.json", grouped["artifacts"])
        write_json(config.output_dir / "candidates.json", grouped["candidates"])
        write_json(config.output_dir / "raw_sam.json", grouped["raw_sam"])
    else:
        write_json(config.output_dir / "run.json", combined)
    return summary


def process_record(record: ImageRecord, config: Config, backend: SamBackend) -> dict[str, Any]:
    candidates, rejected = extract_candidates_from_record(record, config.source_variant)
    selected = candidates[: config.top_k]
    raw_sam = []
    objects = []
    merged_masks: list[dict[str, Any]] = []

    for candidate in selected:
        prompts = _prompts_for_candidate(candidate)
        for prompt in prompts:
            raw_instances = [instance.as_dict() for instance in backend.segment(record.image_path, prompt)]
            raw_sam.append({"image_id": record.image_id, "prompt": prompt, "instances": raw_instances})
            for instance in raw_instances:
                accepted = _accepted_instance(instance, config)
                if not accepted:
                    rejected.append({"phrase": prompt, "source": "sam3", "reason": "mask_filter", "instance": instance})
                    continue
                _merge_or_add_mask(merged_masks, candidate, prompt, instance, config)

    for index, mask in enumerate(merged_masks, start=1):
        objects.append(
            {
                "object_id": f"obj_{index:03d}",
                "source_phrases": sorted(mask["source_phrases"]),
                "canonical_name": mask["canonical_name"],
                "mask_id": f"mask_{index:03d}",
                "bbox": mask["bbox"],
                "confidence": round(mask["confidence"], 3),
                "sources": sorted(mask["sources"]),
                **({"backend_mask_id": mask["backend_mask_id"]} if mask.get("backend_mask_id") else {}),
                **({"mask_path": mask["mask_path"]} if mask.get("mask_path") else {}),
                "status": "accepted",
                **({"label_conflict": True, "candidate_labels": sorted(mask["candidate_labels"])} if len(mask["candidate_labels"]) > 1 else {}),
            }
        )

    final = {
        "image_id": record.image_id,
        "image_path": str(record.image_path),
        "objects": objects,
        "rejected_candidates": rejected,
        "sam3_prompt_log": [{"prompt": item["prompt"], "instance_count": len(item["instances"])} for item in raw_sam],
        "metadata": {
            "caption_model": "external_or_not_provided",
            "source_variant": config.source_variant,
            "top_k": config.top_k,
            "mask_threshold": config.mask_threshold,
            "sam_backend": config.sam_backend,
        },
    }

    _retain_final_mask_pngs(final, raw_sam)
    return {
        "final": final,
        "candidates": [candidate.as_dict() for candidate in candidates],
        "raw_sam": {"image_id": record.image_id, "prompt_results": raw_sam},
    }


def _clear_sam3_mask_pngs(config: Config) -> None:
    if config.sam_backend.lower().strip() != "sam3":
        return
    mask_dir = config.output_dir / "masks"
    if not mask_dir.exists():
        return
    for mask_path in mask_dir.glob("*.png"):
        mask_path.unlink()


def _sam3_progress(records: list[ImageRecord], config: Config):
    if config.sam_backend.lower().strip() != "sam3":
        return records
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("SAM3 progress display requires tqdm.") from exc
    return tqdm(records, total=len(records), desc="SAM3 images", unit="image", dynamic_ncols=True)


def _retain_final_mask_pngs(final: dict[str, Any], raw_sam: list[dict[str, Any]]) -> None:
    retained_paths = {
        str(obj["mask_path"])
        for obj in final["objects"]
        if obj.get("mask_path")
    }
    for prompt_result in raw_sam:
        for instance in prompt_result["instances"]:
            raw_mask_path = instance.get("mask_path")
            if not raw_mask_path:
                continue
            retained = str(raw_mask_path) in retained_paths
            instance["mask_retained"] = retained
            if retained:
                continue
            Path(raw_mask_path).unlink(missing_ok=True)
            instance.pop("mask_path", None)


def _prompts_for_candidate(candidate: Candidate) -> list[str]:
    prompts = [candidate.canonical_name]
    for phrase in sorted(candidate.candidate_phrases, key=lambda item: (len(item.split()), item), reverse=True):
        if phrase != candidate.canonical_name and len(prompts) < 2:
            prompts.append(phrase)
    return prompts


def _accepted_instance(instance: dict[str, Any], config: Config) -> bool:
    if float(instance["score"]) < config.mask_threshold:
        return False
    image_area = max(1.0, float(instance.get("image_area", 640 * 480)))
    area_fraction = float(instance["area"]) / image_area
    return config.min_area_fraction <= area_fraction <= config.max_area_fraction


def _merge_or_add_mask(
    masks: list[dict[str, Any]],
    candidate: Candidate,
    prompt: str,
    instance: dict[str, Any],
    config: Config,
) -> None:
    for existing in masks:
        if _instance_iou(existing, instance) >= config.duplicate_iou:
            existing["source_phrases"].add(prompt)
            existing["source_phrases"].update(candidate.candidate_phrases)
            existing["sources"].update(candidate.sources)
            if float(instance["score"]) > existing["confidence"]:
                existing["backend_mask_id"] = instance.get("mask_id", existing.get("backend_mask_id", ""))
                existing["mask_path"] = instance.get("mask_path", existing.get("mask_path", ""))
            existing["confidence"] = max(existing["confidence"], float(instance["score"]))
            existing["candidate_labels"].add(candidate.canonical_name)
            if existing["canonical_name"] != candidate.canonical_name:
                existing["canonical_name"] = sorted(existing["candidate_labels"])[0]
            return

    masks.append(
        {
            "canonical_name": candidate.canonical_name,
            "source_phrases": set(candidate.candidate_phrases | {prompt}),
            "bbox": instance["bbox"],
            "confidence": float(instance["score"]),
            "backend_mask_id": instance.get("mask_id", ""),
            "mask_path": instance.get("mask_path", ""),
            "sources": set(candidate.sources),
            "candidate_labels": {candidate.canonical_name},
        }
    )


def _instance_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    mask_overlap = mask_iou(left.get("mask_path", ""), right.get("mask_path", ""))
    if mask_overlap is not None:
        return mask_overlap
    return bbox_iou(left["bbox"], right["bbox"])


def mask_iou(left_path: str, right_path: str) -> float | None:
    if not left_path or not right_path:
        return None
    try:
        from PIL import Image, ImageChops

        with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
            if left_image.size != right_image.size:
                return None
            left_mask = left_image.convert("1")
            right_mask = right_image.convert("1")
            intersection = ImageChops.logical_and(left_mask, right_mask).histogram()[255]
            union = ImageChops.logical_or(left_mask, right_mask).histogram()[255]
    except OSError:
        return None
    return intersection / union if union else 0.0


def bbox_iou(left: list[int], right: list[int]) -> float:
    left_x1, left_y1, left_w, left_h = left
    right_x1, right_y1, right_w, right_h = right
    left_x2, left_y2 = left_x1 + left_w, left_y1 + left_h
    right_x2, right_y2 = right_x1 + right_w, right_y1 + right_h

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = left_w * left_h + right_w * right_h - intersection
    return intersection / union if union else 0.0
