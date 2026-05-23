from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import inventory
from .io_utils import write_json
from .pipeline import run_pipeline
from .captioning import generate_captions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 1 object list + mask artifacts.")
    parser.add_argument("--config", default="phrase1_sam3/configs/default.yaml", help="Path to config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="Inspect configured VQA and image data.")
    inventory_parser.add_argument("--output", default="", help="Optional path for inventory JSON.")

    run_parser = subparsers.add_parser("run", help="Run candidate extraction and SAM backend.")
    run_parser.add_argument("--max-images", type=int, default=None, help="Override config max_images.")
    run_parser.add_argument("--sam-backend", default="", help="Override SAM backend: dryrun, none, sam3.")
    run_parser.add_argument("--source-variant", default="", help="Override source variant A/B/C/D/E.")
    run_parser.add_argument("--top-k", type=int, default=None, help="Override top_k prompts per image.")
    run_parser.add_argument("--output-layout", choices=["grouped_json", "single_json", "per_image"], default="", help="Override output layout.")
    run_parser.add_argument("--captions-file", default="", help="Override captions file read by the run command.")

    caption_parser = subparsers.add_parser("caption", help="Generate image captions with BLIP-2 and/or Florence-2.")
    caption_parser.add_argument("--max-images", type=int, default=None, help="Override config max_images.")
    caption_parser.add_argument("--models", default="", help="Comma-separated caption models: blip2,florence2.")
    caption_parser.add_argument("--output", default="", help="Output captions JSON path.")
    caption_parser.add_argument("--device", default="", help="Override caption device: auto, cuda, cuda:0, cpu.")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "inventory":
        report = inventory(config)
        output = Path(args.output) if args.output else config.output_dir / "inventory.json"
        write_json(output, report)
        print(f"Wrote inventory report to {output}")
        print(f"Questions: {report.get('questions_file', {}).get('question_count', 0)}")
        print(f"Images with questions: {report.get('questions_file', {}).get('image_count', 0)}")
        return 0

    if args.command == "caption":
        if args.output:
            config.caption_output_file = args.output
            config.captions_file = args.output
        if args.device:
            config.caption_device = args.device
        output = generate_captions(config, max_images=args.max_images, models=args.models or None)
        metadata = output["metadata"]
        print(f"Wrote captions to {config.caption_output_file}")
        print(f"Images: {metadata['image_count']}")
        print(f"Captions: {metadata['caption_count']}")
        print(f"Models: {', '.join(metadata['models'])}")
        print(f"Loaded models: {', '.join(metadata.get('loaded_models', [])) or 'none'}")
        if metadata.get("model_errors"):
            print("Model errors:")
            for model_name, error in metadata["model_errors"].items():
                print(f"- {model_name}: {error}")
        if metadata["caption_count"] == 0:
            return 1
        return 0

    if args.command == "run":
        if args.sam_backend:
            config.sam_backend = args.sam_backend
        if args.source_variant:
            config.source_variant = args.source_variant.upper()
        if args.top_k is not None:
            config.top_k = args.top_k
        if args.output_layout:
            config.output_layout = args.output_layout
        if args.captions_file:
            config.captions_file = args.captions_file
        summary = run_pipeline(config, max_images=args.max_images)
        if config.output_layout == "grouped_json":
            target = "artifacts.json, candidates.json, raw_sam.json, and summary.json"
        elif config.output_layout == "single_json":
            target = "run.json"
        else:
            target = "per-image JSON files"
        print(f"Wrote {target} to {summary['output_dir']}")
        print(f"Images: {summary['image_count']}")
        print(f"Prompts: {summary['prompt_count']}")
        print(f"Accepted objects: {summary['accepted_object_count']}")
        return 0

    return 2
