from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .data import ImageRecord, load_records
from .io_utils import write_json


@dataclass
class CaptionResult:
    image_id: str
    image_path: str
    captions: list[str]
    model_outputs: dict[str, str]
    errors: dict[str, str]


class CaptionBackend:
    name: str

    def caption(self, image_path: Path) -> str:
        raise NotImplementedError


class Blip2CaptionBackend(CaptionBackend):
    name = "blip2"

    def __init__(self, config: Config):
        torch, Image = _load_vision_dependencies()
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.torch = torch
        self.Image = Image
        self.device = _resolve_device(torch, config.caption_device)
        self.dtype = _resolve_dtype(torch, config.caption_torch_dtype, self.device)
        self.max_new_tokens = config.caption_max_new_tokens
        self.processor = Blip2Processor.from_pretrained(config.blip2_model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            config.blip2_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path) -> str:
        image = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device, self.dtype, image_keys={"pixel_values"})
        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


class Florence2CaptionBackend(CaptionBackend):
    name = "florence2"

    def __init__(self, config: Config):
        torch, Image = _load_vision_dependencies()
        _check_florence2_transformers_version()
        _patch_florence2_remote_compatibility()
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.Image = Image
        self.device = _resolve_device(torch, config.caption_device)
        self.dtype = _resolve_dtype(torch, config.caption_torch_dtype, self.device)
        self.max_new_tokens = config.caption_max_new_tokens
        self.num_beams = config.caption_num_beams
        self.task = "<CAPTION>"
        self.processor = AutoProcessor.from_pretrained(config.florence2_model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.florence2_model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: Path) -> str:
        image = self.Image.open(image_path).convert("RGB")
        image = _make_square_image(image, size=768)
        inputs = self.processor(text=self.task, images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device, self.dtype, image_keys={"pixel_values"})
        with self.torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=self.task,
            image_size=image.size,
        )
        return str(parsed.get(self.task, generated_text)).strip()


def generate_captions(config: Config, max_images: int | None = None, models: str | None = None) -> dict[str, Any]:
    records = load_records(config)
    limit = max_images if max_images is not None else config.max_images
    selected_records = records[:limit]
    model_names = _parse_model_names(models or config.caption_models)

    captions_by_image: dict[str, list[str]] = {record.image_id: [] for record in selected_records}
    details: dict[str, dict[str, Any]] = {
        record.image_id: {"image_path": str(record.image_path), "model_outputs": {}, "errors": {}}
        for record in selected_records
    }
    summary = {
        "image_count": len(selected_records),
        "caption_count": 0,
        "models": model_names,
        "loaded_models": [],
        "model_errors": {},
        "output_file": config.caption_output_file,
    }

    for model_name in model_names:
        try:
            backend = make_caption_backend(model_name, config)
        except Exception as exc:
            summary["model_errors"][model_name] = f"{type(exc).__name__}: {exc}"
            continue
        summary["loaded_models"].append(backend.name)
        try:
            for record in selected_records:
                try:
                    caption = backend.caption(record.image_path)
                except Exception as exc:  # keep batch generation moving and record failures.
                    details[record.image_id]["errors"][backend.name] = f"{type(exc).__name__}: {exc}"
                    continue
                if caption:
                    captions_by_image[record.image_id].append(caption)
                    details[record.image_id]["model_outputs"][backend.name] = caption
                    summary["caption_count"] += 1
        finally:
            _release_backend(backend)

    output = {"metadata": summary, "captions": captions_by_image, "details": details}
    write_json(Path(config.caption_output_file), output)
    return output


def generate_captions_for_records(
    records: list[ImageRecord],
    backends: list[CaptionBackend],
    config: Config,
) -> dict[str, Any]:
    captions_by_image: dict[str, list[str]] = {}
    details: dict[str, dict[str, Any]] = {}
    summary = {
        "image_count": 0,
        "caption_count": 0,
        "models": [backend.name for backend in backends],
        "output_file": config.caption_output_file,
    }

    for record in records:
        captions: list[str] = []
        model_outputs: dict[str, str] = {}
        errors: dict[str, str] = {}
        for backend in backends:
            try:
                caption = backend.caption(record.image_path)
            except Exception as exc:  # keep batch generation moving and record failures.
                errors[backend.name] = f"{type(exc).__name__}: {exc}"
                continue
            if caption:
                captions.append(caption)
                model_outputs[backend.name] = caption

        captions_by_image[record.image_id] = captions
        details[record.image_id] = {
            "image_path": str(record.image_path),
            "model_outputs": model_outputs,
            "errors": errors,
        }
        summary["image_count"] += 1
        summary["caption_count"] += len(captions)

    output = {"metadata": summary, "captions": captions_by_image, "details": details}
    write_json(Path(config.caption_output_file), output)
    return output


def make_caption_backend(name: str, config: Config) -> CaptionBackend:
    normalized = name.lower().strip()
    if normalized in {"blip2", "blip-2"}:
        return Blip2CaptionBackend(config)
    if normalized in {"florence2", "florence-2"}:
        return Florence2CaptionBackend(config)
    raise ValueError(f"Unknown caption model: {name}")


def _parse_model_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one caption model must be configured")
    return names




def _check_florence2_transformers_version() -> None:
    import transformers

    major = int(transformers.__version__.split(".", 1)[0])
    if major >= 5:
        raise RuntimeError(
            "Florence-2 Hub remote code is not compatible with the installed "
            f"transformers {transformers.__version__} in this environment. "
            "Install the captioning requirements with `transformers<5` "
            "or run `--models blip2`."
        )

def _patch_florence2_remote_compatibility() -> None:
    # Florence-2 Hub remote code expects these legacy attributes. Transformers 5.x
    # no longer exposes them in the same places, so add minimal class defaults
    # before loading with trust_remote_code=True.
    try:
        from transformers import PreTrainedModel, PretrainedConfig
        from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
    except ImportError:
        return
    if not hasattr(PretrainedConfig, "forced_bos_token_id"):
        PretrainedConfig.forced_bos_token_id = None
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
    if not hasattr(RobertaTokenizer, "additional_special_tokens"):
        RobertaTokenizer.additional_special_tokens = []

def _load_vision_dependencies():
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Captioning requires optional dependencies. Install them with "
            "`pip install -r phrase1_sam3/requirements-captioning.txt`."
        ) from exc
    return torch, Image


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(torch: Any, requested: str, device: str) -> Any:
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if requested != "auto":
        raise ValueError(f"Unsupported caption_torch_dtype: {requested}")
    return torch.float16 if device.startswith("cuda") else torch.float32



def _make_square_image(image: Any, size: int = 768) -> Any:
    if image.size == (size, size):
        return image
    return image.resize((size, size))

def _move_inputs(inputs: Any, device: str, dtype: Any, image_keys: set[str]) -> dict[str, Any]:
    moved = {}
    for key, value in dict(inputs).items():
        if key in image_keys:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def _release_backend(backend: CaptionBackend) -> None:
    torch = getattr(backend, "torch", None)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
