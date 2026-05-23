from __future__ import annotations

import hashlib
import struct
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .text import canonicalize


@dataclass
class SamInstance:
    mask_id: str
    bbox: list[int]
    score: float
    area: int
    image_area: int
    mask_path: str = ""
    sam_box: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        data = {
            "mask_id": self.mask_id,
            "bbox": self.bbox,
            "score": self.score,
            "area": self.area,
            "image_area": self.image_area,
        }
        if self.mask_path:
            data["mask_path"] = self.mask_path
        if self.sam_box is not None:
            data["sam_box"] = self.sam_box
        return data


class SamBackend:
    def segment(self, image_path: Path, prompt: str) -> list[SamInstance]:
        raise NotImplementedError


class DryRunSamBackend(SamBackend):
    """Dependency-free backend that makes deterministic boxes for pipeline testing."""

    def segment(self, image_path: Path, prompt: str) -> list[SamInstance]:
        width, height = image_size(image_path)
        prompt_key = canonicalize(prompt)
        digest = hashlib.sha256(f"{image_path}:{prompt_key}".encode("utf-8")).digest()
        box_w = max(12, int(width * (0.12 + digest[0] / 2550)))
        box_h = max(12, int(height * (0.12 + digest[1] / 2550)))
        max_x = max(1, width - box_w)
        max_y = max(1, height - box_h)
        x = int(digest[2] / 255 * max_x)
        y = int(digest[3] / 255 * max_y)
        score = 0.62 + (digest[4] / 255) * 0.33
        area = box_w * box_h
        mask_hash = hashlib.md5(f"{prompt_key}:{x}:{y}:{box_w}:{box_h}".encode("utf-8")).hexdigest()[:10]
        return [
            SamInstance(
                mask_id=f"dry_{mask_hash}",
                bbox=[x, y, box_w, box_h],
                score=round(score, 3),
                area=area,
                image_area=width * height,
            )
        ]


class NoOpSamBackend(SamBackend):
    def segment(self, image_path: Path, prompt: str) -> list[SamInstance]:
        return []


class Sam3Backend(SamBackend):
    def __init__(self, mask_dir: Path):
        self.mask_dir = mask_dir
        self._processor: Any = None
        self._state: Any = None
        self._image_path: Path | None = None
        self._image_area = 0

    def segment(self, image_path: Path, prompt: str) -> list[SamInstance]:
        processor = self._get_processor()
        state = self._set_image(processor, image_path)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("SAM3 requires PyTorch in the active environment.") from exc

        with torch.inference_mode(), _sam3_autocast(torch):
            output = processor.set_text_prompt(state=state, prompt=prompt)
        return self._instances_from_output(image_path, prompt, output)

    def _get_processor(self) -> Any:
        if self._processor is None:
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
            warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning)
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except ImportError as exc:
                raise RuntimeError(
                    "sam_backend=sam3 needs the official facebookresearch/sam3 package. "
                    "Install SAM3 in a Python 3.12 environment first."
                ) from exc

            try:
                self._processor = Sam3Processor(build_sam3_image_model())
            except Exception as exc:
                raise RuntimeError(
                    "Failed to build the SAM3 image model. Make sure Hugging Face checkpoint "
                    "access is approved and the active environment is authenticated."
                ) from exc
        return self._processor

    def _set_image(self, processor: Any, image_path: Path) -> Any:
        if self._image_path != image_path:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("SAM3 image inference requires Pillow.") from exc

            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("SAM3 requires PyTorch in the active environment.") from exc
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                self._image_area = rgb_image.width * rgb_image.height
                with _sam3_autocast(torch):
                    self._state = processor.set_image(rgb_image)
            self._image_path = image_path
        return self._state

    def _instances_from_output(self, image_path: Path, prompt: str, output: dict[str, Any]) -> list[SamInstance]:
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None:
            return []

        instances = []
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        for index, mask in enumerate(masks):
            mask_2d = mask.detach().to("cpu").bool().squeeze()
            if getattr(mask_2d, "ndim", 0) != 2:
                continue
            area = int(mask_2d.sum().item())
            if area <= 0:
                continue
            mask_id = _sam3_mask_id(image_path, prompt, index)
            mask_path = self.mask_dir / f"{mask_id}.png"
            _write_mask_png(mask_path, mask_2d)
            instances.append(
                SamInstance(
                    mask_id=mask_id,
                    bbox=_bbox_from_mask(mask_2d),
                    score=round(_tensor_value(scores, index), 3),
                    area=area,
                    image_area=self._image_area or int(mask_2d.numel()),
                    mask_path=str(mask_path),
                    sam_box=_tensor_box(boxes, index),
                )
            )
        return instances


def make_sam_backend(name: str, output_dir: Path | None = None) -> SamBackend:
    normalized = name.lower().strip()
    if normalized == "dryrun":
        return DryRunSamBackend()
    if normalized in {"none", "noop"}:
        return NoOpSamBackend()
    if normalized == "sam3":
        mask_dir = (output_dir or Path("phrase1_sam3/outputs")) / "masks"
        return Sam3Backend(mask_dir)
    raise ValueError(f"Unknown SAM backend: {name}")



def _sam3_autocast(torch_module: Any) -> Any:
    if torch_module.cuda.is_available():
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return nullcontext()


def _sam3_mask_id(image_path: Path, prompt: str, index: int) -> str:
    digest = hashlib.sha256(f"{image_path}:{prompt}:{index}".encode("utf-8")).hexdigest()[:16]
    return f"sam3_{digest}"


def _bbox_from_mask(mask: Any) -> list[int]:
    rows, cols = mask.nonzero(as_tuple=True)
    x1 = int(cols.min().item())
    y1 = int(rows.min().item())
    x2 = int(cols.max().item()) + 1
    y2 = int(rows.max().item()) + 1
    return [x1, y1, x2 - x1, y2 - y1]


def _tensor_value(values: Any, index: int) -> float:
    if values is None:
        return 0.0
    value = values[index]
    return float(value.detach().to("cpu").item()) if hasattr(value, "detach") else float(value)


def _tensor_box(values: Any, index: int) -> list[float] | None:
    if values is None:
        return None
    value = values[index]
    raw = value.detach().to("cpu").flatten().tolist() if hasattr(value, "detach") else list(value)
    return [round(float(item), 6) for item in raw]


def _write_mask_png(path: Path, mask: Any) -> None:
    from PIL import Image

    encoded = mask.numpy().astype("uint8") * 255
    Image.fromarray(encoded).save(path)


def image_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 640, 480
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header.startswith(b"\xff\xd8"):
            return _jpeg_size(path)
    return 640, 480


def _jpeg_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        handle.read(2)
        while True:
            marker_start = handle.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                handle.read(3)
                height, width = struct.unpack(">HH", handle.read(4))
                return int(width), int(height)
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            handle.seek(length - 2, 1)
    return 640, 480
