# Phrase 2 DAM Object Descriptions

This package builds **Phrase 2: DAM localized object descriptions** from Phrase 1 SAM3 artifacts.

The core method is:

```text
original image + single-object binary mask -> Describe Anything Model -> localized object description
```

Phrase 2 preserves the object identity and visual evidence from Phrase 1. It does not recompute primary boxes, regenerate object ids, infer relations, rank objects, or replace SAM3 mask evidence with text.

## Method

For each accepted object from Phrase 1:

1. Load the original image by `image_id`.
2. Load the object's single binary mask from `mask_path`.
3. Validate required Phrase 1 fields: `image_id`, `object_id`, `mask_path`, and `bbox`.
4. Skip invalid objects with a log message instead of crashing the run.
5. Skip objects with missing masks as `missing_mask`.
6. Skip masks with zero foreground pixels as `empty_mask`.
7. Send the original image, object mask, and DAM prompt to DAM.
8. Attach the returned localized description and DAM metadata to the object.
9. Preserve Phrase 1 `object_id`, `bbox`, `mask_id`, `mask_path`, `confidence`, `source_phrases`, and `sources`.

The default DAM prompt is:

```text
Describe the masked object in detail. Focus only on the visible object inside the mask. Mention its color, material, shape, texture, visible parts, pose, state, and distinctive visual attributes. Do not describe unrelated background objects.
```

## DAM Backend

The implemented backend uses the official `NVlabs/describe-anything` local Python API:

```python
from dam import DescribeAnythingModel, disable_torch_init

disable_torch_init()
model = DescribeAnythingModel(
    model_path="nvidia/DAM-3B",
    conv_mode="v1",
    prompt_mode="full+focal_crop",
).to(device)

description = model.get_description(
    image,
    mask,
    prompt,
    streaming=False,
    temperature=0.2,
    top_p=0.5,
    num_beams=1,
    max_new_tokens=512,
)
```

This is a local model wrapper, not a hosted cloud API. The package does not clone model weights manually, but the DAM package/checkpoint must be available in the active environment. The official repo also supports an OpenAI-compatible server mode, but this Phase 2 implementation currently uses the local Python API.

Install DAM before real inference:

```bash
conda activate datpt_rs_aug
pip install git+https://github.com/NVlabs/describe-anything
```

## Inputs

Configure the Phrase 1 artifact path in `configs/default.yaml`:

```yaml
artifact_path: "phrase1_sam3/outputs/artifacts.json"
image_dir: "/mnt/VLAI_data/COCO_Images"
mask_dir: ""
```

Supported artifact layouts:

- `phrase1_sam3/outputs/artifacts.json`, grouped by `image_id`
- `phrase1_sam3/outputs/artifacts/<image_id>.json`, one file per image
- a single per-image artifact JSON file

Phrase 2 iterates accepted objects only. Objects with missing required fields are logged and skipped.

## Outputs

Outputs are written separately from Phrase 1:

```text
phrase2_dam/outputs/
├── summary.json
├── descriptions.json
└── all_images.json
```

Each output object contains:

- copied Phrase 1 identity and provenance: `object_id`, `canonical_name`, `source_phrases`, `sources`
- `bbox.xywh`: original Phrase 1 bbox, preserved exactly
- `bbox.xyxy`: derived `[x, y, x + w, y + h]`
- `bbox.normalized_xywh`: bbox normalized by original image width and height
- `mask`: `mask_id`, `mask_path`, `area_pixels`, `area_fraction`
- `sam3`: SAM3 prompt when available and `confidence`
- `dam`: description, DAM prompt, model id, status, and optional error

Description statuses:

- `success`
- `missing_mask`
- `empty_mask`
- `dam_error`
- `filtered`

`summary.json` records total images, total objects, status counts, model name, prompt, timestamp, and skipped artifact metadata.

## Run

Use the project conda environment:

```bash
conda run -n datpt_rs_aug python -m phrase2_dam --config phrase2_dam/configs/default.yaml
```

Useful overrides:

```bash
conda run -n datpt_rs_aug python -m phrase2_dam \
  --config phrase2_dam/configs/default.yaml \
  --artifact-path phrase1_sam3/outputs/artifacts.json \
  --image-dir /mnt/VLAI_data/COCO_Images \
  --mask-dir phrase1_sam3/outputs/masks \
  --output-dir phrase2_dam/outputs \
  --device cuda
```

The CLI prints a brief summary with images processed, objects processed, successes, and failure counts by status.

## Implemented Modules

- `config.py`: central config dataclass, default prompt, generation parameters, and simple YAML-style config parser.
- `artifacts.py`: Phrase 1 artifact reader, grouped/per-image support, accepted-object filtering, and required field validation.
- `dam_wrapper.py`: official DAM local inference wrapper with lazy import and per-object exception handling.
- `pipeline.py`: image/mask loading, mask validation, DAM calls, enrichment schema creation, summary creation, and output writing.
- `cli.py`: single entry point for running M1 -> M3 -> M4.
- `tests/test_pipeline.py`: lightweight tests using a fake DAM wrapper so CI/smoke tests do not require the 3B model.

## Verification

These checks passed in `datpt_rs_aug`:

```bash
conda run -n datpt_rs_aug python -m unittest discover -s phrase2_dam/tests
conda run -n datpt_rs_aug python -m unittest discover -s phrase1_sam3/tests
conda run -n datpt_rs_aug python -m compileall phrase2_dam
```

