# Phrase 1 SAM 3 Object Artifacts

This package builds the Phase 1 artifact described in `PLAN.md`:

```text
VQA questions/answers + optional captions/labels
-> noun/object candidates
-> ranked SAM prompts
-> raw mask results
-> accepted per-image object artifact
```

Candidate extraction uses spaCy noun chunks. Install the base requirements and `en_core_web_sm` model before running the pipeline.

## Run

```bash
conda activate datpt_rs
pip install -r phrase1_sam3/requirements.txt
python -m spacy download en_core_web_sm
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml inventory
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml run --max-images 5
```

Outputs are written to `phrase1_sam3/outputs/`.

The default `output_layout` is `grouped_json`. This keeps the number of files small for large datasets without putting every intermediate result into one huge JSON file. A `run` command writes:

- `artifacts.json`: final per-image artifacts keyed by `image_id`. This is the clean object list intended for downstream phases.
- `candidates.json`: intermediate object candidates keyed by `image_id`, extracted from VQA text, optional captions, and optional labels before segmentation. Use this to inspect which phrases become SAM prompts.
- `raw_sam.json`: raw per-prompt backend output keyed by `image_id`. With the default `dryrun` backend, these masks/boxes are deterministic placeholders, not real SAM 3 predictions. For real SAM3 runs, this JSON keeps raw instance metadata even when duplicate or rejected PNG masks are pruned from disk.
- `masks/`: final retained SAM3 binary mask PNGs when `sam_backend: sam3` is used. The directory is refreshed at the start of each SAM3 run.
- `summary.json`: run-level summary with image count, prompt count, accepted object count, and key config values.

The `inventory` command writes a separate `inventory.json`, which is only a dataset/config sanity report. It is not model output.

Output flow:

```text
VQA text / optional captions / optional labels
-> candidates.json
-> raw_sam.json
-> artifacts.json
-> summary.json
```

Alternative layouts:

```bash
# Debug layout: many small files, one per image per output type.
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml run --max-images 5 --output-layout per_image

# Tiny experiment layout: one combined run.json. Not recommended for large datasets.
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml run --max-images 5 --output-layout single_json
```

`per_image` writes `summary.json`, `candidates/<image_id>.json`, `raw_sam/<image_id>.json`, and `artifacts/<image_id>.json`.


## Candidate Extraction Method

Candidate extraction is implemented in `text.py`. Its public entry point is:

```python
extract_candidates_from_record(record, source_variant) -> tuple[list[Candidate], list[dict]]
```

The method uses a hybrid open-vocabulary design:

1. spaCy noun chunks provide candidate noun phrases from arbitrary English text.
2. Small domain rules keep VQA and COCO object handling stable.
3. Known object vocabularies influence ranking but do not decide whether an unknown noun is allowed to exist.

This means the pipeline can keep a noun such as `stroller` even though it is not in the project object lists. A known object such as `bench` still receives a higher score and is more likely to become an early SAM prompt.

### Input Sources

Each image record may provide image captions from `captions_file`, VQA questions, VQA answers and annotations, and configured dataset labels. `source_variant` decides which sources are active before extraction:

- `A`: captions
- `B`: VQA questions and answers
- `C`: dataset labels
- `D`: captions plus VQA questions and answers
- `E`: captions plus VQA questions, answers, and dataset labels

The selected source is preserved in each candidate through the `sources` field. A candidate supported by both caption text and VQA text can rank above a candidate seen in only one weak source.

### Noun Phrase Extraction

Caption and VQA text are sent through the lazily loaded spaCy model `en_core_web_sm`. The model is loaded once per Python process and reused for later records.

For each spaCy noun chunk:

1. Chunks whose head token is a pronoun or determiner are skipped.
2. Chunks whose head lemma is a stop word or configured reject term are skipped.
3. The clean phrase keeps adjective and noun lemmas only.
4. The head noun lemma is also added separately when it is usable.

For example:

```text
A stroller is parked near the red bench
```

can produce text candidates similar to:

```text
stroller
red bench
bench
```

The extractor also keeps a regex pass for known multi-word object names in `COMMON_OBJECTS`. This protects phrases such as `traffic light`, `baseball bat`, and `tennis racket` when spaCy chunks them differently from the preferred SAM prompt phrase.

### Canonicalization

Raw phrases are merged into a canonical object name before ranking. Canonicalization first applies domain-specific aliases from `SYNONYMS`, for example:

- `woman`, `man`, `girl`, `children`, and `people` become `person`
- `bike` becomes `bicycle`
- `racket` becomes `tennis racket`
- `surf board` becomes `surfboard`

After those aliases, spaCy lemmatization normalizes noun inflection. This stabilizes plural and irregular plural text:

- `donuts` becomes `donut`
- `mice` becomes `mouse`
- `knives` becomes `knife`

The original surface phrases are still kept in `candidate_phrases`. That field explains which caption, VQA, or label wording contributed to the final `canonical_name`.

### Filtering Policy

The current extractor is open-vocabulary. `COMMON_OBJECTS` is no longer a hard acceptance gate.

`_add_phrase` rejects only phrases whose canonical name is in `REJECT_TERMS`, which covers generic, abstract, or non-groundable heads such as `reason`, `color`, `brand`, and `scene`. Other extracted nouns are kept and left for ranking to prioritize.

Some lists still provide useful control signals:

- `STOP_WORDS` removes function words while phrases are cleaned.
- `ATTRIBUTE_WORDS` helps answer handling avoid treating pure attributes as objects.
- `RISKY_TERMS` marks broadly groundable regions such as `water`, `road`, or `wall` as lower-priority risky candidates.
- `_looks_like_groundable_phrase` remains in code for future soft scoring experiments but is not used as a rejection gate.

### VQA Answer Handling

VQA answer text follows the same extractor, with small safeguards:

- yes/no answers are not converted into object candidates
- numeric answers are skipped
- an attribute answer can be joined with an object phrase from its question

For example, if a question mentions a `cup` and the answer is `red`, the code can add a phrase such as `red cup` instead of treating `red` as a standalone object.

### Ranking

Each canonical candidate accumulates `support_count`, `sources`, `candidate_phrases`, and a ranking `score`. The current score combines mention count, source weights, known-object bonuses, unknown-object penalties, risky-term penalties, and a small short-name bonus.

Source bonuses are:

- dataset label: `+4.0`
- VQA text: `+3.0`
- caption: `+2.0`
- generated-only text: `+1.0`

Vocabulary signals are ranking signals only:

- COCO object in `COCO_OBJECTS`: `+1.5`
- common non-COCO object in `COMMON_OBJECTS`: `+0.75`
- unknown object outside `COMMON_OBJECTS`: `-1.0`
- unknown object with one supporting mention: additional `-1.5`

Single-token canonical names receive a small `+0.25` bonus. Names in `RISKY_TERMS` lose `1.0` and are marked with status `risky`.

With one caption mention, a known `bench` should rank above an unknown `stroller`, but both remain available in `candidates.json`:

```text
bench    -> known COCO object, stronger score
stroller -> unknown object, lower score but still accepted
```

### Candidate To Prompt Flow

Candidates are sorted by descending score, then the pipeline keeps the first `top_k` candidates for the record. For each selected candidate it creates at most two segmentation prompts:

1. the canonical name
2. one alternate source phrase when available

The intermediate `candidates.json` file shows extraction and ranking before segmentation. `raw_sam.json` shows backend results for generated prompts. `artifacts.json` shows accepted objects after mask filtering and duplicate merging.

## SAM Backend

`sam_backend: dryrun` is the default. It creates deterministic placeholder boxes so the full pipeline can be tested before a real SAM3 runtime and checkpoint are available.

Use `sam_backend: none` to produce candidates without masks. Use `sam_backend: sam3` to pass the selected text prompts into the official SAM3 image processor and save real mask predictions.

### Install SAM3

SAM3 should run from the Python 3.12 environment `datpt_rs_aug`. The official SAM3 README requires Python 3.12+, PyTorch 2.7+, and a CUDA-compatible GPU with CUDA 12.6+. It also requires approved Hugging Face checkpoint access before the default model builder can download checkpoints.

The setup used for this project is:

```bash
conda activate datpt_rs_aug
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git /tmp/facebookresearch-sam3
pip install /tmp/facebookresearch-sam3
pip install "setuptools<81" einops pycocotools psutil
pip install -r phrase1_sam3/requirements.txt
python -m spacy download en_core_web_sm
hf auth login
```

The `hf auth login` step must use an account whose access request for the SAM3 checkpoint repository has been approved. After authentication, the SAM3 builder used by this pipeline downloads its default image checkpoint through Hugging Face when the backend first loads.

### Run SAM3

Run the normal Phase 1 command in `datpt_rs_aug` and override the backend:

```bash
conda activate datpt_rs_aug
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml run --max-images 5 --sam-backend sam3
```

SAM3 runs show a `tqdm` progress bar over images. It reports completed images, percent complete, elapsed time, ETA, and images per second without printing one line per prompt:

```text
SAM3 images:  60%|████████████████████▍             | 3/5 [00:18<00:03,  1.56s/image]
```

For each selected candidate, the pipeline still sends at most two text prompts to the backend: the canonical name and one alternate source phrase. The SAM3 backend caches the current image state across those prompts, then returns one instance per SAM3 mask above the processor confidence threshold.

`raw_sam.json` stores per-prompt instance metadata from SAM3:

- `score`: SAM3 confidence score
- `sam_box`: SAM3 box output in pixel `xyxy` coordinates
- `bbox`: `xywh` box derived from the saved mask and used by pipeline filtering; duplicate merging uses mask IoU when PNGs are available and falls back to bbox IoU otherwise
- `area` and `image_area`: mask area bookkeeping used by area filters
- `mask_id`: backend mask identity
- `mask_retained`: `true` when that raw instance is the mask kept by the final artifact, `false` when its PNG was pruned after filtering or duplicate merging
- `mask_path`: PNG path only for retained masks; rejected and merged-away raw instances keep metadata but no mask file path

Real binary masks are saved outside JSON to keep output files manageable. `outputs/masks` is intentionally the filtered final set, not every raw mask SAM3 proposed:

```text
phrase1_sam3/outputs/masks/<sam3_mask_id>.png
```

Accepted objects in `artifacts.json` keep the final artifact `mask_id` and also include `backend_mask_id` plus `mask_path` when a SAM3 mask survives threshold filtering and duplicate merging. Those retained mask PNGs are black/white images aligned to the original image size. This avoids filling `outputs/masks` with low-confidence masks and duplicate masks emitted by alternate prompts.

If SAM3 fails while building the model, verify that `sam3` imports in `datpt_rs_aug`, the GPU PyTorch install works, and Hugging Face checkpoint access and login are complete. See `requirements-sam3.txt` for the compact setup notes.

## Prompt Source Variants

- `A`: captions only
- `B`: VQA questions and answers only
- `C`: dataset labels only
- `D`: captions + VQA text
- `E`: captions + VQA text + dataset labels

The default config uses `D`. If the configured `captions_file` does not exist yet, it naturally behaves like VQA text only.

## Image Captioning

The `caption` command generates captions before candidate extraction. It supports BLIP-2 and Florence-2 through Hugging Face Transformers. These dependencies are optional and are not installed by the base `requirements.txt`.

Install captioning dependencies in an environment with a compatible PyTorch setup:

```bash
pip install -r phrase1_sam3/requirements-captioning.txt
```

Florence-2 currently uses Hugging Face Hub remote code and should be run with Transformers 4.x. If your environment has Transformers 5.x, run BLIP-2 only or install the pinned captioning requirements in a separate environment.

Generate captions for a small sample:

```bash
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 5 --models blip2,florence2
```

By default this writes:

```text
phrase1_sam3/outputs/captions.json
```

The file has a `captions` map keyed by `image_id`, plus `metadata` and per-model `details`. The normal `run` command reads the same file through `captions_file` in the config, so the flow becomes:

```bash
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 5 --models blip2,florence2
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml run --max-images 5
```

Useful overrides:

```bash
# Run only one captioner.
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 5 --models blip2

# Write captions somewhere else, then pass that file to run.
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 5 --output /tmp/captions.json
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml run --max-images 5 --captions-file /tmp/captions.json

# Force CPU or a specific GPU.
python -m phrase1_sam3 --config phrase1_sam3/configs/default.yaml caption --max-images 5 --device cuda:0
```

Default model IDs are configured in `configs/default.yaml`:

- BLIP-2: `Salesforce/blip2-opt-2.7b`
- Florence-2: `microsoft/Florence-2-base`

The caption command loads models sequentially instead of keeping BLIP-2 and Florence-2 in memory at the same time.

## Test

```bash
conda activate datpt_rs
python -m unittest discover phrase1_sam3/tests
```
