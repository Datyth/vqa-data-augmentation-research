## Detailed implementation plan


### Goal

Build a reliable **object list + instance masks** artifact for each image by combining:

```
image captioning + existing VQA dataset labels
→ noun phrase candidates
→ filtered SAM 3 prompts
→ SAM 3 masks
→ accepted object list
```

### RULE
- Make everything simple to run test.
- Do not implement exeptional functions that are not required.

### Expected output

```json
{
  "image_id": "example_001",
  "objects": [
    {
      "object_id": "obj_001",
      "source_phrases": ["cup", "red cup"],
      "canonical_name": "cup",
      "mask_id": "mask_001",
      "bbox": [180, 260, 70, 90],
      "confidence": 0.91,
      "sources": ["caption", "vqa_text"],
      "status": "accepted"
    }
  ]
}
```

### Step 1: Data inventory

First verify what the selected VQA dataset actually provides.

| Source | What to check | How it will be used |
| --- | --- | --- |
| Images | image id, image path, split | Input to captioning and SAM 3 |
| Questions | all questions per image | Extract mentioned objects |
| Answers | answer strings and answer frequency | Extract object answers or attribute-object phrases |
| Existing labels | object categories, boxes, masks, if available | Strong candidate prompts or evaluation labels |
| Captions | if already available | Extra noun phrase source |

Decision to make early:

```
Are COCO object annotations allowed as an input source, or only as evaluation labels?
```

If allowed as input, they can provide strong object prompt candidates. If used only for evaluation, they can help measure recall and mask quality.

### Step 2: Generate image captions

Use image captioning to discover objects not mentioned in existing VQA questions.

Candidate model options:

| Model | Use | Strength | Risk |
| --- | --- | --- | --- |
| BLIP-2 | baseline captioning | common, practical, supports captioning and VQA-style tasks | may miss small objects |
| Florence-2 | captioning and object-level visual tasks | supports prompt-based vision tasks such as captioning, detection, grounding | output format must be verified |
| LLaVA-style MLLM | object-focused caption or object listing | flexible instruction following | higher hallucination risk |

Recommended starting setup:

```
Caption model 1: BLIP-2
Caption model 2: Florence-2, if available
Optional: LLaVA-style object listing for comparison only
```

Prompt style for MLLM-based captioning:

```
List only concrete visible objects and object-like regions in the image.
Use short noun phrases.
Avoid abstract concepts, scene mood, and subjective descriptions.
```

### Step 3: Extract noun candidates from captions

From each generated caption, extract noun phrases.

Example:

```
Caption:
"A red cup sits beside a laptop on a desk, with a blue notebook and a pen."

Extracted candidates:
- red cup
- cup
- laptop
- desk
- blue notebook
- notebook
- pen
```

Recommended extraction methods:

| Method | Purpose |
| --- | --- |
| POS tagging | extract simple nouns |
| Dependency parsing | extract noun phrases with modifiers |
| Lemmatization | normalize plural to singular |
| Rule-based cleanup | remove non-object phrases |
| Synonym mapping | merge equivalent terms, such as bike and bicycle |

Candidate representation:

```json
{
  "phrase": "red cup",
  "head": "cup",
  "modifiers": ["red"],
  "source": "caption"
}
```

### Step 4: Extract candidates from VQA questions and answers

Use existing VQA text as a second candidate source.

Examples:

```
Q: What color is the bus?
A: yellow
Candidates:
- bus
- yellow bus
```

```
Q: What is the man holding?
A: umbrella
Candidates:
- man
- umbrella
```

```
Q: Is there a dog on the couch?
A: yes
Candidates:
- dog
- couch
```

Answer handling:

| Answer type | Example | Action |
| --- | --- | --- |
| Object noun | umbrella, bus, dog | add as candidate |
| Attribute | yellow, red, wooden | attach to likely object from question |
| Number | 2, three | do not use as prompt |
| Yes or no | yes, no | use nouns from question only |
| Phrase | tennis racket, traffic light | extract noun phrase |

### Step 5: Add existing dataset labels

If the VQA dataset or linked dataset has object labels, add them as candidates.

Example:

```json
{
  "phrase": "person",
  "source": "dataset_label"
}
```

Recommended source priority:

```
dataset object labels > VQA question nouns > caption nouns > generated-only MLLM nouns
```

If masks or boxes exist, keep them separate from SAM 3 output. They can be used either as:

1. prompt source only, or
2. ground truth for evaluation.

Do not mix these two roles silently.

### Step 6: Merge and normalize candidates

Merge candidates that refer to the same underlying object concept.

Example:

```
caption: red bus
question: bus
dataset label: bus
```

Merged candidate:

```json
{
  "canonical_name": "bus",
  "candidate_phrases": ["bus", "red bus"],
  "sources": ["caption", "question", "dataset_label"],
  "support_count": 3
}
```

Normalization rules:

| Raw term | Canonical term |
| --- | --- |
| people, men, women | person, unless gender is needed |
| bikes | bicycle |
| tv | television |
| cell phone | phone |
| couches | couch |

### Step 7: Filter non-groundable phrases

Reject phrases that SAM 3 should not receive as object prompts.

Reject examples:

```
scene
background
view
image
side
thing
object
beautiful scene
busy street
workspace vibe
productivity
```

Keep examples:

```
person
dog
car
cup
red cup
table
traffic light
blue shirt
wooden bench
```

Mark as risky, not automatically rejected:

```
sky
grass
water
road
street
room
crowd
```

These may be stuff or region concepts rather than object instances.

### Step 8: Rank candidates and choose top-K prompts

Do not send every candidate to SAM 3. Rank candidates first.

Suggested ranking logic:

```
score =
  source_support
  + dataset_label_bonus
  + question_mention_bonus
  + caption_mention_bonus
  + concreteness_bonus
  - ambiguity_penalty
  - generic_phrase_penalty
```

Start with:

```
top_K = 10 prompts per image
```

Then compare:

```
K = 5, 10, 15
```

Choose the K that gives the best trade-off between object yield and cost.

### Step 9: Prepare SAM 3 prompt set

Use both coarse and attribute-level prompts when useful.

Example:

```json
{
  "canonical_name": "cup",
  "prompts": ["cup", "red cup"]
}
```

Prompt strategy:

| Strategy | When to use |
| --- | --- |
| Coarse prompt only | build basic object list |
| Attribute prompt | attribute appears in caption or answer |
| Coarse + attribute | check consistency between prompts |
| Negative or contrastive prompts | optional, for ambiguity checks |

Consistency example:

```
SAM3("cup") → mask A
SAM3("red cup") → mask A
```

This supports the attribute phrase. If the masks disagree, mark the attribute as uncertain.

### Step 10: Run SAM 3 concept segmentation

For each image and selected prompt:

```
SAM3(image, prompt) → instance masks + boxes + scores
```

Raw output to save:

```json
{
  "image_id": "example_001",
  "prompt": "red cup",
  "instances": [
    {
      "mask_id": "raw_mask_001",
      "bbox": [180, 260, 70, 90],
      "score": 0.91
    }
  ]
}
```

Do not treat raw SAM 3 output as final object evidence yet.

### Step 11: Filter and merge masks

Filtering rules:

- Remove masks below confidence threshold.
- Remove masks that are too small or too large.
- Remove masks with implausible shape or area for the prompt.
- Mark masks that cover almost the whole image as risky for object prompts.
- Merge duplicate masks when mask IoU is high.

Suggested initial threshold:

```
confidence >= 0.5 or 0.6
```

Duplicate merging rule:

```
If mask IoU > 0.8, merge masks as the same object candidate.
```

Conflict example:

```
"bus" and "truck" produce the same mask.
```

Do not force a label. Save:

```json
{
  "label_conflict": true,
  "candidate_labels": ["bus", "truck"]
}
```

### Step 12: Save final Phrase 1 artifact

Final per-image object artifact:

```json
{
  "image_id": "example_001",
  "objects": [
    {
      "object_id": "obj_001",
      "canonical_name": "cup",
      "source_phrases": ["cup", "red cup"],
      "mask_id": "mask_001",
      "bbox": [180, 260, 70, 90],
      "confidence": 0.91,
      "sources": ["caption", "vqa_text"],
      "status": "accepted"
    }
  ],
  "rejected_candidates": [],
  "sam3_prompt_log": [],
  "metadata": {
    "caption_model": "BLIP-2 or Florence-2",
    "top_k": 10,
    "mask_threshold": 0.6
  }
}
```

### Prompt-source variants to compare

Run multiple variants before choosing one pipeline.

| Variant | Source setup | Expected strength | Expected risk |
| --- | --- | --- | --- |
| A | Caption only | discovers salient objects | misses small objects |
| B | VQA text only | focuses on task-relevant objects | narrow coverage |
| C | Dataset labels only | clean prompt source | limited vocabulary |
| D | Caption + VQA text | no external object labels needed | more noisy candidates |
| E | Caption + VQA text + dataset labels | strongest practical variant | more duplicates and conflicts |

### Evaluation plan

Start with manual audit, then compute simple metrics.

Suggested sample:

```
30 images for sanity check
100 images for prompt-source comparison
1,000 images for limited scale test
```

Manual audit fields:

| Field | Label |
| --- | --- |
| Prompt is groundable | yes / no |
| Object is present | yes / no |
| Mask quality | good / partial / bad |
| Label is correct | yes / no / uncertain |
| Duplicate mask | yes / no |
| Exhaustive for prompt | yes / no / uncertain |

Metrics:

| Metric | Formula |
| --- | --- |
| Prompt valid rate | valid prompts / total prompts |
| Good mask rate | good masks / total masks |
| Object yield per image | accepted objects / images |
| Duplicate rate | duplicate masks / accepted masks |
| Exhaustivity score | exhaustive prompt-image pairs / positive prompt-image pairs |
| Cost per verified object | runtime or prompt count / accepted objects |

If ground-truth boxes or masks are available, add:

| Metric | Use |
| --- | --- |
| Mask IoU | compare predicted mask with GT mask |
| Object recall | measure how many GT objects are recovered |
| Object precision | measure how many predicted objects are correct |
| AP50 | detection or segmentation quality at IoU threshold 0.5 |

### Milestones

| Milestone | Goal | Output |
| --- | --- | --- |
| M1 | Data inventory | list available fields and allowed labels |
| M2 | Candidate extraction prototype | noun candidates per image |
| M3 | Filtering and ranking | top-K prompts per image |
| M4 | SAM 3 inference on 30 images | raw masks and scores |
| M5 | Mask post-processing | accepted object list |
| M6 | Compare prompt-source variants | metrics table |
| M7 | Scale to 1,000 images | limited-run artifact and failure report |

### Failure cases to log

| Failure | Example | Action |
| --- | --- | --- |
| Caption hallucination | caption says dog but no dog exists | reject after no mask or audit |
| Generic phrase | background, scene, object | filter before SAM 3 |
| Attribute conflict | red cup and blue cup hit same mask | mark attribute uncertain |
| Duplicate masks | bike and bicycle hit same object | merge |
| Over-segmentation | one object split into many masks | merge or mark partial |
| Under-segmentation | many objects merged into one mask | mark bad or partial |
| Missing small objects | pen, fork, remote | log category-level miss |
| Stuff/object confusion | sky, road, grass | mark as region or reject for object list |

### Recommended starting configuration

| Component | Starting choice |
| --- | --- |
| Caption model | BLIP-2 baseline, Florence-2 as alternative |
| Dataset text | VQA questions + answers |
| Existing labels | COCO categories if allowed |
| Candidate extraction | noun phrase parser + simple rules |
| Prompt K | 10 per image |
| SAM 3 threshold | start at 0.5 or 0.6, tune by audit |
| Sanity sample | 30 images |
| Audit sample | 100 images |
| Main metrics | prompt valid rate, good mask rate, object yield, duplicate rate, cost per verified object |

### First thing to verify

The first verification target is **candidate recall**:

```
Do image captions + VQA labels produce enough correct object prompts for SAM 3?
```

If candidate generation misses important objects, SAM 3 cannot recover them because it only segments concepts that were prompted.