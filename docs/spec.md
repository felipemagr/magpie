# magpie: spec

## What it is
A local-first CLI that sorts the images in a folder into up to 5 user-defined
categories. It uses a Jev-style "System One" decision model (typed choice with
probabilities, no free-text generation) as the decision engine.

Images are first turned into text by a small local vision model. That text,
plus EXIF metadata, becomes the `state` for the decision model, which picks a
category.

## Core principles
- 100% local. No image, caption or metadata ever leaves the machine.
  No cloud APIs.
- Lightweight: must run on a normal laptop CPU. A GPU (CUDA / Apple MPS) is
  used if available but never required.
- Safe: never delete or overwrite user files. Every file operation is a
  dry-run unless `--yes` is passed, and is logged so it can be undone.
- Fast on re-runs: captions and decisions are cached by file hash.
- Simple: max 5 categories + an implicit "other" bucket. No training step
  required for the MVP.

## Pipeline (per image)
1. Hash the file (SHA-256) and check the cache. Skip if already processed with
   the same config.
2. Read metadata (pyexiftool, or Pillow + pillow-heif for HEIC): capture date,
   camera make/model, software, width/height, has_gps, filename.
3. Apply deterministic rules from the config (see below). If a rule matches,
   assign the category and stop.
4. Caption: run a small local vision-language model to produce a 1–2 sentence
   English description. Prompt it to always state:
   (a) the kind of image (photo, screenshot, document, drawing, meme…),
   (b) the main subject,
   (c) any clearly visible text (short).
   Example output: "A photo of a supermarket receipt on a wooden table. Visible
   text: 'TOTAL 23.40 EUR'."
5. Build the decision `state` as plain text:
   ```
   Image description: <caption>
   Metadata: taken 2025-04-12, camera: Apple iPhone 15, 4032x3024, no editing
   software, has GPS. Filename: IMG_1234.HEIC
   ```
6. Decide: send the state plus a single `choice` question to the decision
   backend. The options are the user's categories (name + description) plus
   "other: none of the above". Get a probability for each option.
7. If the top probability is at least the confidence threshold, assign that
   category. Otherwise mark the image as "needs review".

## Models (all configurable, all local)

### Captioner
A small VLM, loaded via Hugging Face transformers (or llama.cpp GGUF if
simpler). Start by evaluating small models such as Moondream, SmolVLM and
Florence-2. Pick one based on speed and caption quality on a sample set, and
verify the current model IDs and licenses on Hugging Face before hardcoding.
Resize images so the long side is at most 768 px before captioning.

### Decision backend (interface: `decide(state, options) -> dict[option, prob]`)
1. `laya` (default): Laya (github.com/NandhaKishorM/laya, PyPI `laya`,
   Apache-2.0). ModernBERT-based, 421M parameters, text-only, runs on CPU.
   Use its `choice` primitive. Read the official docs for the exact Python API;
   do not guess it.
   Known limits: 1,024-token input; options share about 256 tokens, so keep
   category descriptions short (5 categories is fine); base checkpoints are
   weak zero-shot, so validate on real data.
2. `semif` (fallback): SemIf-style readout (formerly OpenJev, openjev.com).
   Use llama.cpp (llama-cpp-python) with a small instruct model in GGUF
   (default: Qwen3 0.6B; optional: a larger one). Prompt the model with the
   state, the question and lettered options (A–F), then read the next-token
   logprobs of the letter tokens and softmax over only the allowed letters.
   No text generation.
   Order bias mitigation: evaluate 2 permutations of the option order and
   average the probability per category, not per letter.

Both backends must return the same shape, so they are interchangeable via
config or the `--backend` flag.

## Config file: `magpie.toml` (in the target folder)
```toml
[settings]
backend = "laya"            # or "semif"
threshold = 0.6             # confidence needed to auto-assign
language = "en"             # captions and decisions are in English internally

[[category]]
name = "receipts"
description = "a receipt, invoice or bill"

[[category]]
name = "screenshots"
description = "a screenshot of a phone or computer screen"
rules = { no_camera = true }    # optional deterministic rule

[[category]]
name = "pets"
description = "a photo of a dog, cat or other pet"

[[category]]
name = "japan-trip"
description = "travel photos"
rules = { date_from = "2025-04-01", date_to = "2025-04-20" }
```
Validation: 1 to 5 categories, unique names, names safe for use as folder
names. "other" is reserved and always added automatically.

Supported rules for the MVP: `no_camera`, `date_from` / `date_to`,
`camera_contains`, `filename_contains`, `min_width` / `min_height`.
Rules are evaluated in category order; the first match wins.

## CLI commands (Typer + Rich)
- `magpie init <dir>`: interactive wizard that asks for up to 5 categories
  (name + short description), writes `magpie.toml` and checks that the models
  can be loaded (downloading them on first run with a progress bar).
- `magpie run <dir>`: runs the full pipeline with a progress bar and ETA, then
  prints a summary table (count per category, count needing review).
- `magpie review <dir>`: goes through the "needs review" images one by one.
  Shows the image in the terminal (textual-image or chafa; fall back to
  opening the system viewer), the caption and the probabilities. The user
  picks a category by number, `s` to skip or `o` for other.
- `magpie show <dir> [--category X]`: lists results with caption and
  confidence.
- `magpie apply <dir> --mode copy|move|symlink|csv`: dry-run by default and
  prints the planned operations; `--yes` executes them. Creates
  `<dir>/magpie-output/<category>/` for copy, move and symlink. Writes an
  operation log to `.magpie/ops.jsonl`.
- `magpie undo <dir>`: reverts the last `apply` using the log.

Global flags: `--backend`, `--threshold`, `--verbose`, `--device cpu|cuda|mps`.

## Storage
`<dir>/.magpie/` contains:
- `cache.sqlite` with tables `images(hash, path, metadata_json)`,
  `captions(hash, model_id, caption)` and
  `decisions(hash, config_hash, backend, probs_json, category, status)`
- `ops.jsonl`, the apply/undo log

Changing categories invalidates decisions but not captions, which are the
expensive part.

## Tech stack
Python 3.11+, uv, Typer, Rich, Pillow, pillow-heif, pyexiftool (fall back to
Pillow EXIF if exiftool is not installed), transformers + torch (captioner),
laya, llama-cpp-python (semif backend), SQLite (stdlib), textual-image or
chafa for terminal previews, pytest.
