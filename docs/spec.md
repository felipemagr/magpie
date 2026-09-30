# magpie: spec

A local-first CLI that sorts the images in a folder into up to 5 user-defined categories.

## Principles
- 100% local. No image or metadata ever leaves the machine. Models are downloaded once, then
  loaded from the local cache.
- Runs on a laptop CPU. CUDA or Apple MPS is used when available, never required.
- Never delete or overwrite user files. `apply` is a dry run unless `--yes`, and every apply is
  logged so `undo` can revert it.
- Re-runs are fast: image embeddings are cached by file content hash. Changing categories costs
  nothing but embedding five short texts.

## How it decides
1. Hash each image (SHA-256). Embed the ones not in the cache with the SigLIP 2 vision tower
   (`google/siglip2-base-patch16-224`, Apache-2.0), in batches of 32. JPEGs decode at reduced size.
2. Embed each category description as "This is a photo of {description}."
3. Image-text logit = scale * cosine + bias, the model's own calibration.
4. Softmax over the category logits plus a fixed `other` logit of -7 (a raw match probability of
   about 0.001). An image that fits no category falls to `other`.
5. Top probability at or above the threshold (default 0.6): assigned. Below: needs review.

Measured on 200 random COCO photos (`make samples`): 34 ms per image on an Apple laptop CPU,
decode included.

## Config: `magpie.toml` in the target folder
```toml
[settings]
threshold = 0.6

[[category]]
name = "pets"
description = "a dog, a cat or another pet"

[[category]]
name = "receipts"
description = "a receipt, invoice or bill"
```
1 to 5 categories, unique names made of `a-z 0-9 - _`. `other` is reserved and always added.

## Commands
- `magpie init <dir>`: prompt for up to 5 categories, write `magpie.toml`.
- `magpie run <dir>`: classify everything, print the count per category.
- `magpie show <dir> [--category X]`: every image with its category and confidence.
- `magpie apply <dir> --mode copy|move|symlink [--yes]`: place confident images in
  `<dir>/magpie-output/<category>/`. Name clashes get a suffix; files already placed are skipped.
- `magpie undo <dir>`: revert the last apply.

Global flags: `--device cpu|cuda|mps`, `--threshold`.

## Storage: `<dir>/.magpie/`
- `cache.sqlite`: `embeddings(hash, model, vector)`.
- `ops.jsonl`: one line per apply, `{"mode", "ops": [[src, dst], ...]}`, paths relative to the
  folder. Written before acting, so an interrupted apply can still be undone.

## Next
- `review`: step through the uncertain images, show a preview and a caption from a small local VLM
  (Moondream, SmolVLM or Florence-2), pick a category by number. Store overrides in the cache.
- Deterministic rules from EXIF and filename (`no_camera`, `date_from`/`date_to`,
  `camera_contains`, `filename_contains`, `min_width`/`min_height`), tried before the model.
- Text-dependent categories ("invoices from X"): caption plus OCR, scored by a zero-shot text
  classifier such as GLiClass v3, only if SigLIP proves insufficient on real data.
- Recursive folders, `--mode csv`.
