# magpie

Sort a folder of images into your own categories. Fully local: nothing leaves your machine.

```bash
make install
uv run magpie init ~/Pictures/inbox     # name up to 5 categories
uv run magpie run ~/Pictures/inbox      # classify and summarise
uv run magpie apply ~/Pictures/inbox    # preview the plan; add --yes to copy
uv run magpie undo ~/Pictures/inbox     # changed your mind
```

magpie scores every image against your category descriptions with
[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224), a zero-shot image-text model.
About 30 ms per image on a laptop CPU, and re-runs reuse cached embeddings.

Try it on 200 random everyday photos:

```bash
make samples
uv run magpie run samples
```

How it works: [`docs/spec.md`](docs/spec.md).
