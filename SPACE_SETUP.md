# Hugging Face Space setup — VisionAnomaly

## Form fields (copy-paste)

| Field | Value |
|-------|--------|
| **Owner** | `naman1821` |
| **Space name** | `VisionAnomaly` |
| **Short description** | `Industrial defect detection — PatchCore heatmaps on uploaded images (MVTec-AD).` |
| **License** | MIT |
| **SDK** | Gradio |
| **Template** | Blank (default) |
| **Hardware** | CPU basic (free) |
| **Visibility** | Public |

> **Do not use `VisionAnamoly`** — typo. Use **`VisionAnomaly`** (matches GitHub).

## After creating the Space

### Option A — Sync from GitHub (recommended)

1. Space → **Settings** → **Repository** → link `Naman1821/VisionAnomaly`
2. Ensure repo has root `app.py` + README YAML frontmatter (already added)
3. Space rebuilds automatically

### Option B — Push to HF directly

```bash
pip install huggingface_hub
huggingface-cli login
cd VisionAnomaly
git remote add space https://huggingface.co/spaces/naman1821/VisionAnomaly
git push space main
```

## First launch

- First visitor waits **~1–3 min** (downloads WideResNet weights + trains toy PatchCore)
- Later requests are fast until Space sleeps

## Live URL (after deploy)

https://huggingface.co/spaces/naman1821/VisionAnomaly

Put this on your resume next to GitHub.
