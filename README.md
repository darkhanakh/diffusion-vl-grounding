# diffusion-vl-grounding

**Can Diffusion VLMs Ground Faster Than Autoregressive VLMs?**

We fine-tune existing diffusion-based Vision-Language Models (Dream-VL, LaViDa) for visual grounding (text + image → bounding box) and benchmark them against autoregressive VLMs (Qwen3-VL, InternVL3.5). Diffusion VLMs generate all output tokens in parallel — reasoning AND coordinates — which should yield significant speedups on grounding tasks.

## Thesis

Autoregressive VLMs generate bbox outputs token-by-token: first reasoning about spatial layout (~200-500 tokens), then coordinates (~20 tokens). The bottleneck is the sequential reasoning, not the coordinates. Diffusion VLMs generate the entire output in parallel via iterative denoising, potentially **2-5x faster** end-to-end while maintaining grounding accuracy.

**Nobody has benchmarked diffusion VLMs on visual grounding.** This is the gap.

## Architecture

```
                   AR VLM (baseline)              Diffusion VLM (ours)
                   ─────────────────              ────────────────────
Image + Query  →   Sequential tokens         →    Parallel denoising
                   t1 → t2 → ... → tN             [t1, t2, ..., tN] refined together
                   "the table" → "is" → ...        all tokens denoised simultaneously
                   → "[342," → "567," → ...]       over K steps (K << N)

Time:              O(N) forward passes             O(K) forward passes, K ≈ 5-10
```

## Milestones

- [ ] **M1**: Baselines — Eval AR VLMs (Qwen3-VL-7B, InternVL3) on RefCOCO/+/g. Measure Acc@0.5 + latency
- [ ] **M2**: Diffusion VLM grounding — Fine-tune Dream-VL on RefCOCO for bbox output. Measure same metrics
- [ ] **M3**: Head-to-head — Compare AR vs Diffusion: accuracy, latency, throughput. Ablate denoising steps
- [ ] **M4**: Analysis — Where does diffusion help/hurt? Easy vs hard queries, single vs multi-object
- [ ] **M5**: Paper — Write up results, submit to CVPR/ECCV/NeurIPS workshop

## Models

| Model | Type | Why |
|---|---|---|
| **Dream-VL** | Diffusion VLM | Open-source, Qwen2ViT encoder, parallel generation |
| **LaViDa-7B** | Diffusion VLM | NeurIPS 2025, competitive with AR VLMs |
| **Qwen3-VL-7B** | AR VLM (baseline) | Strong grounding, same-scale comparison |
| **InternVL3-8B** | AR VLM (baseline) | SOTA grounding accuracy |

## Datasets

| Dataset | Samples | Task |
|---|---|---|
| RefCOCO | 142K | Single-object referring expression → bbox |
| RefCOCO+ | 141K | No location words, harder language |
| RefCOCOg | 104K | Longer, more complex expressions |

## Setup

```bash
uv sync
uv run python scripts/download_refcoco.py
```

## Project Structure

```
src/
  models/
    ar_baseline.py         # AR VLM wrapper (Qwen3-VL, InternVL3)
    diffusion_vlm.py       # Diffusion VLM wrapper (Dream-VL, LaViDa)
    grounding_adapter.py   # Fine-tuning adapter for grounding task
  data/
    refcoco.py             # RefCOCO/+/g dataset loader
    transforms.py          # Image + bbox augmentations
  eval/
    metrics.py             # IoU, Acc@0.5, Acc@0.75, latency, throughput
    benchmark.py           # Full eval pipeline (AR vs Diffusion)
  training/
    finetune.py            # Fine-tune diffusion VLM on grounding
    lora.py                # LoRA config for efficient fine-tuning
configs/
  ar_baseline.yaml         # AR model eval config
  diffusion_finetune.yaml  # Diffusion VLM fine-tune config
  eval.yaml                # Benchmark config
scripts/
  download_refcoco.py      # Dataset download
  run_baseline.py          # Run AR baseline eval
  run_diffusion.py         # Run diffusion VLM eval
  run_benchmark.py         # Head-to-head comparison
notebooks/
  01_explore_refcoco.ipynb
  02_ar_baseline.ipynb
  03_diffusion_finetune.ipynb
  04_comparison.ipynb
```

## Hardware

| Machine | Use |
|---|---|
| MacBook M4 Pro 48GB | FP16 inference, prototyping, eval |
| RTX 4070 12GB | FP8 inference, fine-tuning (CUDA) |
| Cloud A100 | Full fine-tuning runs, paper benchmarks |

## Related Work

- [Dream-VL](https://github.com/DreamLM/Dream-VLX) — Diffusion VLM with Qwen2ViT encoder
- [LaViDa](https://arxiv.org/abs/2505.16839) — Diffusion VLM, NeurIPS 2025
- [MMaDA](https://github.com/Gen-Verse/MMaDA) — Unified multimodal diffusion LM
- [LG-DVG](https://arxiv.org/abs/2308.09599) — Diffusion for visual grounding (specialized, not VLM)

## License

MIT
