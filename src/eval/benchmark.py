"""Head-to-head benchmark: AR VLM vs Diffusion VLM on visual grounding."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from ..data.refcoco import RefCOCODataset, RefCOCOSample
from ..models.ar_baseline import ARGroundingBaseline, GroundingResult
from ..models.diffusion_vlm import DiffusionVLMGrounding
from .metrics import accuracy_at_threshold, mean_iou


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a model."""

    model_name: str
    dataset: str
    split: str
    num_samples: int
    acc_at_05: float
    acc_at_075: float
    mean_iou: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    mean_tokens: float


def run_benchmark(
    model: ARGroundingBaseline | DiffusionVLMGrounding,
    dataset: RefCOCODataset,
    max_samples: int | None = None,
    output_dir: str | Path = "results",
) -> BenchmarkResult:
    """Run grounding benchmark on a model.

    Args:
        model: Model to evaluate (AR or Diffusion).
        dataset: RefCOCO dataset split.
        max_samples: Limit number of samples (for quick testing).
        output_dir: Directory to save per-sample results.

    Returns:
        Aggregated benchmark metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = dataset.samples[:max_samples] if max_samples else dataset.samples
    predictions: list[tuple[float, float, float, float]] = []
    ground_truths: list[tuple[float, float, float, float]] = []
    latencies: list[float] = []
    token_counts: list[int] = []
    per_sample_results: list[dict] = []

    for sample in tqdm(samples, desc=f"Evaluating {model.model_name}"):
        image = sample.load_image()
        result = model.predict(image, sample.query)

        predictions.append(result.bbox)
        ground_truths.append(sample.bbox)
        latencies.append(result.latency_ms)
        token_counts.append(result.num_tokens)

        per_sample_results.append(
            {
                "image_id": sample.image_id,
                "ann_id": sample.ann_id,
                "query": sample.query,
                "gt_bbox": list(sample.bbox),
                "pred_bbox": list(result.bbox),
                "latency_ms": result.latency_ms,
                "num_tokens": result.num_tokens,
            }
        )

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    benchmark_result = BenchmarkResult(
        model_name=model.model_name,
        dataset=dataset.dataset_name,
        split=dataset.split,
        num_samples=len(samples),
        acc_at_05=accuracy_at_threshold(predictions, ground_truths, 0.5),
        acc_at_075=accuracy_at_threshold(predictions, ground_truths, 0.75),
        mean_iou=mean_iou(predictions, ground_truths),
        mean_latency_ms=sum(latencies) / n if n else 0,
        median_latency_ms=sorted_latencies[n // 2] if n else 0,
        p95_latency_ms=sorted_latencies[int(n * 0.95)] if n else 0,
        mean_tokens=sum(token_counts) / n if n else 0,
    )

    # Save results
    result_file = output_dir / f"{model.model_name.replace('/', '_')}_{dataset.dataset_name}_{dataset.split}.json"
    with open(result_file, "w") as f:
        json.dump(
            {
                "summary": asdict(benchmark_result),
                "per_sample": per_sample_results,
            },
            f,
            indent=2,
        )

    return benchmark_result


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print side-by-side comparison of benchmark results."""
    header = f"{'Model':<30} {'Acc@0.5':>8} {'Acc@0.75':>9} {'mIoU':>6} {'Latency(ms)':>12} {'Tokens':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.model_name:<30} {r.acc_at_05:>8.1%} {r.acc_at_075:>9.1%} "
            f"{r.mean_iou:>6.3f} {r.mean_latency_ms:>12.1f} {r.mean_tokens:>7.1f}"
        )
