"""Diffusion VLM wrapper for visual grounding.

Wraps Dream-VL and LaViDa for bbox prediction via parallel denoising.
This is the experimental model we're benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from PIL import Image

from .ar_baseline import GroundingResult


class DiffusionVLMGrounding:
    """Diffusion VLM for visual grounding.

    Generates bbox output (reasoning + coordinates) via iterative denoising.
    All tokens are generated in parallel, refined over K steps.
    """

    def __init__(
        self,
        model_name: str = "DreamLM/Dream-VL-7B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_denoise_steps: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.num_denoise_steps = num_denoise_steps
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load diffusion VLM and processor.

        TODO: Implement once Dream-VL weights/API are finalized.
        The loading will differ from AR models — diffusion VLMs use
        a different generation pipeline.
        """
        raise NotImplementedError(
            f"Loading {self.model_name} — implement once weights are available. "
            "Check https://github.com/DreamLM/Dream-VLX for updates."
        )

    def predict(
        self,
        image: Image.Image,
        query: str,
        num_steps: int | None = None,
    ) -> GroundingResult:
        """Predict bounding box using diffusion-based parallel decoding.

        Args:
            image: Input image.
            query: Referring expression.
            num_steps: Override default denoising steps (for ablation).

        Returns:
            GroundingResult with predicted bbox and timing info.
        """
        assert self.model is not None, "Call .load() first"

        steps = num_steps or self.num_denoise_steps
        prompt = self._format_prompt(query)

        # TODO: Replace with actual Dream-VL / LaViDa inference
        # The key difference from AR: instead of sequential token generation,
        # we initialize a full sequence of masked/noisy tokens and iteratively
        # denoise them in parallel over `steps` forward passes.
        #
        # Pseudocode:
        #   tokens = initialize_masked_sequence(max_len=64)
        #   for step in range(steps):
        #       tokens = model.denoise_step(tokens, image_features, step)
        #   output = decode(tokens)

        start = time.perf_counter()
        # Placeholder for actual diffusion inference
        raw_output = "[0, 0, 0, 0]"
        latency_ms = (time.perf_counter() - start) * 1000

        bbox = self._parse_bbox(raw_output)

        return GroundingResult(
            bbox=bbox,
            latency_ms=latency_ms,
            num_tokens=0,  # All tokens generated in parallel
            raw_output=raw_output,
        )

    def predict_with_step_ablation(
        self,
        image: Image.Image,
        query: str,
        step_counts: list[int] = [1, 2, 5, 10, 20],
    ) -> list[tuple[int, GroundingResult]]:
        """Run prediction with varying denoising steps for ablation study.

        Returns list of (num_steps, result) pairs.
        """
        results = []
        for steps in step_counts:
            result = self.predict(image, query, num_steps=steps)
            results.append((steps, result))
        return results

    def _format_prompt(self, query: str) -> str:
        """Format the grounding prompt."""
        return (
            f"Please provide the bounding box coordinates of the region "
            f"this sentence describes: {query}\n"
            f"Output format: [x1, y1, x2, y2] with normalized coordinates [0-1000]."
        )

    @staticmethod
    def _parse_bbox(output: str) -> tuple[float, float, float, float]:
        """Parse bbox coordinates from model output."""
        import re

        numbers = re.findall(r"[\d.]+", output)
        if len(numbers) >= 4:
            coords = [float(n) / 1000.0 for n in numbers[:4]]
            return (coords[0], coords[1], coords[2], coords[3])
        return (0.0, 0.0, 0.0, 0.0)
