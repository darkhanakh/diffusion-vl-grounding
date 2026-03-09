"""Autoregressive VLM baseline for visual grounding.

Wraps Qwen3-VL and InternVL3 for bbox prediction via token generation.
This is the baseline we benchmark against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class GroundingResult:
    """Result of a visual grounding prediction."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized [0, 1]
    latency_ms: float
    num_tokens: int
    raw_output: str


class ARGroundingBaseline:
    """Autoregressive VLM baseline for visual grounding.

    Generates bbox coordinates token-by-token using standard AR decoding.
    Supports Qwen3-VL and InternVL3.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model and processor."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.model.eval()

    def predict(self, image: Image.Image, query: str) -> GroundingResult:
        """Predict bounding box for a referring expression.

        Args:
            image: Input image.
            query: Referring expression (e.g. "the red car on the left").

        Returns:
            GroundingResult with predicted bbox and timing info.
        """
        assert self.model is not None, "Call .load() first"

        prompt = self._format_prompt(query)
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        raw_output = self.processor.decode(generated_ids, skip_special_tokens=True)
        bbox = self._parse_bbox(raw_output)

        return GroundingResult(
            bbox=bbox,
            latency_ms=latency_ms,
            num_tokens=len(generated_ids),
            raw_output=raw_output,
        )

    def _format_prompt(self, query: str) -> str:
        """Format the grounding prompt for the model."""
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
