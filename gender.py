from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort


class GenderClassifier:
    MODEL_REPO = "onnx-community/gender-classification"
    MODEL_FILENAME = "gender_mobilev3_small_int8.onnx"
    PREFERRED_LOCAL_FILES = [
        "gender_mobilev3_small_int8.onnx",
        "gender_mobilev3_small.onnx",
        "gender_mobilenetv2_int8.onnx",
        "model_int8.onnx",
        "model.onnx",
    ]
    LABELS = ("female", "male")

    def __init__(self, model_dir: Optional[Path] = None, model_path: Optional[Path] = None) -> None:
        models_dir = model_dir or Path(__file__).with_name("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        resolved_path: Optional[Path] = None
        if model_path is not None:
            candidate = Path(model_path)
            if candidate.exists():
                resolved_path = candidate
            else:
                raise FileNotFoundError(f"Specified gender classifier not found: {candidate}")

        if resolved_path is None:
            for candidate_name in self.PREFERRED_LOCAL_FILES:
                candidate = models_dir / candidate_name
                if candidate.exists():
                    resolved_path = candidate
                    break

        if resolved_path is None:
            fallback_candidates = sorted(
                models_dir.glob("*gender*.onnx"),
                key=lambda path: path.name.lower(),
            )
            if fallback_candidates:
                resolved_path = fallback_candidates[0]

        if resolved_path is None:
            try:
                resolved_path = Path(
                    hf_hub_download(
                        repo_id=self.MODEL_REPO,
                        filename=self.MODEL_FILENAME,
                        cache_dir=str(models_dir),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load gender classifier. Download a compact gender classification "
                    "ONNX model (e.g. 'gender_mobilev3_small_int8.onnx') and place it inside the "
                    "'models/' directory."
                ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = ort.InferenceSession(str(resolved_path), sess_options=options, providers=["CPUExecutionProvider"])

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        default_size = 224
        self.input_height = default_size
        self.input_width = default_size
        try:
            if len(shape) >= 4:
                if isinstance(shape[2], int) and shape[2] > 0:
                    self.input_height = int(shape[2])
                if isinstance(shape[3], int) and shape[3] > 0:
                    self.input_width = int(shape[3])
        except (TypeError, IndexError):
            pass

        output_meta = self.session.get_outputs()[0]
        self.output_name = output_meta.name
        self._rescale = 1.0 / 255.0
        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._convert_to_rgb = True

        config_path = models_dir / "preprocessor_config.json"
        if not config_path.exists():
            alt_config = resolved_path.with_name("preprocessor_config.json")
            if alt_config.exists():
                config_path = alt_config
        if config_path.exists():
            try:
                import json

                with config_path.open("r", encoding="utf-8") as stream:
                    config = json.load(stream)
                size = config.get("size") or {}
                height = size.get("height")
                width = size.get("width")
                if isinstance(height, int) and height > 0:
                    self.input_height = height
                if isinstance(width, int) and width > 0:
                    self.input_width = width
                if bool(config.get("do_rescale", True)):
                    factor = config.get("rescale_factor")
                    if isinstance(factor, (float, int)) and factor > 0:
                        self._rescale = float(factor)
                if bool(config.get("do_normalize", True)):
                    mean = config.get("image_mean")
                    std = config.get("image_std")
                    if isinstance(mean, (list, tuple)) and len(mean) == 3:
                        self._mean = np.array(mean, dtype=np.float32)
                    if isinstance(std, (list, tuple)) and len(std) == 3:
                        self._std = np.array(std, dtype=np.float32)
                convert_rgb = config.get("do_convert_rgb")
                if convert_rgb is False:
                    self._convert_to_rgb = False
            except Exception:
                pass

    def warmup(self, runs: int = 1) -> None:
        dummy = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        for _ in range(max(0, runs)):
            try:
                self.session.run([self.output_name], {self.input_name: dummy})
            except Exception:
                break

    def classify(self, face: np.ndarray | Image.Image) -> Tuple[str, float, np.ndarray]:
        tensor = self._prepare(face)
        raw_output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        if raw_output.ndim == 2:
            raw_output = raw_output[0]
        probabilities = self._softmax(raw_output.astype(np.float32))
        best_idx = int(np.argmax(probabilities))
        label = self.LABELS[best_idx] if 0 <= best_idx < len(self.LABELS) else "unknown"
        confidence = float(probabilities[best_idx]) if label != "unknown" else 0.0
        return label, confidence, probabilities

    def _prepare(self, face: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(face, np.ndarray):
            if self._convert_to_rgb:
                image = Image.fromarray(face.astype("uint8"), mode="RGB")
            else:
                image = Image.fromarray(face.astype("uint8"))
        else:
            image = face

        if self._convert_to_rgb:
            image = image.convert("RGB")
        resized = image.resize((self.input_width, self.input_height), Image.LANCZOS)
        array = np.asarray(resized, dtype=np.float32)
        if self._rescale != 1.0:
            array = array * self._rescale
        array = (array - self._mean) / self._std
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0).astype(np.float32)
        return array

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        total = np.sum(exp)
        if total <= 0:
            return np.full(len(self.LABELS), 0.5, dtype=np.float32)
        return (exp / total).astype(np.float32)


class GenderSmoother:
    LABELS = ("female", "male")

    def __init__(self, alpha: float = 0.65, min_confidence: float = 0.55) -> None:
        self.alpha = alpha
        self.min_confidence = min_confidence
        self._probabilities: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._probabilities = None

    def update(self, probabilities: Optional[np.ndarray]) -> Tuple[str, float]:
        if probabilities is None:
            self.reset()
            return "unknown", 0.0

        probs = np.asarray(probabilities, dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[0]
        if probs.shape[0] != len(self.LABELS):
            return "unknown", 0.0

        probs_sum = float(np.sum(probs))
        probs = probs / probs_sum if probs_sum > 0 else np.full_like(probs, 1.0 / len(self.LABELS))

        if self._probabilities is None:
            self._probabilities = probs
        else:
            self._probabilities = self.alpha * probs + (1.0 - self.alpha) * self._probabilities

        best_idx = int(np.argmax(self._probabilities))
        label = self.LABELS[best_idx]
        confidence = float(self._probabilities[best_idx])
        if confidence < self.min_confidence:
            return "unknown", confidence
        return label, confidence


__all__ = ["GenderClassifier", "GenderSmoother"]
