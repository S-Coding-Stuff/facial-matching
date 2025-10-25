from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download


class FaceDetector:
    MODEL_REPO = "onnx-community/face-detection-yunet"
    MODEL_SUBFOLDER: Optional[str] = None
    MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

    def __init__(self, model_dir: Optional[Path] = None, score_threshold: float = 0.6) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "opencv-contrib-python is required for face detection. Install it via "
                "'pip install opencv-contrib-python'."
            ) from exc

        self.cv2 = cv2
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError(
                "Your OpenCV build does not include FaceDetectorYN. Install "
                "'opencv-contrib-python>=4.8' and ensure it is the importable OpenCV."
            )
        if model_dir is None:
            model_dir = Path(__file__).with_name("models")
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        manual_path = self.model_dir / self.MODEL_FILENAME
        if manual_path.exists():
            self.model_path = manual_path
        else:
            try:
                resolved_path = hf_hub_download(
                    repo_id=self.MODEL_REPO,
                    filename=self.MODEL_FILENAME,
                    subfolder=self.MODEL_SUBFOLDER,
                    cache_dir=str(self.model_dir),
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to download YuNet model automatically. "
                    "Download 'face_detection_yunet_2023mar.onnx' manually (for example from "
                    "https://huggingface.co/onnx-community/face-detection-yunet) and place it in the 'models/' folder."
                ) from exc
            self.model_path = Path(resolved_path)

        if self.model_path.stat().st_size < 1024 * 100:
            raise RuntimeError(
                "YuNet model appears to be invalid (file is unexpectedly small). "
                f"Remove {self.model_path} and download the ONNX manually (e.g. from "
                "https://huggingface.co/onnx-community/face-detection-yunet)."
            )

        try:
            self._detector = cv2.FaceDetectorYN_create(
                str(self.model_path),
                "",
                (0, 0),
                score_threshold,
                0.3,
                5000,
            )
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV failed to load the YuNet ONNX model. Ensure "
                "'opencv-contrib-python>=4.8' is installed and that the model file "
                "is valid (approx. 4.8 MB). Delete "
                f"{self.model_path} and download it manually from https://huggingface.co/onnx-community/face-detection-yunet."
            ) from exc

    def detect(self, frame_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if frame_rgb is None or frame_rgb.size == 0:
            return []

        height, width = frame_rgb.shape[:2]
        if height == 0 or width == 0:
            return []

        frame_bgr = self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((width, height))
        success, faces = self._detector.detect(frame_bgr)
        boxes: List[Tuple[int, int, int, int]] = []
        if not success or faces is None:
            return boxes

        for face in faces:
            x, y, w, h = face[:4]
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            w = int(w)
            h = int(h)
            if w <= 0 or h <= 0:
                continue
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            w = min(w, width - x1)
            h = min(h, height - y1)
            boxes.append((x1, y1, w, h))
        return boxes

    def close(self) -> None:
        self._detector = None


def extract_face_region(
    source: Image.Image,
    face_detector: Optional[FaceDetector],
) -> Image.Image:
    """Return the largest detected face region, or a centered crop if detection fails."""
    if face_detector is None:
        return _center_square_crop(source)

    array = np.array(source.convert("RGB"))
    boxes = face_detector.detect(array)
    if not boxes:
        return _center_square_crop(source)

    x, y, w, h = max(boxes, key=lambda rect: rect[2] * rect[3])
    return source.crop((x, y, x + w, y + h))


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


__all__ = ["FaceDetector", "extract_face_region"]
