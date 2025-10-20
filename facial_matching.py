from tkinter import Tk, Menu, ttk, filedialog, messagebox, StringVar
from pathlib import Path
from typing import Optional, Any, List, Tuple, Callable
import queue
import threading
from dataclasses import dataclass
import json
import urllib.request
import time

import numpy as np
from PIL import Image, ImageTk
from annoy import AnnoyIndex
from huggingface_hub import hf_hub_download
import onnxruntime as ort

DATASET_DIR = Path(__file__).with_name("celebrity_dataset")
SAMPLE_CELEBRITY_URLS = {
    "zendaya": "https://upload.wikimedia.org/wikipedia/commons/0/0d/Zendaya_2019_by_Glenn_Francis.jpg",
    "ryan_gosling": "https://upload.wikimedia.org/wikipedia/commons/4/46/Ryan_gosling_cannes_2014.jpg",
    "emma_watson": "https://upload.wikimedia.org/wikipedia/commons/1/1b/Emma_Watson_2013.jpg",
    "michael_b_jordan": "https://upload.wikimedia.org/wikipedia/commons/5/5d/Michael_B._Jordan_in_2018.jpg",
    "scarlett_johansson": "https://upload.wikimedia.org/wikipedia/commons/9/90/Scarlett_Johansson_by_Gage_Skidmore_2.jpg",
}

SAMPLE_CELEBRITY_GENDERS = {
    "zendaya": "female",
    "ryan_gosling": "male",
    "emma_watson": "female",
    "michael_b_jordan": "male",
    "scarlett_johansson": "female",
}


def format_display_name(raw: str) -> str:
    return raw.replace("_", " ").title()


@dataclass
class CelebrityEntry:
    name: str
    image: Image.Image
    embedding: np.ndarray
    path: Path
    thumbnail: Image.Image
    gender: str = "unknown"
    gender_confidence: float = 0.0


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


class FaceEmbedder:
    MODEL_REPO = "onnx-community/arcface-resnet100"
    DEFAULT_FILENAME = "arcfaceresnet100-8.onnx"
    PREFERRED_LOCAL_FILES = [
        "arcfaceresnet100-11-int8.onnx",
        "arcfaceresnet100-8.onnx",
    ]

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        models_dir = model_dir or Path(__file__).with_name("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        manual_path: Optional[Path] = None
        for filename in self.PREFERRED_LOCAL_FILES:
            candidate = models_dir / filename
            if candidate.exists():
                manual_path = candidate
                break

        if manual_path is not None:
            resolved_path = manual_path
        else:
            try:
                resolved_path = Path(
                    hf_hub_download(
                        repo_id=self.MODEL_REPO,
                        filename=self.DEFAULT_FILENAME,
                        cache_dir=str(models_dir),
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to download ArcFace ONNX model. Download it manually from "
                    "https://huggingface.co/onnx-community/arcface-resnet100 "
                    "and place it in the 'models/' directory."
                ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = ort.InferenceSession(str(resolved_path), sess_options=options, providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.input_height = int(self.input_shape[2]) if len(self.input_shape) > 2 and self.input_shape[2] is not None else 112
        self.input_width = int(self.input_shape[3]) if len(self.input_shape) > 3 and self.input_shape[3] is not None else 112

    def embed(self, face: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(face, np.ndarray):
            image = Image.fromarray(face.astype("uint8"))
        else:
            image = face

        image = image.convert("RGB").resize((112, 112), Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32)
        array = array[:, :, ::-1]  # RGB -> BGR
        array = (array - 127.5) / 128.0
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0).astype(np.float32)

        embedding = self.session.run(None, {self.input_name: array})[0][0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)

    def warmup(self, runs: int = 1) -> None:
        dummy = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
        for _ in range(max(0, runs)):
            try:
                self.session.run(None, {self.input_name: dummy})
            except Exception:
                break


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
        # Default to 224 if the model uses dynamic dimensions.
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

        # Read optional Hugging Face preprocessor config if available.
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


class CelebrityDataset:
    def __init__(self, dataset_dir: Path, url_map: dict[str, str]) -> None:
        self.CACHE_VERSION = 3
        self.dataset_dir = dataset_dir
        self.url_map = url_map
        self.entries: List[CelebrityEntry] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.annoy_index: Optional[AnnoyIndex] = None
        self.gender_indices: dict[str, List[int]] = {"male": [], "female": [], "unknown": []}
        self.gender_centroids: dict[str, np.ndarray] = {}
        self.status_callback: Optional[Callable[[Optional[str], Optional[float]], None]] = None
        self.gender_classifier: Optional[GenderClassifier] = None
        self._dataset_signature: Optional[Tuple[Tuple[str, int, int], ...]] = None
        self._metadata_mtime_ns: Optional[int] = None
        self.cache_dir = self.dataset_dir / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_manifest_path = self.cache_dir / "manifest.json"
        self._cache_embedding_path = self.cache_dir / "embedding_matrix.npy"
        self._cache_annoy_path = self.cache_dir / "index.ann"
        self._load_thread: Optional[threading.Thread] = None
        self._load_thread_lock = threading.Lock()
        self._load_event = threading.Event()
        self._load_event.set()

    def ensure_directory(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def set_status_callback(self, callback: Optional[Callable[[Optional[str], Optional[float]], None]]) -> None:
        self.status_callback = callback

    def set_gender_classifier(self, classifier: Optional[GenderClassifier]) -> None:
        self.gender_classifier = classifier

    def _notify_status(self, message: Optional[str], progress: Optional[float]) -> None:
        if self.status_callback is not None:
            try:
                self.status_callback(message, progress)
            except Exception:
                pass

    def _compute_signature(self) -> Tuple[Tuple[Tuple[str, int, int], ...], Optional[int]]:
        files = self.available_files()
        file_records: List[Tuple[str, int, int]] = []
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            file_records.append((str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)))
        file_records_tuple = tuple(sorted(file_records))
        metadata_path = self.dataset_dir / "metadata.json"
        metadata_mtime = None
        if metadata_path.exists():
            try:
                metadata_mtime = int(metadata_path.stat().st_mtime_ns)
            except OSError:
                metadata_mtime = None
        return file_records_tuple, metadata_mtime

    def _load_cache_manifest(self) -> dict:
        if not self._cache_manifest_path.exists():
            return {}
        try:
            return json.loads(self._cache_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache_manifest(self, manifest: dict) -> None:
        try:
            self._cache_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _cache_name(self, slug: str, suffix: str) -> str:
        return f"{slug}{suffix}"

    def _cache_path(self, slug: str, suffix: str) -> Path:
        return self.cache_dir / self._cache_name(slug, suffix)

    def _load_from_cache_manifest(
        self,
        manifest: dict,
        signature_files: Tuple[Tuple[str, int, int], ...],
        metadata_mtime: Optional[int],
    ) -> bool:
        if (
            manifest.get("signature") != list(signature_files)
            or manifest.get("metadata_mtime") != metadata_mtime
            or manifest.get("cache_version") != self.CACHE_VERSION
        ):
            return False
        embedding_file = manifest.get("embedding_matrix")
        index_file = manifest.get("annoy_index")
        if not embedding_file or not index_file:
            return False
        embedding_path = self.cache_dir / embedding_file
        index_path = self.cache_dir / index_file
        if not embedding_path.exists() or not index_path.exists():
            return False
        order: List[str] = manifest.get("order") or []
        images_meta: dict = manifest.get("images") or {}
        try:
            embedding_matrix = np.load(embedding_path)
        except Exception:
            return False
        if not order or embedding_matrix.shape[0] != len(order):
            return False
        dim = embedding_matrix.shape[1]
        index = AnnoyIndex(dim, "angular")
        try:
            index.load(str(index_path))
        except Exception:
            return False

        new_entries: List[CelebrityEntry] = []
        embeddings: List[np.ndarray] = []

        for idx, slug in enumerate(order):
            meta = images_meta.get(slug)
            if not meta:
                return False
            path = Path(meta.get("path", ""))
            if not path.exists():
                return False
            try:
                with Image.open(path) as img:
                    rgb_image = img.convert("RGB")
            except Exception:
                return False

            thumbnail_name = meta.get("thumbnail")
            if thumbnail_name:
                thumb_path = self.cache_dir / thumbnail_name
            else:
                thumb_path = None
            thumbnail_image: Optional[Image.Image] = None
            if thumb_path and thumb_path.exists():
                try:
                    with Image.open(thumb_path) as thumb_img:
                        thumbnail_image = thumb_img.convert("RGB")
                except Exception:
                    thumbnail_image = None
            if thumbnail_image is None:
                thumbnail_image = rgb_image.copy()
                thumbnail_image.thumbnail((360, 270), Image.LANCZOS)

            embedding = embedding_matrix[idx]
            embeddings.append(embedding)
            entry = CelebrityEntry(
                name=meta.get("name", slug),
                image=rgb_image.copy(),
                embedding=embedding.astype(np.float32),
                path=path,
                thumbnail=thumbnail_image.copy(),
                gender=meta.get("gender", "unknown"),
                gender_confidence=float(meta.get("gender_confidence", 0.0)),
            )
            new_entries.append(entry)

        self.entries = new_entries
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.annoy_index = index
        self.gender_indices = {"male": [], "female": [], "unknown": []}
        for idx, entry in enumerate(self.entries):
            self.gender_indices.setdefault(entry.gender, []).append(idx)

        self.gender_centroids = {}
        if self.embedding_matrix is not None:
            for gender in ("male", "female"):
                idxs = [i for i, entry in enumerate(self.entries) if entry.gender == gender]
                if not idxs:
                    continue
                centroid = self.embedding_matrix[idxs].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self.gender_centroids[gender] = centroid.astype(np.float32)

        self._dataset_signature = signature_files
        self._metadata_mtime_ns = metadata_mtime
        self._notify_status(None, 1.0)
        self._load_event.set()
        return True

    def _load_metadata(self) -> dict:
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def download_samples(self) -> Tuple[int, List[str]]:
        """
        Download a curated set of celebrity portraits. Returns count of saved files and failures.
        """
        self.ensure_directory()
        saved = 0
        failures: List[str] = []

        for name, url in self.url_map.items():
            target = self.dataset_dir / f"{name}.png"
            if target.exists():
                continue
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read()
                target.write_bytes(data)
                saved += 1
            except Exception:
                failures.append(name)

        if saved > 0:
            self.entries.clear()
        return saved, failures

    def available_files(self) -> List[Path]:
        self.ensure_directory()
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(self.dataset_dir.glob(pattern)))
        return files

    def load_entries(
        self,
        face_detector: Optional[FaceDetector],
        face_embedder: FaceEmbedder,
        gender_classifier: Optional[GenderClassifier] = None,
    ) -> None:
        signature_files, metadata_mtime = self._compute_signature()
        manifest = self._load_cache_manifest()
        if self._load_from_cache_manifest(manifest, signature_files, metadata_mtime):
            return

        self._load_event.clear()
        try:
            files = self.available_files()
            entries: List[CelebrityEntry] = []
            embeddings: List[np.ndarray] = []
            total = len(files)

            metadata = self._load_metadata()
            self.gender_indices = {"male": [], "female": [], "unknown": []}

            if total == 0:
                self.entries = []
                self.embedding_matrix = None
                self._notify_status("No celebrity images found.", None)
                return

            self._notify_status(f"Loading celebrity images 0/{total}", 0.0)

            classifier = gender_classifier or self.gender_classifier
            manifest_images: dict = manifest.get("images", {}) if manifest else {}
            updated_manifest_images: dict = {}
            order: List[str] = []

            for path in files:
                try:
                    with Image.open(path) as img:
                        rgb_image = img.convert("RGB")
                except OSError:
                    continue

                slug = path.stem
                stat = path.stat()
                cache_entry = manifest_images.get(slug) if manifest_images else None
                use_cache = (
                    cache_entry is not None
                    and cache_entry.get("size") == int(stat.st_size)
                    and cache_entry.get("mtime_ns") == int(stat.st_mtime_ns)
                )

                face_image: Optional[Image.Image] = None
                embedding: Optional[np.ndarray] = None
                thumbnail: Optional[Image.Image] = None
                cached_gender = "unknown"
                cached_gender_confidence = 0.0

                if use_cache:
                    embedding_name = cache_entry.get("embedding")
                    if embedding_name:
                        embedding_path = self.cache_dir / embedding_name
                        if embedding_path.exists():
                            try:
                                embedding = np.load(embedding_path).astype(np.float32)
                            except Exception:
                                embedding = None
                    face_crop_name = cache_entry.get("face_crop")
                    if face_crop_name:
                        face_crop_path = self.cache_dir / face_crop_name
                        if face_crop_path.exists():
                            try:
                                with Image.open(face_crop_path) as face_img:
                                    face_image = face_img.convert("RGB")
                            except Exception:
                                face_image = None
                    thumbnail_name = cache_entry.get("thumbnail")
                    if thumbnail_name:
                        thumb_path = self.cache_dir / thumbnail_name
                        if thumb_path.exists():
                            try:
                                with Image.open(thumb_path) as thumb_img:
                                    thumbnail = thumb_img.convert("RGB")
                            except Exception:
                                thumbnail = None
                    cached_gender = cache_entry.get("gender", "unknown")
                    cached_gender_confidence = float(cache_entry.get("gender_confidence", 0.0))

                if embedding is None:
                    if face_image is None:
                        face_image = extract_face_region(rgb_image, face_detector)
                    embedding = face_embedder.embed(np.array(face_image))

                if thumbnail is None:
                    thumbnail = rgb_image.copy()
                    thumbnail.thumbnail((360, 270), Image.LANCZOS)

                meta_entry = metadata.get(slug, {})
                gender_source = meta_entry.get("gender") or SAMPLE_CELEBRITY_GENDERS.get(slug, cached_gender)
                gender = (gender_source or "unknown").lower()
                if gender not in ("male", "female"):
                    gender = "unknown"

                predicted_gender = "unknown"
                predicted_confidence = 0.0
                if gender == "unknown" and cached_gender in ("male", "female"):
                    predicted_gender = cached_gender
                    predicted_confidence = cached_gender_confidence
                if classifier is not None and face_image is not None and gender == "unknown":
                    try:
                        predicted_gender, predicted_confidence, _ = classifier.classify(face_image)
                    except Exception:
                        predicted_gender = "unknown"
                        predicted_confidence = 0.0

                if gender == "unknown" and predicted_gender in ("male", "female"):
                    gender = predicted_gender

                if meta_entry.get("gender") in ("male", "female"):
                    gender_confidence = 1.0
                elif gender == predicted_gender:
                    gender_confidence = predicted_confidence
                else:
                    gender_confidence = cached_gender_confidence if use_cache else 0.0

                entry = CelebrityEntry(
                    name=meta_entry.get("name", slug),
                    image=rgb_image.copy(),
                    embedding=embedding,
                    path=path,
                    thumbnail=thumbnail.copy(),
                    gender=gender,
                    gender_confidence=gender_confidence,
                )
                entries.append(entry)
                embeddings.append(embedding)
                order.append(slug)

                face_crop_name = cache_entry.get("face_crop") if use_cache else None
                face_crop_default = self._cache_name(slug, "_face.png")
                face_crop_path = self.cache_dir / (face_crop_name or face_crop_default)
                if face_image is not None and (not face_crop_path.exists() or not use_cache):
                    try:
                        face_image.save(face_crop_path, format="PNG")
                    except Exception:
                        pass
                    face_crop_name = face_crop_path.name
                elif face_crop_path.exists():
                    face_crop_name = face_crop_path.name
                else:
                    face_crop_name = face_crop_default

                thumbnail_name = self._cache_name(slug, "_thumbnail.jpg")
                thumbnail_path = self.cache_dir / thumbnail_name
                if not thumbnail_path.exists() or not use_cache:
                    try:
                        thumbnail.convert("RGB").save(thumbnail_path, format="JPEG", quality=90)
                    except Exception:
                        pass

                embedding_name = cache_entry.get("embedding") if use_cache else None
                embedding_path = self.cache_dir / (embedding_name or self._cache_name(slug, "_embedding.npy"))
                if not embedding_path.exists() or not use_cache:
                    try:
                        np.save(embedding_path, embedding.astype(np.float32))
                    except Exception:
                        pass
                    embedding_name = embedding_path.name
                else:
                    embedding_name = embedding_path.name

                updated_manifest_images[slug] = {
                    "path": str(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "name": meta_entry.get("name", slug),
                    "gender": gender,
                    "gender_confidence": gender_confidence,
                    "thumbnail": thumbnail_path.name,
                    "face_crop": face_crop_name,
                    "embedding": embedding_name,
                }

                if total:
                    progress = len(entries) / total
                    self._notify_status(f"Loading celebrity images {len(entries)}/{total}", progress)

            self.entries = entries
            if embeddings:
                self.embedding_matrix = np.vstack(embeddings)
                dim = embeddings[0].shape[0]
                index = AnnoyIndex(dim, "angular")
                for idx, emb in enumerate(embeddings):
                    index.add_item(idx, emb.tolist())
                index.build(10)
                self.annoy_index = index
                try:
                    np.save(self._cache_embedding_path, self.embedding_matrix.astype(np.float32))
                except Exception:
                    pass
                try:
                    index.save(str(self._cache_annoy_path))
                except Exception:
                    pass
            else:
                self.embedding_matrix = None
                self.annoy_index = None

            for idx, entry in enumerate(self.entries):
                self.gender_indices.setdefault(entry.gender, []).append(idx)
            if total:
                self._notify_status(None, 1.0)

            self.gender_centroids = {}
            if self.embedding_matrix is not None:
                for gender in ("male", "female"):
                    idxs = [i for i, entry in enumerate(self.entries) if entry.gender == gender]
                    if not idxs:
                        continue
                    centroid = self.embedding_matrix[idxs].mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    self.gender_centroids[gender] = centroid.astype(np.float32)

            self._dataset_signature = signature_files
            self._metadata_mtime_ns = metadata_mtime
            manifest_update = {
                "signature": list(signature_files),
                "metadata_mtime": metadata_mtime,
                "cache_version": self.CACHE_VERSION,
                "images": updated_manifest_images,
                "order": order,
                "embedding_matrix": self._cache_embedding_path.name,
                "annoy_index": self._cache_annoy_path.name,
            }
            self._save_cache_manifest(manifest_update)
        finally:
            self._load_event.set()

    def prefetch_entries(
        self,
        face_detector: Optional[FaceDetector],
        face_embedder: FaceEmbedder,
        gender_classifier: Optional[GenderClassifier] = None,
        *,
        force: bool = False,
    ) -> None:
        with self._load_thread_lock:
            if self._load_thread is not None and self._load_thread.is_alive():
                return
            if (
                not force
                and self.entries
                and self._dataset_signature is not None
                and self._metadata_mtime_ns is not None
                and self._load_event.is_set()
            ):
                return

            self._load_event.clear()
            def worker() -> None:
                try:
                    self.load_entries(face_detector, face_embedder, gender_classifier)
                except Exception:
                    pass
                finally:
                    with self._load_thread_lock:
                        self._load_thread = None

            self._load_thread = threading.Thread(target=worker, daemon=True)
            self._load_thread.start()

    def require_entries(
        self,
        face_detector: Optional[FaceDetector],
        face_embedder: FaceEmbedder,
        gender_classifier: Optional[GenderClassifier] = None,
        *,
        wait: bool = True,
    ) -> None:
        current_signature, current_metadata = self._compute_signature()
        dataset_changed = (
            self._dataset_signature is None
            or self._dataset_signature != current_signature
            or self._metadata_mtime_ns != current_metadata
        )

        if dataset_changed:
            self._notify_status("Dataset change detected; refreshing faces…", 0.0)
            self.prefetch_entries(face_detector, face_embedder, gender_classifier, force=True)
        elif not self.entries:
            self.prefetch_entries(face_detector, face_embedder, gender_classifier, force=True)

        if wait:
            self._load_event.wait()

        if not self.entries:
            raise FileNotFoundError(
                "No celebrity images were found. Add images to the dataset or download the curated set."
            )

    def best_match(
        self,
        source_image: Image.Image,
        face_detector: Optional[FaceDetector],
        face_embedder: FaceEmbedder,
        allowed_genders: Optional[List[str]] = None,
        gender_classifier: Optional[GenderClassifier] = None,
    ) -> Tuple[CelebrityEntry, float]:
        self.require_entries(face_detector, face_embedder, gender_classifier)

        face = extract_face_region(source_image, face_detector)
        embedding = face_embedder.embed(np.array(face))
        return self.best_match_from_embedding(embedding, allowed_genders)

    def best_match_from_embedding(
        self,
        embedding: np.ndarray,
        allowed_genders: Optional[List[str]] = None,
    ) -> Tuple[CelebrityEntry, float]:
        if not self.entries or self.annoy_index is None or self.embedding_matrix is None:
            raise FileNotFoundError(
                "No celebrity images were found. Add images to the dataset or download the curated set."
            )

        top_n = min(50, len(self.entries))
        candidate_ids = self.annoy_index.get_nns_by_vector(embedding.tolist(), top_n, include_distances=False)
        best_idx: Optional[int] = None

        if allowed_genders:
            for idx in candidate_ids:
                if self.entries[idx].gender in allowed_genders:
                    best_idx = idx
                    break

        if best_idx is None:
            best_idx = candidate_ids[0] if candidate_ids else 0

        similarity = float(np.dot(embedding, self.embedding_matrix[best_idx]))
        return self.entries[best_idx], similarity

    def infer_gender(self, embedding: np.ndarray) -> str:
        if not self.gender_centroids:
            return "unknown"

        best_gender = "unknown"
        best_score = -1.0
        for gender, centroid in self.gender_centroids.items():
            score = float(np.dot(embedding, centroid))
            if score > best_score:
                best_score = score
                best_gender = gender

        if best_score < 0.25:
            return "unknown"
        return best_gender


class CelebrityMatcherApp:
    """
    Simple scaffold for matching real faces to known celebrities.
    Real detection/matching logic should plug into `match_celebrity`.
    """

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Face-to-Celebrity Matcher")
        self.selected_image: Optional[Path] = None
        self.dataset = CelebrityDataset(DATASET_DIR, SAMPLE_CELEBRITY_URLS)
        self.model_dir = Path(__file__).with_name("models")
        self.available_cameras: List[int] = [0, 1, 2]
        self.camera_index_var = StringVar(value="0")
        self.camera_index_combo: Optional[ttk.Combobox] = None
        self.camera_capture: Optional[Any] = None
        self.camera_running = False
        self.camera_photo: Optional[ImageTk.PhotoImage] = None
        self.celebrity_photo: Optional[ImageTk.PhotoImage] = None
        self.latest_frame: Optional[Any] = None
        self.cv2: Optional[Any] = None
        self.match_interval_ms: int = 1000
        self.match_job: Optional[str] = None
        self.dataset_warning_shown: bool = False
        self.current_match_name: Optional[str] = None
        self.match_request_queue: queue.Queue = queue.Queue(maxsize=2)
        self.match_result_queue: queue.Queue = queue.Queue()
        self.match_worker_thread: Optional[threading.Thread] = None
        self.match_worker_stop = threading.Event()
        self.last_sent_frame_digest: Optional[int] = None
        self.latest_face_array: Optional[np.ndarray] = None
        self.current_face_gender: str = "unknown"
        self.current_gender_confidence: float = 0.0
        self.is_frozen = False
        self.frozen_frame: Optional[np.ndarray] = None
        self.freeze_button: Optional[ttk.Button] = None
        self.dataset_status_label: Optional[ttk.Label] = None
        self.dataset_progress: Optional[ttk.Progressbar] = None
        self.freeze_status_label: Optional[ttk.Label] = None
        self.similarity_label: Optional[ttk.Label] = None
        self.gender_status: Optional[ttk.Label] = None
        self.gender_classifier: Optional[GenderClassifier] = None
        self.gender_classifier_warning: Optional[str] = None
        self.gender_smoother = GenderSmoother()
        self.last_sent_frame_time: float = 0.0

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.face_detector = FaceDetector(self.model_dir)
        self.face_embedder = FaceEmbedder(self.model_dir)
        self.face_embedder.warmup(runs=2)
        try:
            custom_gender_model = self.model_dir / "model_int8.onnx"
            model_path = custom_gender_model if custom_gender_model.exists() else None
            self.gender_classifier = GenderClassifier(self.model_dir, model_path=model_path)
            self.gender_classifier.warmup(runs=2)
        except Exception as exc:
            self.gender_classifier = None
            self.gender_classifier_warning = str(exc)

        self.dataset.ensure_directory()
        self.dataset.set_gender_classifier(self.gender_classifier)
        self.dataset.set_status_callback(self._handle_dataset_status)
        self._build_menu()
        self._build_main_panel()
        if self.gender_classifier is None and self.gender_classifier_warning and self.gender_status is not None:
            self.gender_status.config(
                text="Detected gender: classifier unavailable (see README for model download instructions)."
            )
        self.dataset.prefetch_entries(self.face_detector, self.face_embedder, self.gender_classifier, force=False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _handle_dataset_status(self, message: Optional[str], progress: Optional[float]) -> None:
        def update() -> None:
            if self.dataset_progress is not None and progress is not None:
                value = max(0.0, min(progress, 1.0)) * 100.0
                self.dataset_progress['value'] = value
            if self.dataset_status_label is not None:
                if message is None:
                    self.dataset_status_label.config(text="Dataset status: Ready")
                else:
                    self.dataset_status_label.config(text=f"Dataset status: {message}")
        self.root.after(0, update)

    def _update_freeze_status(self, mode: str, color: str) -> None:
        if self.freeze_status_label is not None:
            self.freeze_status_label.config(text=f"Mode: {mode}", foreground=color)

    def _bucket_similarity(self, score: float) -> str:
        if score >= 0.85:
            return "high"
        if score >= 0.65:
            return "medium"
        if score >= 0.5:
            return "low"
        return "very low"

    def _format_similarity(self, cosine_similarity: Optional[float]) -> str:
        if cosine_similarity is None:
            return "Similarity: --"
        clamped = max(-1.0, min(float(cosine_similarity), 1.0))
        normalized = (clamped + 1.0) / 2.0
        normalized = max(0.0, min(normalized, 1.0))
        bucket = self._bucket_similarity(normalized)
        return f"Similarity: {normalized * 100:.1f}% ({bucket})"

    def _update_gender_status(self, gender: str, confidence: float) -> None:
        resolved_gender = gender if gender in ("male", "female") else "unknown"
        resolved_confidence = confidence if resolved_gender != "unknown" else 0.0
        self.current_face_gender = resolved_gender
        self.current_gender_confidence = resolved_confidence
        if self.gender_status is None:
            return
        if resolved_gender == "unknown":
            self.gender_status.config(text="Detected gender: unknown (0%)")
            return
        self.gender_status.config(text=f"Detected gender: {resolved_gender} ({resolved_confidence * 100:.0f}%)")

    def _on_camera_changed(self, event: Optional[Any] = None) -> None:
        selection = self.camera_index_var.get()
        if self.camera_running:
            self.image_status.config(text=f"Using camera {selection}. Stop and restart to switch.")
        else:
            self.image_status.config(text=f"Selected camera index {selection}. Ready to start camera.")

    def _build_menu(self) -> None:
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image…", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_main_panel(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid(column=0, row=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        description = (
            "Load a photo or use the camera to identify the closest celebrity match.\n"
            "Download the curated dataset or add your own images inside 'celebrity_dataset/'."
        )
        ttk.Label(container, text=description, wraplength=520, justify="left").grid(
            column=0, row=0, columnspan=2, pady=(0, 12), sticky="w"
        )

        display_frame = ttk.Frame(container)
        display_frame.grid(column=0, row=1, columnspan=2, sticky="nsew")
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)

        self.camera_label = ttk.Label(
            display_frame,
            text="Camera preview not started.",
            anchor="center",
            width=40,
        )
        self.camera_label.grid(column=0, row=0, padx=(0, 8), sticky="nsew")

        self.celebrity_label = ttk.Label(
            display_frame,
            text="Celebrity match will appear here.",
            anchor="center",
            width=40,
        )
        self.celebrity_label.grid(column=1, row=0, padx=(8, 0), sticky="nsew")

        self.image_status = ttk.Label(container, text="No image selected.")
        self.image_status.grid(column=0, row=2, columnspan=2, pady=(12, 0), sticky="w")

        controls = ttk.Frame(container)
        controls.grid(column=0, row=3, columnspan=2, pady=(12, 0), sticky="ew")
        for col in range(4):
            controls.columnconfigure(col, weight=1)

        ttk.Button(controls, text="Load Photo", command=self.load_image).grid(
            column=0, row=0, padx=(0, 8), sticky="ew"
        )
        ttk.Button(controls, text="Use Camera", command=self.capture_from_camera).grid(
            column=1, row=0, padx=4, sticky="ew"
        )
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera).grid(
            column=2, row=0, padx=(8, 0), sticky="ew"
        )
        self.freeze_button = ttk.Button(controls, text="Freeze Frame", command=self.toggle_freeze)
        self.freeze_button.grid(column=3, row=0, padx=(8, 0), sticky="ew")
        ttk.Label(controls, text="Camera Index").grid(column=0, row=1, pady=(8, 0), sticky="w")
        self.camera_index_combo = ttk.Combobox(
            controls,
            textvariable=self.camera_index_var,
            values=[str(idx) for idx in self.available_cameras] or ["0"],
            state="readonly",
            width=6,
        )
        self.camera_index_combo.grid(column=1, row=1, pady=(8, 0), sticky="w")
        self.camera_index_combo.bind("<<ComboboxSelected>>", self._on_camera_changed)

        self.match_result = ttk.Label(container, text="Start the camera to see live matches.")
        self.match_result.grid(column=0, row=4, columnspan=2, pady=(16, 0), sticky="w")

        self.similarity_label = ttk.Label(container, text="Similarity: --")
        self.similarity_label.grid(column=0, row=5, columnspan=2, pady=(4, 0), sticky="w")

        self.gender_status = ttk.Label(container, text="Detected gender: unknown")
        self.gender_status.grid(column=0, row=6, columnspan=2, pady=(8, 0), sticky="w")

        self.dataset_status_label = ttk.Label(container, text="Dataset status: Ready")
        self.dataset_status_label.grid(column=0, row=7, columnspan=2, sticky="w")
        self.dataset_progress = ttk.Progressbar(
            container,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            value=0,
        )
        self.dataset_progress.grid(column=0, row=8, columnspan=2, sticky="ew", pady=(4, 0))

        self.freeze_status_label = ttk.Label(container, text="Mode: Idle", foreground="grey")
        self.freeze_status_label.grid(column=0, row=9, columnspan=2, sticky="w", pady=(8, 0))
        self._update_freeze_status("Idle", "grey")

    def load_image(self) -> None:
        if self.camera_running:
            self.stop_camera()

        filename = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ),
        )
        if not filename:
            return

        self.selected_image = Path(filename)
        self.image_status.config(text=f"Selected: {self.selected_image.name}")
        self.current_match_name = None
        try:
            with Image.open(self.selected_image) as img:
                source_image = img.convert("RGB")
        except (OSError, FileNotFoundError):
            self.match_result.config(text="Unable to open the selected image.")
            return

        try:
            match_entry, similarity = self.match_celebrity(source_image)
        except FileNotFoundError as exc:
            self.match_result.config(text=str(exc))
            return
        except Exception:
            self.match_result.config(text="Failed to compute a match for this image.")
            return

        self._apply_match_entry(match_entry, similarity)
        self.gender_smoother.reset()
        predicted_gender = match_entry.gender
        predicted_confidence = match_entry.gender_confidence
        if self.gender_classifier is not None:
            try:
                face_crop = extract_face_region(source_image, self.face_detector)
                classifier_label, classifier_confidence, _ = self.gender_classifier.classify(face_crop)
                if classifier_label in ("male", "female") and classifier_confidence >= predicted_confidence:
                    predicted_gender = classifier_label
                    predicted_confidence = classifier_confidence
            except Exception:
                pass
        self._update_gender_status(predicted_gender, predicted_confidence)

    def get_cv2_module(self, require: bool = False) -> Optional[Any]:
        if self.cv2 is None:
            try:
                import cv2  # type: ignore

                self.cv2 = cv2
            except ImportError:
                if require:
                    messagebox.showerror(
                        "OpenCV Missing",
                        "OpenCV (cv2) is required for camera and face matching features.\n"
                        "Install it with `pip install opencv-python` in the active environment.",
                    )
                return None
        return self.cv2

    def capture_from_camera(self) -> None:
        if self.camera_running:
            return

        cv2_module = self.get_cv2_module(require=True)
        if cv2_module is None:
            return

        values = [str(idx) for idx in self.available_cameras] or ["0"]
        if self.camera_index_combo is not None:
            self.camera_index_combo["values"] = values
        if self.camera_index_var.get() not in values:
            self.camera_index_var.set(values[0])

        capture = self._open_camera(cv2_module)

        if capture is None:
            return

        self.cv2 = cv2_module
        self.camera_capture = capture
        self.camera_running = True
        self.selected_image = None
        self.current_match_name = None
        self.latest_face_array = None
        self.last_sent_frame_digest = None
        self.gender_smoother.reset()
        self.current_gender_confidence = 0.0
        self.image_status.config(text=f"Camera live preview active (index {int(self.camera_index_var.get())}).")
        try:
            self.dataset.require_entries(self.face_detector, self.face_embedder, self.gender_classifier)
        except FileNotFoundError:
            self.match_result.config(text="No celebrity images available. Add some to 'celebrity_dataset/'.")
        else:
            self.match_result.config(text="Looking for the closest celebrity match...")
        self._update_freeze_status("Live", "green")
        self._start_live_matching()
        self._update_camera_frame()

    def download_celebrity_set(self) -> None:
        saved, failures = self.dataset.download_samples()

        messages: List[str] = []
        if saved:
            messages.append(f"Downloaded {saved} new image(s).")
        if failures:
            readable = ", ".join(format_display_name(name) for name in failures)
            messages.append(f"Failed to download: {readable}.")
        if not messages:
            messages.append("All curated celebrity images are already available.")

        messagebox.showinfo("Celebrity Dataset", "\n".join(messages))

        try:
            self.dataset.load_entries(self.face_detector, self.face_embedder, self.gender_classifier)
            self.current_match_name = None
            self.latest_face_array = None
            self.last_sent_frame_digest = None
            self.match_result.config(text="Celebrity dataset refreshed.")
        except Exception:
            # Loading is best-effort; errors will surface during matching if needed.
            pass

    def stop_camera(self) -> None:
        self.camera_running = False
        self._stop_live_matching()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None

        self.camera_photo = None
        self.latest_frame = None
        self.current_match_name = None
        self.latest_face_array = None
        self.last_sent_frame_digest = None
        self.gender_smoother.reset()
        self.last_sent_frame_time = 0.0
        self.current_face_gender = "unknown"
        self.current_gender_confidence = 0.0
        self.is_frozen = False
        self.frozen_frame = None
        if self.freeze_button is not None:
            self.freeze_button.config(text="Freeze Frame")
        self.camera_label.config(text="Camera preview stopped.", image="")
        self.image_status.config(text="Camera stopped. Select a photo or restart the camera.")
        self._update_gender_status("unknown", 0.0)
        if self.similarity_label is not None:
            self.similarity_label.config(text="Similarity: --")
        self._update_freeze_status("Stopped", "grey")

    def _detect_cameras(self, max_devices: int = 6) -> List[int]:
        cv2_module = self.get_cv2_module()
        indices: List[int] = []
        if cv2_module is None:
            return [0]
        for idx in range(max_devices):
            cap = cv2_module.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                indices.append(idx)
            if cap is not None:
                cap.release()
        return indices or [0]

    def _open_camera(self, cv2_module: Any) -> Optional[Any]:
        index = int(self.camera_index_var.get())
        attempts = []
        attempts.append(("default backend", lambda: cv2_module.VideoCapture(index)))

        if hasattr(cv2_module, "CAP_AVFOUNDATION"):
            attempts.append(
                (
                    "AVFoundation backend",
                    lambda: cv2_module.VideoCapture(index, cv2_module.CAP_AVFOUNDATION),
                )
            )

        for attr in ("CAP_MSMF", "CAP_DSHOW", "CAP_V4L2"):
            if hasattr(cv2_module, attr):
                backend = getattr(cv2_module, attr)
                attempts.append(
                    (
                        f"{attr} backend",
                        lambda src=index, backend=backend: cv2_module.VideoCapture(src, backend),
                    )
                )

        errors = []
        for label, factory in attempts:
            candidate = factory()
            if candidate.isOpened():
                return candidate
            errors.append(label)
            candidate.release()

        attempted = ", ".join(errors) or "no backends"
        messagebox.showerror(
            "Camera Capture",
            f"Unable to access camera index {index} with OpenCV.\n"
            f"Tried backends: {attempted}.\n\n"
            "Confirm the camera is connected and that the app has permission "
            "to use it (macOS: System Settings → Privacy & Security → Camera).",
        )
        return None

    def _update_camera_frame(self) -> None:
        if not self.camera_running or self.camera_capture is None or self.cv2 is None:
            return

        success, frame = self.camera_capture.read()
        if not success or frame is None:
            self.camera_label.config(text="Unable to read from camera.", image="")
            self.root.after(500, self._update_camera_frame)
            return

        frame_rgb_detection = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        face_region_rgb: Optional[np.ndarray] = None
        face_bbox: Optional[Tuple[int, int, int, int]] = None
        detections = self.face_detector.detect(frame_rgb_detection) if self.face_detector else []
        if detections:
            x, y, w, h = detections[0]
            self.cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_bbox = (x, y, w, h)

        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_region_rgb = frame_rgb_detection[y : y + h, x : x + w]

        if face_region_rgb is not None and face_region_rgb.size > 0:
            self.latest_face_array = face_region_rgb.copy()
        else:
            self._handle_no_face_detected()

        frame_rgb_display = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        self.latest_frame = frame_rgb_display

        preview_image = Image.fromarray(frame_rgb_display)
        preview_image = preview_image.resize((360, 270))

        self.camera_photo = ImageTk.PhotoImage(preview_image)
        self.camera_label.config(image=self.camera_photo, text="")
        if not self.is_frozen:
            self.root.after(30, self._update_camera_frame)

    def _handle_no_face_detected(self) -> None:
        self.latest_face_array = None
        self.last_sent_frame_digest = None
        self.last_sent_frame_time = 0.0
        if self.current_face_gender != "unknown" or self.current_gender_confidence != 0.0:
            self.gender_smoother.reset()
            self._update_gender_status("unknown", 0.0)

    def _start_live_matching(self) -> None:
        if self.match_worker_thread is None or not self.match_worker_thread.is_alive():
            self.match_worker_stop.clear()
            self.match_worker_thread = threading.Thread(target=self._match_worker_loop, daemon=True)
            self.match_worker_thread.start()
        if self.match_job is None:
            self.match_job = self.root.after(self.match_interval_ms, self._process_live_match)

    def _stop_live_matching(self) -> None:
        if self.match_job is not None:
            self.root.after_cancel(self.match_job)
            self.match_job = None
        if self.match_worker_thread is not None:
            self.match_worker_stop.set()
            try:
                self.match_request_queue.put_nowait((-1, None, None))
            except queue.Full:
                pass
            self.match_worker_thread.join(timeout=2.0)
            self.match_worker_thread = None
        self.match_request_queue = queue.Queue(maxsize=2)
        self.match_result_queue = queue.Queue()
        self.last_sent_frame_digest = None
        self.match_worker_stop = threading.Event()
        self.last_sent_frame_time = 0.0

    def _process_live_match(self) -> None:
        self.match_job = None

        if not self.camera_running:
            return

        self._drain_match_results()

        if self.latest_face_array is not None:
            face_digest = hash(self.latest_face_array.tobytes())
            now = time.monotonic()
            stale = now - self.last_sent_frame_time >= max(self.match_interval_ms / 1000.0, 0.05)
            if face_digest != self.last_sent_frame_digest or stale:
                try:
                    self.match_request_queue.put_nowait(
                        (face_digest, self.latest_face_array.copy(), self.current_face_gender)
                    )
                    self.last_sent_frame_digest = face_digest
                    self.last_sent_frame_time = now
                except queue.Full:
                    pass

        if self.camera_running:
            self.match_job = self.root.after(self.match_interval_ms, self._process_live_match)

    def _drain_match_results(self) -> None:
        while True:
            try:
                digest, entry, detected_gender, similarity, classifier_confidence, classifier_probabilities = (
                    self.match_result_queue.get_nowait()
                )
            except queue.Empty:
                break

            if self.last_sent_frame_digest is not None and digest != self.last_sent_frame_digest:
                continue
            if self.last_sent_frame_digest is None and self.latest_face_array is None and digest is not None:
                continue

            smoothed_gender, smoothed_confidence = self.gender_smoother.update(classifier_probabilities)

            resolved_gender = "unknown"
            resolved_confidence = 0.0
            if smoothed_gender in ("male", "female"):
                resolved_gender = smoothed_gender
                resolved_confidence = smoothed_confidence
            elif detected_gender in ("male", "female"):
                resolved_gender = detected_gender
                resolved_confidence = classifier_confidence

            if entry is None:
                if self.similarity_label is not None:
                    self.similarity_label.config(text="Similarity: --")
                if not self.dataset_warning_shown:
                    self.match_result.config(text="Add celebrity images to enable live matching.")
                    self.dataset_warning_shown = True
                self._update_gender_status(resolved_gender, resolved_confidence)
                continue

            self.dataset_warning_shown = False
            self._apply_match_entry(entry, similarity)
            self._update_gender_status(resolved_gender, resolved_confidence)

    def _match_worker_loop(self) -> None:
        while not self.match_worker_stop.is_set():
            try:
                digest, face_array, detected_gender = self.match_request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if digest is None or face_array is None or digest == -1:
                continue

            classifier_label = "unknown"
            classifier_confidence = 0.0
            classifier_probabilities: Optional[np.ndarray] = None
            predicted_gender = "unknown"
            try:
                self.dataset.require_entries(self.face_detector, self.face_embedder, self.gender_classifier)
            except Exception:
                pass

            try:
                face_uint8 = np.clip(face_array, 0, 255).astype("uint8")
                face_image = Image.fromarray(face_uint8)
            except Exception:
                face_image = None

            try:
                if self.gender_classifier is not None and face_image is not None:
                    try:
                        classifier_label, classifier_confidence, classifier_probabilities = self.gender_classifier.classify(
                            face_image
                        )
                    except Exception:
                        classifier_label = "unknown"
                        classifier_confidence = 0.0
                        classifier_probabilities = None

                embedding = self.face_embedder.embed(face_array)
                centroid_gender = self.dataset.infer_gender(embedding)
                predicted_gender = classifier_label if classifier_label in ("male", "female") else centroid_gender
                allowed: Optional[List[str]] = None
                if classifier_label in ("male", "female") and classifier_confidence >= 0.70:
                    allowed = [classifier_label]
                elif predicted_gender in ("male", "female") and classifier_label not in ("male", "female"):
                    allowed = [predicted_gender]
                match_entry, similarity = self.dataset.best_match_from_embedding(embedding, allowed)
            except FileNotFoundError:
                try:
                    self.match_result_queue.put_nowait((digest, None, detected_gender, None, 0.0, None))
                except queue.Full:
                    pass
                continue
            except Exception:
                continue

            try:
                gender_for_queue = classifier_label if classifier_label in ("male", "female") else (
                    predicted_gender if predicted_gender in ("male", "female") else detected_gender
                )
                self.match_result_queue.put_nowait(
                    (
                        digest,
                        match_entry,
                        gender_for_queue,
                        similarity,
                        classifier_confidence if gender_for_queue in ("male", "female") else 0.0,
                        classifier_probabilities,
                    )
                )
            except queue.Full:
                pass

    def _apply_match_entry(
        self,
        entry: CelebrityEntry,
        similarity: Optional[float],
        *,
        friendly_name: Optional[str] = None,
    ) -> None:
        friendly = friendly_name or format_display_name(entry.name)

        self.current_match_name = entry.name
        self.dataset_warning_shown = False

        self.match_result.config(text=f"Closest celebrity match: {friendly}")
        self.display_celebrity_entry(entry, friendly)
        if self.similarity_label is not None:
            self.similarity_label.config(text=self._format_similarity(similarity))

    def run_match(self) -> None:
        messagebox.showinfo(
            "Live Matching",
            "Live matching runs automatically when the camera is active.",
        )

    def display_celebrity_entry(self, entry: CelebrityEntry, friendly_name: Optional[str] = None) -> None:
        display_image = entry.thumbnail.copy() if entry.thumbnail else entry.image.copy()
        resolved_name = friendly_name or format_display_name(entry.name)
        self.celebrity_photo = ImageTk.PhotoImage(display_image)
        self.celebrity_label.config(image=self.celebrity_photo, text=resolved_name)

    def match_celebrity(
        self,
        image: Image.Image,
        allowed_genders: Optional[List[str]] = None,
    ) -> Tuple[CelebrityEntry, float]:
        return self.dataset.best_match(
            image,
            self.face_detector,
            self.face_embedder,
            allowed_genders,
            self.gender_classifier,
        )

    def toggle_freeze(self) -> None:
        if not self.camera_running:
            messagebox.showwarning("Freeze Frame", "Start the camera before freezing a frame.")
            return
        if self.latest_frame is None:
            messagebox.showwarning("Freeze Frame", "No camera frame available to freeze.")
            return

        if not self.is_frozen:
            self.is_frozen = True
            self.frozen_frame = self.latest_frame.copy()
            if self.freeze_button is not None:
                self.freeze_button.config(text="Resume Camera")
            self._stop_live_matching()
            self._update_freeze_status("Frozen", "orange")
            self.match_result.config(text="Analyzing frozen frame...")
            self._run_frozen_match()
        else:
            self.is_frozen = False
            self.frozen_frame = None
            if self.freeze_button is not None:
                self.freeze_button.config(text="Freeze Frame")
            self.match_result.config(text="Resumed live matching.")
            self._update_freeze_status("Live", "green")
            self._start_live_matching()
            self._update_camera_frame()

    def _run_frozen_match(self) -> None:
        if self.frozen_frame is None:
            return

        try:
            frozen_image = Image.fromarray(self.frozen_frame.astype("uint8"))
        except Exception:
            self.match_result.config(text="Unable to process frozen frame.")
            return

        allowed = [self.current_face_gender] if self.current_face_gender in ("male", "female") else None

        try:
            match_entry, similarity = self.match_celebrity(frozen_image, allowed)
        except FileNotFoundError as exc:
            self.match_result.config(text=str(exc))
            return
        except Exception:
            self.match_result.config(text="Matching failed for frozen frame.")
            return

        self._apply_match_entry(match_entry, similarity)
        self.match_result.config(text="Frozen frame analysis complete.")

    def on_close(self) -> None:
        self.stop_camera()
        if hasattr(self, "face_detector") and self.face_detector:
            try:
                self.face_detector.close()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = CelebrityMatcherApp(root)
    root.mainloop()
