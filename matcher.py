from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

from face_detection import FaceDetector, extract_face_region
from gender import GenderClassifier


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
    primary_image: Image.Image
    primary_thumbnail: Image.Image
    primary_path: Path
    gallery_paths: List[Path] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    gender: str = "unknown"
    gender_confidence: float = 0.0


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
        array = array[:, :, ::-1]
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


class CelebrityDataset:
    def __init__(self, dataset_dir: Path, url_map: dict[str, str]) -> None:
        self.CACHE_VERSION = 3
        self.dataset_dir = dataset_dir
        self.url_map = url_map
        self.entries: List[CelebrityEntry] = []
        self.embedding_matrix: Optional[np.ndarray] = None
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
        if not embedding_file:
            return False
        embedding_path = self.cache_dir / embedding_file
        if not embedding_path.exists():
            return False
        order: List[str] = manifest.get("order") or []
        profiles_meta: dict = manifest.get("profiles") or {}
        try:
            embedding_matrix = np.load(embedding_path)
        except Exception:
            return False
        if not order or embedding_matrix.shape[0] != len(order):
            return False
        new_entries: List[CelebrityEntry] = []
        aggregate_rows: List[np.ndarray] = []
        gender_indices: dict[str, List[int]] = {"male": [], "female": [], "unknown": []}
        for idx, slug in enumerate(order):
            profile_meta = profiles_meta.get(slug)
            if not profile_meta:
                return False
            gallery_meta = profile_meta.get("gallery") or []
            if not gallery_meta:
                return False
            embeddings: List[np.ndarray] = []
            gallery_paths: List[Path] = []
            for item in gallery_meta:
                image_path = Path(item.get("path", ""))
                embedding_name = item.get("embedding")
                if not image_path.exists() or not embedding_name:
                    return False
                embedding_path_item = self.cache_dir / embedding_name
                if not embedding_path_item.exists():
                    return False
                try:
                    embedding = np.load(embedding_path_item).astype(np.float32)
                except Exception:
                    return False
                embeddings.append(embedding)
                gallery_paths.append(image_path)
            if not embeddings:
                return False
            primary_path = Path(profile_meta.get("primary_path", gallery_paths[0]))
            if not primary_path.exists():
                primary_path = gallery_paths[0]
            try:
                with Image.open(primary_path) as primary_img:
                    primary_image = primary_img.convert("RGB")
            except Exception:
                return False
            primary_thumb_name = profile_meta.get("primary_thumbnail")
            primary_thumbnail: Image.Image
            if primary_thumb_name:
                thumb_path = self.cache_dir / primary_thumb_name
                if thumb_path.exists():
                    try:
                        with Image.open(thumb_path) as thumb_img:
                            primary_thumbnail = thumb_img.convert("RGB")
                    except Exception:
                        primary_thumbnail = primary_image.copy()
                        primary_thumbnail.thumbnail((360, 270), Image.LANCZOS)
                else:
                    primary_thumbnail = primary_image.copy()
                    primary_thumbnail.thumbnail((360, 270), Image.LANCZOS)
            else:
                primary_thumbnail = primary_image.copy()
                primary_thumbnail.thumbnail((360, 270), Image.LANCZOS)
            aggregate = embedding_matrix[idx].astype(np.float32)
            norm = np.linalg.norm(aggregate)
            if norm > 0:
                aggregate = aggregate / norm
            entry = CelebrityEntry(
                name=profile_meta.get("name", format_display_name(slug)),
                primary_image=primary_image.copy(),
                primary_thumbnail=primary_thumbnail.copy(),
                primary_path=primary_path,
                gallery_paths=gallery_paths,
                embeddings=[emb.astype(np.float32) for emb in embeddings],
                embedding=aggregate,
                gender=profile_meta.get("gender", "unknown"),
                gender_confidence=float(profile_meta.get("gender_confidence", 0.0)),
            )
            new_entries.append(entry)
            aggregate_rows.append(aggregate)
            gender_indices.setdefault(entry.gender, []).append(idx)
        if not new_entries:
            return False
        self.entries = new_entries
        self.embedding_matrix = np.vstack(aggregate_rows).astype(np.float32)
        self.gender_indices = gender_indices
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

    def _discover_sources(self, metadata: dict) -> List[dict]:
        sources: List[dict] = []
        image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for entry in sorted(self.dataset_dir.iterdir(), key=lambda p: p.name.lower()):
            if entry.name.startswith(".") or entry == self.cache_dir:
                continue
            if entry.is_dir():
                profile_data: dict = {}
                profile_path = entry / "profile.json"
                if profile_path.exists():
                    try:
                        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        profile_data = {}
                images = [p for p in sorted(entry.iterdir()) if p.is_file() and p.suffix.lower() in image_exts]
                if not images:
                    continue
                slug = profile_data.get("slug") or entry.name
                meta_entry = metadata.get(slug, {})
                display_name = (
                    profile_data.get("name")
                    or meta_entry.get("name")
                    or format_display_name(slug)
                )
                gender = (
                    profile_data.get("gender")
                    or meta_entry.get("gender")
                    or SAMPLE_CELEBRITY_GENDERS.get(slug, "unknown")
                )
                primary_hint = profile_data.get("primary_image")
                primary_path = entry / primary_hint if primary_hint else images[0]
                if not primary_path.exists():
                    primary_path = images[0]
                sources.append(
                    {
                        "slug": slug,
                        "display_name": display_name,
                        "gender": (gender or "unknown").lower(),
                        "image_paths": images,
                        "primary_path": primary_path,
                    }
                )
            elif entry.is_file() and entry.suffix.lower() in image_exts:
                slug = entry.stem
                meta_entry = metadata.get(slug, {})
                display_name = meta_entry.get("name") or format_display_name(slug)
                gender = meta_entry.get("gender") or SAMPLE_CELEBRITY_GENDERS.get(slug, "unknown")
                sources.append(
                    {
                        "slug": slug,
                        "display_name": display_name,
                        "gender": (gender or "unknown").lower(),
                        "image_paths": [entry],
                        "primary_path": entry,
                    }
                )
        return sources

    def download_samples(self) -> Tuple[int, List[str]]:
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
            for path in self.dataset_dir.rglob(pattern):
                if self.cache_dir in path.parents:
                    continue
                files.append(path)
        files.sort()
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
                    primary_image=rgb_image.copy(),
                    primary_thumbnail=thumbnail.copy(),
                    primary_path=path,
                    gallery_paths=[path],
                    embeddings=[embedding.astype(np.float32)],
                    embedding=embedding.astype(np.float32),
                    gender=gender,
                    gender_confidence=gender_confidence,
                )
                entries.append(entry)
                embeddings.append(embedding.astype(np.float32))
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
                try:
                    np.save(self._cache_embedding_path, self.embedding_matrix.astype(np.float32))
                except Exception:
                    pass
            else:
                self.embedding_matrix = None
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
    ) -> Tuple[CelebrityEntry, float, float]:
        self.require_entries(face_detector, face_embedder, gender_classifier)
        face = extract_face_region(source_image, face_detector)
        embedding = face_embedder.embed(np.array(face))
        return self.best_match_from_embedding(embedding, allowed_genders)

    def best_match_from_embedding(
        self,
        embedding: np.ndarray,
        allowed_genders: Optional[List[str]] = None,
    ) -> Tuple[CelebrityEntry, float, float]:
        if not self.entries or self.embedding_matrix is None:
            raise FileNotFoundError(
                "No celebrity images were found. Add images to the dataset or download the curated set."
            )
        scores = self.embedding_matrix @ embedding
        best_idx: Optional[int] = None
        if allowed_genders:
            mask = np.array([entry.gender in allowed_genders for entry in self.entries], dtype=bool)
            if mask.any():
                masked_scores = np.where(mask, scores, -np.inf)
                best_idx = int(np.argmax(masked_scores))
        if best_idx is None:
            best_idx = int(np.argmax(scores))
        descending = np.argsort(scores)[::-1]
        rank_position = int(np.where(descending == best_idx)[0][0])
        percentile = (rank_position + 1) / len(scores)
        similarity = float(scores[best_idx])
        return self.entries[best_idx], similarity, float(percentile)

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


__all__ = [
    "CelebrityEntry",
    "FaceEmbedder",
    "CelebrityDataset",
    "DATASET_DIR",
    "SAMPLE_CELEBRITY_URLS",
    "SAMPLE_CELEBRITY_GENDERS",
    "format_display_name",
]
