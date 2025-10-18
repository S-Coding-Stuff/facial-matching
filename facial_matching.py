from tkinter import Tk, Menu, ttk, filedialog, messagebox, StringVar
from pathlib import Path
from typing import Optional, Any, List, Tuple, Callable
import queue
import threading
from dataclasses import dataclass
import urllib.request
import json

import numpy as np
from PIL import Image, ImageTk

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


def extract_face_region(
    source: Image.Image,
    cv2_module: Optional[Any],
    detector: Optional[Any],
) -> Image.Image:
    """
    Attempt to isolate the most prominent face in the image. Falls back to a center crop.
    """
    if cv2_module is None or detector is None:
        return _center_square_crop(source)

    array = np.array(source.convert("RGB"))
    try:
        gray = cv2_module.cvtColor(array, cv2_module.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    except Exception:
        faces = ()

    if faces is None or len(faces) == 0:
        return _center_square_crop(source)

    # Select the largest detected face.
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    return source.crop((x, y, x + w, y + h))


def _center_square_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def compute_embedding(face_image: Image.Image) -> np.ndarray:
    """
    Generate a simple feature vector by combining color histograms and channel statistics.
    """
    face = face_image.resize((128, 128), Image.LANCZOS)
    array = np.asarray(face, dtype=np.float32) / 255.0

    hist_bins = 32
    hist_components: List[np.ndarray] = []
    for channel in range(3):
        channel_hist, _ = np.histogram(array[:, :, channel], bins=hist_bins, range=(0.0, 1.0), density=True)
        hist_components.append(channel_hist.astype(np.float32))

    means = array.mean(axis=(0, 1))
    stds = array.std(axis=(0, 1))
    feature_vector = np.concatenate(hist_components + [means, stds])

    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector /= norm

    return feature_vector.astype(np.float32)


class GenderClassifier:
    def __init__(self, model_name: str = "prithivMLmods/Realistic-Gender-Classification") -> None:
        self.available = False
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.confidence_threshold = 0.6
        self.labels_map: dict[int, str] = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoImageProcessor, AutoModelForImageClassification  # type: ignore
        except ImportError:
            print("[gender] Install torch and transformers to enable gender classification.")
            return

        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.labels_map = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
            self.available = True
            self.torch = torch
        except Exception as exc:
            print(f"[gender] Failed to load '{self.model_name}': {exc}")
            self.available = False

    def predict(self, face_rgb: Optional[np.ndarray]) -> str:
        if not self.available or face_rgb is None or face_rgb.size == 0:
            return "unknown"

        try:
            image = Image.fromarray(face_rgb.astype("uint8"))
            inputs = self.processor(images=image, return_tensors="pt")
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                probs = self.torch.nn.functional.softmax(outputs.logits, dim=-1)
                score, idx = probs.max(dim=-1)
                confidence = float(score.item())
                label = self.labels_map.get(int(idx.item()), "unknown")
                if confidence < self.confidence_threshold:
                    return "unknown"
                if "male" in label:
                    return "male"
                if "female" in label:
                    return "female"
                return label
        except Exception as exc:
            print(f"[gender] Inference error: {exc}")
            return "unknown"


class CelebrityDataset:
    def __init__(self, dataset_dir: Path, url_map: dict[str, str]) -> None:
        self.dataset_dir = dataset_dir
        self.url_map = url_map
        self.entries: List[CelebrityEntry] = []
        self.embedding_matrix: Optional[np.ndarray] = None
        self.gender_indices: dict[str, List[int]] = {"male": [], "female": [], "unknown": []}
        self.status_callback: Optional[Callable[[Optional[str], Optional[float]], None]] = None

    def ensure_directory(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def set_status_callback(self, callback: Optional[Callable[[Optional[str], Optional[float]], None]]) -> None:
        self.status_callback = callback
    def _notify_status(self, message: Optional[str], progress: Optional[float]) -> None:
        if self.status_callback is not None:
            try:
                self.status_callback(message, progress)
            except Exception:
                pass

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

    def load_entries(self, cv2_module: Optional[Any], detector: Optional[Any]) -> None:
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

        for path in files:
            try:
                with Image.open(path) as img:
                    rgb_image = img.convert("RGB")
            except OSError:
                continue

            face = extract_face_region(rgb_image, cv2_module, detector)
            embedding = compute_embedding(face)
            thumbnail = rgb_image.copy()
            thumbnail.thumbnail((360, 270), Image.LANCZOS)
            slug = path.stem
            meta_entry = metadata.get(slug, {})
            gender = meta_entry.get("gender") or SAMPLE_CELEBRITY_GENDERS.get(slug, "unknown")
            gender = gender.lower()
            if gender not in ("male", "female"):
                gender = "unknown"

            entry = CelebrityEntry(
                name=meta_entry.get("name", slug),
                image=rgb_image.copy(),
                embedding=embedding,
                path=path,
                thumbnail=thumbnail,
                gender=gender,
            )
            entries.append(entry)
            embeddings.append(embedding)

            if total:
                progress = len(entries) / total
                self._notify_status(f"Loading celebrity images {len(entries)}/{total}", progress)

        self.entries = entries
        if embeddings:
            self.embedding_matrix = np.vstack(embeddings)
        else:
            self.embedding_matrix = None

        for idx, entry in enumerate(self.entries):
            self.gender_indices.setdefault(entry.gender, []).append(idx)
        if total:
            self._notify_status(None, 1.0)

    def require_entries(self, cv2_module: Optional[Any], detector: Optional[Any]) -> None:
        if not self.entries:
            self.load_entries(cv2_module, detector)

        if not self.entries:
            raise FileNotFoundError(
                "No celebrity images were found. Add images to the dataset or download the curated set."
            )

    def best_match(
        self,
        source_image: Image.Image,
        cv2_module: Optional[Any],
        detector: Optional[Any],
        allowed_genders: Optional[List[str]] = None,
    ) -> CelebrityEntry:
        self.require_entries(cv2_module, detector)

        face = extract_face_region(source_image, cv2_module, detector)
        embedding = compute_embedding(face)
        return self.best_match_from_embedding(embedding, allowed_genders)

    def best_match_from_embedding(
        self,
        embedding: np.ndarray,
        allowed_genders: Optional[List[str]] = None,
    ) -> CelebrityEntry:
        if not self.entries or self.embedding_matrix is None or self.embedding_matrix.size == 0:
            raise FileNotFoundError(
                "No celebrity images were found. Add images to the dataset or download the curated set."
            )

        candidate_indices: List[int] = []
        if allowed_genders:
            for gender in allowed_genders:
                candidate_indices.extend(self.gender_indices.get(gender, []))
        if not candidate_indices:
            candidate_indices = list(range(len(self.entries)))

        subset = self.embedding_matrix[candidate_indices]
        diffs = subset - embedding
        distances = np.einsum("ij,ij->i", diffs, diffs)
        best_local_index = int(np.argmin(distances))
        best_index = candidate_indices[best_local_index]
        return self.entries[best_index]


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
        self.available_cameras: List[int] = []
        self.camera_index_var = StringVar()
        self.camera_capture: Optional[Any] = None
        self.camera_running = False
        self.camera_photo: Optional[ImageTk.PhotoImage] = None
        self.celebrity_photo: Optional[ImageTk.PhotoImage] = None
        self.latest_frame: Optional[Any] = None
        self.cv2: Optional[Any] = None
        self.face_detector: Optional[Any] = None
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
        self.gender_classifier = GenderClassifier()
        self.current_face_gender: str = "unknown"
        self.is_frozen = False
        self.frozen_frame: Optional[np.ndarray] = None
        self.freeze_button: Optional[ttk.Button] = None
        self.dataset_status_label: Optional[ttk.Label] = None
        self.dataset_progress: Optional[ttk.Progressbar] = None
        self.freeze_status_label: Optional[ttk.Label] = None

        self.dataset.ensure_directory()
        self.available_cameras = self._detect_cameras()
        default_cam = str(self.available_cameras[0]) if self.available_cameras else "0"
        self.camera_index_var.set(default_cam)
        self.dataset.set_status_callback(self._handle_dataset_status)
        self._build_menu()
        self._build_main_panel()
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

        self.gender_status = ttk.Label(container, text="Detected gender: unknown")
        self.gender_status.grid(column=0, row=5, columnspan=2, pady=(8, 0), sticky="w")

        self.dataset_status_label = ttk.Label(container, text="Dataset status: Ready")
        self.dataset_status_label.grid(column=0, row=6, columnspan=2, sticky="w")
        self.dataset_progress = ttk.Progressbar(
            container,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            value=0,
        )
        self.dataset_progress.grid(column=0, row=7, columnspan=2, sticky="ew", pady=(4, 0))

        self.freeze_status_label = ttk.Label(container, text="Mode: Idle", foreground="grey")
        self.freeze_status_label.grid(column=0, row=8, columnspan=2, sticky="w", pady=(8, 0))
        self._update_freeze_status("Idle", "grey")

        self.dataset_status_label = ttk.Label(container, text="Dataset status: Ready")
        self.dataset_status_label.grid(column=0, row=6, columnspan=2, sticky="w")
        self.dataset_progress = ttk.Progressbar(container, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.dataset_progress.grid(column=0, row=7, columnspan=2, sticky="ew", pady=(4, 0))

        self.freeze_status_label = ttk.Label(container, text="Mode: Idle", foreground="grey")
        self.freeze_status_label.grid(column=0, row=8, columnspan=2, sticky="w", pady=(8, 0))

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

        cv2_module = self.get_cv2_module()
        detector = self.ensure_face_detector(cv2_module)
        predicted_gender = self.gender_classifier.predict(np.array(source_image))
        if predicted_gender in ("male", "female"):
            allowed = [predicted_gender]
            self.gender_status.config(text=f"Detected gender: {predicted_gender}")
            self.current_face_gender = predicted_gender
        else:
            allowed = None
            self.gender_status.config(text="Detected gender: unknown")
            self.current_face_gender = "unknown"
        try:
            match_entry = self.match_celebrity(source_image, cv2_module, detector, allowed)
        except FileNotFoundError as exc:
            self.match_result.config(text=str(exc))
            return
        except Exception:
            self.match_result.config(text="Failed to compute a match for this image.")
            return

        self._apply_match_entry(match_entry)

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

    def ensure_face_detector(self, cv2_module: Optional[Any]) -> Optional[Any]:
        if cv2_module is None:
            return None

        if self.face_detector is not None:
            return self.face_detector

        cascade_path = getattr(cv2_module.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        if not Path(cascade_path).exists():
            messagebox.showwarning(
                "Face Detector Missing",
                "OpenCV Haar cascade not found. Facial tracking will fallback to center crops.",
            )
            return None

        detector = cv2_module.CascadeClassifier(cascade_path)
        if detector.empty():
            messagebox.showwarning(
                "Face Detector Error",
                "Failed to initialise Haar cascade classifier."
            )
            return None

        self.face_detector = detector
        return detector

    def capture_from_camera(self) -> None:
        if self.camera_running:
            return

        cv2_module = self.get_cv2_module(require=True)
        if cv2_module is None:
            return

        self.ensure_face_detector(cv2_module)

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
        self.image_status.config(text=f"Camera live preview active (index {int(self.camera_index_var.get())}).")
        try:
            self.dataset.require_entries(self.cv2, self.face_detector)
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
            cv2_module = self.get_cv2_module()
            detector = self.ensure_face_detector(cv2_module)
            self.dataset.load_entries(cv2_module, detector)
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
        self.current_face_gender = "unknown"
        self.is_frozen = False
        self.frozen_frame = None
        if self.freeze_button is not None:
            self.freeze_button.config(text="Freeze Frame")
        self.camera_label.config(text="Camera preview stopped.", image="")
        self.image_status.config(text="Camera stopped. Select a photo or restart the camera.")
        self.gender_status.config(text="Detected gender: unknown")
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

        detector = self.ensure_face_detector(self.cv2)
        face_region_rgb: Optional[np.ndarray] = None
        face_bbox: Optional[Tuple[int, int, int, int]] = None
        if detector is not None:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces[:1]:
                self.cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_bbox = (x, y, w, h)

        frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_region_rgb = frame_rgb[y : y + h, x : x + w]

        if face_region_rgb is None or face_region_rgb.size == 0:
            height, width = frame_rgb.shape[:2]
            side = min(height, width)
            top = (height - side) // 2
            left = (width - side) // 2
            face_region_rgb = frame_rgb[top : top + side, left : left + side]

        if face_region_rgb is not None and face_region_rgb.size > 0:
            self.latest_face_array = face_region_rgb.copy()
        else:
            self.latest_face_array = None

        predicted_gender = self.gender_classifier.predict(self.latest_face_array)
        if predicted_gender != self.current_face_gender:
            self.current_face_gender = predicted_gender
            self.gender_status.config(text=f"Detected gender: {predicted_gender}")

        self.latest_frame = frame_rgb
        preview_image = Image.fromarray(frame_rgb)
        preview_image = preview_image.resize((360, 270))

        self.camera_photo = ImageTk.PhotoImage(preview_image)
        self.camera_label.config(image=self.camera_photo, text="")
        if not self.is_frozen:
            self.root.after(30, self._update_camera_frame)

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

    def _process_live_match(self) -> None:
        self.match_job = None

        if not self.camera_running:
            return

        self._drain_match_results()

        if self.latest_face_array is not None:
            face_digest = int(np.sum(self.latest_face_array, dtype=np.uint64))
            if face_digest != self.last_sent_frame_digest:
                try:
                    self.match_request_queue.put_nowait(
                        (face_digest, self.latest_face_array.copy(), self.current_face_gender)
                    )
                    self.last_sent_frame_digest = face_digest
                except queue.Full:
                    pass

        if self.camera_running:
            self.match_job = self.root.after(self.match_interval_ms, self._process_live_match)

    def _drain_match_results(self) -> None:
        while True:
            try:
                digest, entry, detected_gender = self.match_result_queue.get_nowait()
            except queue.Empty:
                break
            if entry is None:
                if not self.dataset_warning_shown:
                    self.match_result.config(text="Add celebrity images to enable live matching.")
                    self.dataset_warning_shown = True
                continue
            self.dataset_warning_shown = False
            if detected_gender and detected_gender in ("male", "female"):
                self.current_face_gender = detected_gender
                self.gender_status.config(text=f"Detected gender: {detected_gender}")
            self._apply_match_entry(entry)

    def _match_worker_loop(self) -> None:
        while not self.match_worker_stop.is_set():
            try:
                digest, face_array, detected_gender = self.match_request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if digest is None or face_array is None or digest == -1:
                continue

            try:
                face_image = Image.fromarray(face_array)
                embedding = compute_embedding(face_image)
                allowed = None
                if detected_gender in ("male", "female"):
                    allowed = [detected_gender]
                match_entry = self.dataset.best_match_from_embedding(embedding, allowed)
            except FileNotFoundError:
                try:
                    self.match_result_queue.put_nowait((digest, None, detected_gender))
                except queue.Full:
                    pass
                continue
            except Exception:
                continue

            try:
                self.match_result_queue.put_nowait((digest, match_entry, detected_gender))
            except queue.Full:
                pass

    def _apply_match_entry(
        self,
        entry: CelebrityEntry,
        *,
        friendly_name: Optional[str] = None,
    ) -> None:
        if self.current_match_name == entry.name:
            return

        friendly = friendly_name or format_display_name(entry.name)

        self.current_match_name = entry.name
        self.dataset_warning_shown = False

        self.match_result.config(text=f"Closest celebrity match: {friendly}")
        self.display_celebrity_entry(entry, friendly)

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
        cv2_module: Optional[Any],
        detector: Optional[Any],
        allowed_genders: Optional[List[str]] = None,
    ) -> CelebrityEntry:
        return self.dataset.best_match(image, cv2_module, detector, allowed_genders)

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

        cv2_module = self.get_cv2_module()
        detector = self.ensure_face_detector(cv2_module)
        allowed = [self.current_face_gender] if self.current_face_gender in ("male", "female") else None

        try:
            match_entry = self.match_celebrity(frozen_image, cv2_module, detector, allowed)
        except FileNotFoundError as exc:
            self.match_result.config(text=str(exc))
            return
        except Exception:
            self.match_result.config(text="Matching failed for frozen frame.")
            return

        self._apply_match_entry(match_entry)
        self.match_result.config(text="Frozen frame analysis complete.")

    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = CelebrityMatcherApp(root)
    root.mainloop()
