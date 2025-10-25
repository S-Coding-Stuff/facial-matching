from tkinter import Tk, Menu, ttk, filedialog, messagebox, StringVar
from pathlib import Path
from typing import Optional, Any, List, Tuple
import queue
import threading
import time

import numpy as np
from PIL import Image, ImageTk

from face_detection import FaceDetector, extract_face_region
from gender import GenderClassifier, GenderSmoother
from matcher import (
    CelebrityDataset,
    CelebrityEntry,
    FaceEmbedder,
    DATASET_DIR,
    SAMPLE_CELEBRITY_URLS,
    format_display_name,
)


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
        self.capture_button: Optional[ttk.Button] = None
        self.load_photo_button: Optional[ttk.Button] = None
        self.use_camera_button: Optional[ttk.Button] = None
        self.stop_camera_button: Optional[ttk.Button] = None
        self.dataset_status_label: Optional[ttk.Label] = None
        self.dataset_progress: Optional[ttk.Progressbar] = None
        self.freeze_status_label: Optional[ttk.Label] = None
        self.similarity_label: Optional[ttk.Label] = None
        self.gender_status: Optional[ttk.Label] = None
        self.gender_classifier: Optional[GenderClassifier] = None
        self.gender_classifier_warning: Optional[str] = None
        self.gender_smoother = GenderSmoother()
        self.last_sent_frame_time: float = 0.0
        self.snapshot_photo: Optional[ImageTk.PhotoImage] = None

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

    def _set_button_state(self, button: Optional[ttk.Button], enabled: bool) -> None:
        if button is None:
            return
        if enabled:
            button.state(["!disabled"])
        else:
            button.state(["disabled"])

    def _disable_controls_for_snapshot(self) -> None:
        for button in (self.load_photo_button, self.use_camera_button, self.stop_camera_button, self.capture_button):
            self._set_button_state(button, False)

    def _restore_controls_after_snapshot(self) -> None:
        self._set_button_state(self.load_photo_button, True)
        self._set_button_state(self.use_camera_button, True)
        self._set_button_state(self.stop_camera_button, False)
        self._set_button_state(self.capture_button, False)

    def _show_captured_frame(self, frame_array: np.ndarray) -> None:
        try:
            captured_image = Image.fromarray(frame_array.astype("uint8"))
        except Exception:
            return
        preview = captured_image.resize((360, 270))
        self.snapshot_photo = ImageTk.PhotoImage(preview)
        self.camera_photo = self.snapshot_photo
        self.camera_label.config(image=self.snapshot_photo, text="Captured photo")
        self.image_status.config(text="Captured photo ready.")

    def _bucket_similarity(self, score: float) -> str:
        if score >= 0.85:
            return "high"
        if score >= 0.65:
            return "medium"
        if score >= 0.5:
            return "low"
        return "very low"

    def _format_similarity(self, cosine_similarity: Optional[float], percentile: Optional[float] = None) -> str:
        if cosine_similarity is None:
            return "Similarity: --"
        clamped = max(-1.0, min(float(cosine_similarity), 1.0))
        normalized = (clamped + 1.0) / 2.0
        normalized = max(0.0, min(normalized, 1.0))
        bucket = self._bucket_similarity(normalized)
        detail = f"Similarity: {normalized * 100:.1f}% ({bucket}"
        if percentile is not None:
            percentile = max(0.0, min(float(percentile), 1.0))
            detail += f", top {percentile * 100:.1f}%"
        detail += ")"
        return detail

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

        self.load_photo_button = ttk.Button(controls, text="Load Photo", command=self.load_image)
        self.load_photo_button.grid(column=0, row=0, padx=(0, 8), sticky="ew")
        self.use_camera_button = ttk.Button(controls, text="Use Camera", command=self.capture_from_camera)
        self.use_camera_button.grid(column=1, row=0, padx=4, sticky="ew")
        self.stop_camera_button = ttk.Button(controls, text="Stop Camera", command=self.stop_camera)
        self.stop_camera_button.grid(column=2, row=0, padx=(8, 0), sticky="ew")
        self.capture_button = ttk.Button(controls, text="Capture Photo", command=self.capture_snapshot)
        self.capture_button.grid(column=3, row=0, padx=(8, 0), sticky="ew")
        if self.stop_camera_button is not None:
            self.stop_camera_button.state(["disabled"])
        if self.capture_button is not None:
            self.capture_button.state(["disabled"])
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
            match_entry, similarity, percentile = self.match_celebrity(source_image)
        except FileNotFoundError as exc:
            self.match_result.config(text=str(exc))
            return
        except Exception:
            self.match_result.config(text="Failed to compute a match for this image.")
            return

        self._apply_match_entry(match_entry, similarity, percentile)
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
        self._set_button_state(self.use_camera_button, False)
        self._set_button_state(self.stop_camera_button, True)
        self._set_button_state(self.capture_button, True)

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
        self.camera_label.config(text="Camera preview stopped.", image="")
        self.image_status.config(text="Camera stopped. Select a photo or restart the camera.")
        self._update_gender_status("unknown", 0.0)
        if self.similarity_label is not None:
            self.similarity_label.config(text="Similarity: --")
        self._set_button_state(self.capture_button, False)
        self._set_button_state(self.stop_camera_button, False)
        self._set_button_state(self.use_camera_button, True)
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
        if self.camera_running:
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
                (
                    digest,
                    entry,
                    detected_gender,
                    similarity,
                    percentile,
                    classifier_confidence,
                    classifier_probabilities,
                ) = self.match_result_queue.get_nowait()
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
            self._apply_match_entry(entry, similarity, percentile)
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
                match_entry, similarity, percentile = self.dataset.best_match_from_embedding(embedding, allowed)
            except FileNotFoundError:
                try:
                    self.match_result_queue.put_nowait((digest, None, detected_gender, None, None, 0.0, None))
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
                        percentile,
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
        percentile: Optional[float] = None,
        *,
        friendly_name: Optional[str] = None,
    ) -> None:
        friendly = friendly_name or format_display_name(entry.name)

        self.current_match_name = entry.name
        self.dataset_warning_shown = False

        self.match_result.config(text=f"Closest celebrity match: {friendly}")
        self.display_celebrity_entry(entry, friendly)
        if self.similarity_label is not None:
            self.similarity_label.config(text=self._format_similarity(similarity, percentile))

    def run_match(self) -> None:
        messagebox.showinfo(
            "Live Matching",
            "Live matching runs automatically when the camera is active.",
        )

    def display_celebrity_entry(self, entry: CelebrityEntry, friendly_name: Optional[str] = None) -> None:
        display_image = entry.primary_thumbnail.copy() if entry.primary_thumbnail else entry.primary_image.copy()
        resolved_name = friendly_name or format_display_name(entry.name)
        self.celebrity_photo = ImageTk.PhotoImage(display_image)
        self.celebrity_label.config(image=self.celebrity_photo, text=resolved_name)

    def match_celebrity(
        self,
        image: Image.Image,
        allowed_genders: Optional[List[str]] = None,
    ) -> Tuple[CelebrityEntry, float, float]:
        return self.dataset.best_match(
            image,
            self.face_detector,
            self.face_embedder,
            allowed_genders,
            self.gender_classifier,
        )

    def capture_snapshot(self) -> None:
        if not self.camera_running:
            messagebox.showwarning("Capture Photo", "Start the camera before capturing a photo.")
            return
        if self.latest_frame is None:
            messagebox.showwarning("Capture Photo", "No camera frame available to capture.")
            return

        frame_copy = self.latest_frame.copy()
        self.latest_frame = frame_copy.copy()
        self._disable_controls_for_snapshot()
        if self.camera_running:
            self._stop_live_matching()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.camera_running = False
        self.latest_face_array = None
        self.last_sent_frame_digest = None
        self.last_sent_frame_time = 0.0
        self._show_captured_frame(frame_copy)

        self._update_freeze_status("Capturing", "orange")
        self.match_result.config(text="Analyzing captured photo...")
        should_restart = False

        def worker() -> None:
            try:
                snapshot_image = Image.fromarray(frame_copy.astype("uint8"))
            except Exception:
                self.root.after(
                    0,
                    lambda: self._conclude_snapshot_error(should_restart, "Unable to process captured photo."),
                )
                return

            try:
                self.dataset.require_entries(self.face_detector, self.face_embedder, self.gender_classifier, wait=True)
            except Exception as exc:
                message = str(exc) if str(exc) else "Celebrity dataset is unavailable."
                self.root.after(0, lambda: self._conclude_snapshot_error(should_restart, message))
                return

            try:
                face_crop = extract_face_region(snapshot_image, self.face_detector)
                face_array = np.array(face_crop)
            except Exception:
                self.root.after(
                    0,
                    lambda: self._conclude_snapshot_error(should_restart, "No face detected in captured photo."),
                )
                return

            try:
                embedding = self.face_embedder.embed(face_array)
            except Exception:
                self.root.after(
                    0,
                    lambda: self._conclude_snapshot_error(should_restart, "Failed to compute face embedding."),
                )
                return

            detected_gender = "unknown"
            detected_confidence = 0.0
            if self.gender_classifier is not None:
                try:
                    detected_gender, detected_confidence, _ = self.gender_classifier.classify(face_crop)
                except Exception:
                    detected_gender = "unknown"
                    detected_confidence = 0.0

            allowed: Optional[List[str]] = None
            if detected_gender in ("male", "female") and detected_confidence >= 0.7:
                allowed = [detected_gender]
            elif self.current_face_gender in ("male", "female") and self.current_gender_confidence >= 0.7:
                allowed = [self.current_face_gender]

            try:
                match_entry, similarity, percentile = self.dataset.best_match_from_embedding(embedding, allowed)
            except Exception:
                self.root.after(
                    0,
                    lambda: self._conclude_snapshot_error(should_restart, "Matching failed for captured photo."),
                )
                return

            self.root.after(
                0,
                lambda: self._conclude_snapshot_success(
                    should_restart,
                    match_entry,
                    similarity,
                    percentile,
                    detected_gender,
                    detected_confidence,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _conclude_snapshot_success(
        self,
        should_restart: bool,
        entry: CelebrityEntry,
        similarity: float,
        percentile: float,
        detected_gender: str,
        detected_confidence: float,
    ) -> None:
        self._apply_match_entry(entry, similarity, percentile)
        if detected_gender in ("male", "female"):
            self._update_gender_status(detected_gender, detected_confidence)
        self.match_result.config(text="Snapshot analysis complete.")
        self._finalize_snapshot_ui(should_restart)

    def _conclude_snapshot_error(self, should_restart: bool, message: str) -> None:
        self.match_result.config(text=message)
        self._finalize_snapshot_ui(should_restart)

    def _finalize_snapshot_ui(self, should_restart: bool) -> None:
        self._restore_controls_after_snapshot()
        self.image_status.config(text="Captured photo displayed. Click 'Use Camera' to capture again.")
        if should_restart and self.camera_running:
            self._start_live_matching()
            self._update_freeze_status("Live", "green")
        else:
            self._update_freeze_status("Idle", "grey")

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
