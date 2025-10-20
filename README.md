# Face-to-Celebrity Matcher

An interactive Tkinter desktop app that matches a live (or uploaded) face image against a local library of celebrity portraits. Each new frame is compared against pre-computed embeddings, and the closest celebrity match is displayed in real time alongside smoothed gender predictions and an interpretable similarity score.

## Features

- Tkinter GUI with live camera preview, freeze/resume toggle, and inline status indicators (including dataset loading progress).
- OpenCV YuNet (downloaded automatically from Hugging Face via `huggingface-hub`) for fast CPU-only face detection.
- ArcFace ResNet-100 INT8 (ONNX + `onnxruntime`) for generating 512‑D embeddings, pre-warmed for low-latency cosine similarity lookups via Annoy.
- Lightweight MobileNetV3 gender classifier (INT8 ONNX) with exponential smoothing to stabilise frame-to-frame predictions and provide confidence estimates.
- Cosine similarity gauge that translates the raw embedding score into an easy-to-read percentage plus qualitative band (“very low” → “high”).
- Automatic dataset refresh: new celebrity images or metadata dropped into `celebrity_dataset/` are detected on the fly and trigger a background rebuild of embeddings.
- Persistent cache (`celebrity_dataset/.cache/`) stores face crops, embeddings, and Annoy indexes so restarts reuse precomputed data.
- Asynchronous worker that performs matching without blocking the UI.
- CLI scraper (`celebrity_scraper.py`) to download portraits + metadata from Wikimedia/Wikidata with face-count validation.

## Getting Started

1. **Create & activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install Python dependencies (note the use of opencv-contrib + huggingface-hub):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Populate the dataset (optional):**
   ```bash
   python celebrity_scraper.py "Zendaya" "Ryan Gosling" --output-dir celebrity_dataset
   ```

4. **Run the matcher UI:**
   ```bash
   python facial_matching.py
   ```
   On first launch the app downloads the YuNet ONNX model from the Hugging Face repo defined in `FaceDetector.MODEL_REPO` (currently `onnx-community/face-detection-yunet`). If your network blocks the download, fetch `face_detection_yunet_2023mar.onnx` manually from that repository and drop it into the `models/` directory before starting the app.

   The first run also pulls the ArcFace embedding model (`arcfaceresnet100-8.onnx`) from the Hugging Face repo `onnx-community/arcface-resnet100`. If this download fails—or if you prefer a newer file such as `arcfaceresnet100-11-int8.onnx`—download it manually and place it in the `models/` directory. The app will automatically use the int8 variant when present.

5. **Provide the gender classifier ONNX (once):**

   Drop a compact two-class gender classifier into `models/`. The app now prefers the ViT-based weights from `onnx-community/gender-classification-ONNX` (copy `onnx/model.onnx` or `onnx/model_int8.onnx` to `models/model_int8.onnx`). If no file is present it falls back to the lighter MobileNet options or to downloading from Hugging Face. Without an ONNX file, gender output reverts to embedding-centroid heuristics with confidence disabled.

## Repository Structure

```
.
├── facial_matching.py        # Tkinter app
├── celebrity_scraper.py      # CLI scraper for Wikimedia portraits
├── celebrity_dataset/        # Local celebrity portraits (ignored by git)
├── models/                   # Downloaded models (ignored by git)
├── requirements.txt
├── README.md
└── HANDOFF.md                # Current handoff context for collaborators
```

## Creating a New Git Repository

From the project root:

```bash
git init
git add facial_matching.py celebrity_scraper.py README.md requirements.txt HANDOFF.md .gitignore
git commit -m "Initial commit for face matching project"
```

Then create a new remote repository (e.g., GitHub) and push:

```bash
git remote add origin <REMOTE_URL>
git branch -M main
git push -u origin main
```

## Future Enhancements

- Add multi-face tracking, emotion/age recognition, and richer metadata overlays.
- Export embeddings/metadata to SQLite (or similar) for faster incremental updates and versioning.
- Swap in GPU-optimised detectors/embedders (ArcFace/InsightFace) and accelerate nearest-neighbour search with FAISS when datasets grow.
- Bundle the app with PyInstaller or Briefcase for one-click installs.

See `HANDOFF.md` for more details on outstanding tasks and current architecture.
