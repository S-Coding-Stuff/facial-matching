# Face-to-Celebrity Matcher

An interactive Tkinter desktop app that matches a live (or uploaded) face image against a local library of celebrity portraits. Each new frame is compared against pre-computed embeddings, and the closest celebrity match is displayed in real time alongside gender information.

## Features

- Tkinter GUI with live camera preview, freeze/resume toggle, and inline status indicators.
- Celebrity dataset manager that caches embeddings and metadata (including gender) for fast retrieval.
- Hugging Face image-classification model (`prithivMLmods/Realistic-Gender-Classification`) for up-to-date gender detection.
- Asynchronous worker that performs face matching without blocking the UI.
- CLI scraper (`celebrity_scraper.py`) to download portraits + metadata from Wikimedia/Wikidata with face-count validation.

## Getting Started

1. **Create & activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install Python dependencies:**
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

From the project root (`/Users/samgrant/Desktop/Coding_Fun`):

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

- Replace Haar cascades with a modern face detector (e.g., MediaPipe or RetinaFace).
- Swap the histogram-based embedding with a learned face-recognition model (ArcFace, InsightFace).
- Add multi-face tracking, emotion/age recognition, and richer metadata.
- Export embeddings/metadata to SQLite or another DB for faster incremental updates.

See `HANDOFF.md` for more details on outstanding tasks and current architecture.
