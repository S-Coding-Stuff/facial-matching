"""Download celebrity portrait images into the local dataset.

The script uses the public Wikimedia API to resolve portrait thumbnails for the
supplied celebrity names. Each downloaded image is validated (if OpenCV is
available) to ensure exactly one face is present before saving it into the
``celebrity_dataset`` folder that powers ``facial_matching.py``.

Usage examples::

    # Download a couple of celebrities
    python celebrity_scraper.py Zendaya "Ryan Gosling"

    # Read names from a newline-delimited text file
    python celebrity_scraper.py --names-file celebrities.txt

Dependencies::

    pip install requests pillow opencv-python

Always respect content licenses: Wikimedia assets often have reuse
restrictions. Review attribution requirements before distributing any images.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - OpenCV optional at runtime
    cv2 = None


API_URL = "https://en.wikipedia.org/w/api.php"
DEFAULT_DATASET_DIR = Path(__file__).with_name("celebrity_dataset")
USER_AGENT = "CelebrityScraper/1.0 (+https://example.com)"
REQUEST_TIMEOUT = 20
REQUEST_DELAY = 1.0  # seconds between requests to be polite


@dataclass
class DownloadResult:
    name: str
    page_title: Optional[str]
    image_url: Optional[str]
    filename: Optional[Path]
    gender: Optional[str] = None
    error: Optional[str] = None

    def success(self) -> bool:
        return self.image_url is not None and self.filename is not None and self.error is None


def normalise_slug(value: str) -> str:
    slug = value.strip().lower().replace(" ", "_")
    return "".join(ch for ch in slug if ch.isalnum() or ch in {"_", "-"})


def _parse_name_input(raw: str) -> list[str]:
    pieces = []
    for chunk in raw.split(","):
        cleaned = chunk.strip().replace("_", " ")
        if cleaned:
            pieces.append(cleaned)
    return pieces


def load_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    for raw in args.names:
        names.extend(_parse_name_input(raw))
    if args.names_file:
        file_path = Path(args.names_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Names file not found: {file_path}")
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    names.extend(_parse_name_input(stripped))
    return names


def fetch_wikimedia_thumbnail(session: requests.Session, name: str) -> tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": name,
        "gsrlimit": 1,
        "prop": "pageimages|pageprops",
        "piprop": "thumbnail",
        "pithumbsize": 640,
    }
    response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None, None, None, None

    page = next(iter(pages.values()))
    title = page.get("title")
    pageid = page.get("pageid")
    thumbnail = page.get("thumbnail", {})
    source = thumbnail.get("source")
    wikidata_id = None
    pageprops = page.get("pageprops", {})
    if pageprops:
        wikidata_id = pageprops.get("wikibase_item")
    if not source:
        return title, None, pageid, wikidata_id
    return title, source, pageid, wikidata_id


def fetch_gender_from_wikidata(session: requests.Session, wikidata_id: Optional[str]) -> Optional[str]:
    if not wikidata_id:
        return None

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json"
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return None

    data = response.json()
    entities = data.get("entities", {})
    entity = entities.get(wikidata_id, {})
    claims = entity.get("claims", {})
    gender_claims = claims.get("P21", [])
    if not gender_claims:
        return None

    mainsnak = gender_claims[0].get("mainsnak", {})
    datavalue = mainsnak.get("datavalue", {})
    value = datavalue.get("value", {})
    gender_id = value.get("id")
    if gender_id == "Q6581097":
        return "male"
    if gender_id == "Q6581072":
        return "female"
    return None


def validate_single_face(image_bytes: bytes) -> bool:
    if cv2 is None:
        return True  # Cannot validate without OpenCV

    cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
    if not Path(cascade_path).exists():
        return True

    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return True

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return len(faces) == 1


def download_image(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def persist_image(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as img:
        rgb = img.convert("RGB")
        rgb.save(path, format="JPEG", quality=90)


def update_metadata(output_dir: Path, slug: str, name: str, gender: Optional[str]) -> None:
    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

    metadata[slug] = {"name": name, "gender": gender or "unknown"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def scrape_celebrities(names: Iterable[str], output_dir: Path) -> list[DownloadResult]:
    names_list = list(names)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    results: list[DownloadResult] = []
    total = len(names_list)

    if tqdm is not None and total > 0:
        iterator = tqdm(names_list, desc="Scraping celebrities", unit="name")
    else:
        iterator = names_list

    for index, name in enumerate(iterator, start=1):
        if tqdm is not None and total > 0:
            iterator.set_postfix_str(name)
        try:
            title, image_url, page_id, wikidata_id = fetch_wikimedia_thumbnail(session, name)
        except requests.RequestException as exc:
            results.append(DownloadResult(name=name, page_title=None, image_url=None, filename=None, gender=None, error=f"HTTP error: {exc}"))
            continue

        if not image_url:
            results.append(DownloadResult(name=name, page_title=title, image_url=None, filename=None, gender=None, error="No thumbnail located"))
            continue

        try:
            image_bytes = download_image(session, image_url)
        except requests.RequestException as exc:
            results.append(DownloadResult(name=name, page_title=title, image_url=image_url, filename=None, gender=None, error=f"Download error: {exc}"))
            continue

        if not validate_single_face(image_bytes):
            results.append(DownloadResult(name=name, page_title=title, image_url=image_url, filename=None, gender=None, error="Image does not contain exactly one face"))
            continue

        gender = fetch_gender_from_wikidata(session, wikidata_id)
        slug = normalise_slug(title or name)
        target_path = output_dir / f"{slug}.jpg"

        try:
            persist_image(image_bytes, target_path)
        except Exception as exc:
            results.append(DownloadResult(name=name, page_title=title, image_url=image_url, filename=None, gender=gender, error=f"Failed to save image: {exc}"))
            continue

        update_metadata(output_dir, slug, title or name, gender)

        results.append(DownloadResult(name=name, page_title=title, image_url=image_url, filename=target_path, gender=gender))

        if index < total:
            time.sleep(REQUEST_DELAY)
    if tqdm is not None and total > 0:
        iterator.close()

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download celebrity portraits into the local dataset")
    parser.add_argument("names", nargs="*", help="Celebrity names to fetch")
    parser.add_argument("--names-file", help="Path to a file containing additional celebrity names (one per line)")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Destination directory for downloaded images (default: celebrity_dataset)",
    )

    args = parser.parse_args(argv)
    names = load_names(args)
    if not names:
        parser.error("Please provide at least one celebrity name or a --names-file")

    output_dir = Path(args.output_dir)
    print(f"Saving images to {output_dir.resolve()}")

    results = scrape_celebrities(names, output_dir)

    successes = sum(1 for item in results if item.success())
    failures = len(results) - successes

    for result in results:
        status = "OK" if result.success() else "WARN"
        info = result.filename.name if result.filename else result.error
        gender_note = f" ({result.gender})" if result.gender else ""
        print(f"[{status}] {result.name}{gender_note} -> {info}")

    print(f"Completed: {successes} succeeded, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
