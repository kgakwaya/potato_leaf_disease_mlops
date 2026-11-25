"""Data acquisition + preprocessing for the Irish potato leaf disease classifier.

Preprocessing steps applied to EVERY image (training, prediction, uploads):
    1. Decode the file and verify it is a real image (corrupt files rejected).
    2. Convert to 3-channel RGB (drops alpha / converts greyscale).
    3. Resize to IMG_SIZE x IMG_SIZE (default 128x128) with bilinear filtering.
    4. Scale pixel values from [0, 255] to [0, 1] (float32).

Training-only steps (tf.data pipeline):
    5. Data augmentation: random flips, rotations, zoom, contrast and brightness.
    6. One-hot encoding of labels, shuffling, batching and prefetching.
"""
from __future__ import annotations

import io
import json
import random
import uuid
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

from src import config

# --------------------------------------------------------------------------- #
# Class names helpers
# --------------------------------------------------------------------------- #

def discover_class_names(directory: Path | str) -> list[str]:
    """Class names are the sub-folder names of the training directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    names = sorted(
        d.name for d in directory.iterdir()
        if d.is_dir() and not d.name.startswith(".") and any(_iter_images(d))
    )
    prefix = config.CLASS_PREFIX.lower()
    if prefix and any(n.lower().startswith(prefix) for n in names):
        names = [n for n in names if n.lower().startswith(prefix)]
    return names


def load_class_names(path: Path | str = config.CLASS_NAMES_PATH) -> list[str]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return discover_class_names(config.TRAIN_DIR)


def save_class_names(class_names: list[str], path: Path | str = config.CLASS_NAMES_PATH) -> None:
    Path(path).write_text(json.dumps(class_names, indent=2))


def pretty_label(name: str) -> str:
    """'Potato___Early_blight' -> 'Early blight'."""
    label = name.split("___")[-1] if "___" in name else name
    return label.replace("_", " ").strip().capitalize()


# --------------------------------------------------------------------------- #
# Single-image preprocessing (shared by training, prediction and uploads)
# --------------------------------------------------------------------------- #

def _iter_images(directory: Path) -> Iterable[Path]:
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS:
            yield p


def open_image(source: bytes | str | Path) -> Image.Image:
    """Steps 1-2: decode + verify + convert to RGB. Raises ValueError if invalid."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
            img.verify()  # detects truncated / corrupt files
            img = Image.open(io.BytesIO(source))
        else:
            with Image.open(source) as tmp:
                tmp.verify()
            img = Image.open(source)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError(f"Invalid or corrupt image: {exc}") from exc


def resize_image(img: Image.Image, size: int = config.IMG_SIZE) -> Image.Image:
    """Step 3: resize to a fixed square input size."""
    return img.resize((size, size), Image.BILINEAR)


def image_to_array(img: Image.Image) -> np.ndarray:
    """Step 4: scale to [0, 1] float32, shape (H, W, 3)."""
    return np.asarray(img, dtype=np.float32) / 255.0


def preprocess_image(source: bytes | str | Path, size: int = config.IMG_SIZE) -> np.ndarray:
    """Full inference preprocessing. Returns a batch of one: (1, size, size, 3)."""
    img = resize_image(open_image(source), size)
    return np.expand_dims(image_to_array(img), axis=0)


def is_valid_image(data: bytes) -> bool:
    try:
        open_image(data)
        return True
    except ValueError:
        return False


def save_uploaded_image(data: bytes, label: str, original_name: str = "",
                        upload_dir: Path = config.UPLOAD_DIR) -> Path:
    """Preprocess an uploaded image (verify, RGB, resize) and store it under
    uploads/<label>/ so it can be used for retraining."""
    img = resize_image(open_image(data))
    target_dir = Path(upload_dir) / label
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(original_name).stem[:40] if original_name else "img"
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    path = target_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


# --------------------------------------------------------------------------- #
# Dataset listing / splitting
# --------------------------------------------------------------------------- #

def list_images(directories: Iterable[Path | str], class_names: list[str]):
    """Return (paths, label_indices) for every image in <dir>/<class>/ folders."""
    paths, labels = [], []
    for directory in directories:
        directory = Path(directory)
        for idx, name in enumerate(class_names):
            class_dir = directory / name
            if not class_dir.exists():
                continue
            for p in sorted(_iter_images(class_dir)):
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def stratified_split(paths: list[str], labels: list[int], val_fraction: float = 0.15,
                     seed: int = config.SEED):
    """Simple stratified split that keeps class proportions in train/val."""
    rng = random.Random(seed)
    by_class: dict[int, list[str]] = {}
    for p, l in zip(paths, labels):
        by_class.setdefault(l, []).append(p)
    tr_p, tr_l, va_p, va_l = [], [], [], []
    for l, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
        va_p += items[:n_val]
        va_l += [l] * n_val
        tr_p += items[n_val:]
        tr_l += [l] * (len(items) - n_val)
    return tr_p, tr_l, va_p, va_l


def compute_class_weights(labels: list[int], num_classes: int) -> dict[int, float]:
    """Inverse-frequency class weights to counter class imbalance."""
    counts = Counter(labels)
    total = len(labels)
    return {i: total / (num_classes * counts[i]) if counts.get(i) else 1.0
            for i in range(num_classes)}


# --------------------------------------------------------------------------- #
# tf.data pipeline (TensorFlow imported lazily so the API starts fast)
# --------------------------------------------------------------------------- #

def get_augmentation_layer():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
    ], name="augmentation")


def make_dataset(paths: list[str], labels: list[int], num_classes: int,
                 batch_size: int = config.BATCH_SIZE, img_size: int = config.IMG_SIZE,
                 augment: bool = False, shuffle: bool = False):
    import tensorflow as tf

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = get_augmentation_layer()
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Feature extraction for the dashboard visualisations
# --------------------------------------------------------------------------- #

def image_features(path: str | Path) -> dict[str, float]:
    """Hand-crafted, interpretable features of a leaf image.

    - mean_r / mean_g / mean_b: average colour of each channel (0-255)
    - green_ratio: share of pixels where green dominates red and blue
      (healthy tissue is green; blight lesions are brown/black/yellow)
    - brightness: mean luminance (0-255)
    - edge_density: share of strong-edge pixels, a proxy for spots / texture
    """
    img = resize_image(open_image(path), 128)
    arr = np.asarray(img, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    green_mask = (g > r) & (g > b)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "green_ratio": float(green_mask.mean()),
        "brightness": float(lum.mean()),
        "edge_density": float((edges > 40).mean()),
    }


def compute_insights(class_names: list[str] | None = None, sample_per_class: int = 120,
                     save: bool = True) -> dict:
    """Aggregate dataset statistics used by the UI's visualisations."""
    class_names = class_names or load_class_names()
    rng = random.Random(config.SEED)
    counts = {}
    for split, directory in (("train", config.TRAIN_DIR), ("test", config.TEST_DIR),
                             ("uploads", config.UPLOAD_DIR)):
        counts[split] = {c: len(list(_iter_images(Path(directory) / c)))
                         if (Path(directory) / c).exists() else 0 for c in class_names}

    features: dict[str, dict[str, list[float]]] = {}
    for c in class_names:
        files = []
        for directory in (config.TRAIN_DIR, config.UPLOAD_DIR):
            if (Path(directory) / c).exists():
                files += list(_iter_images(Path(directory) / c))
        rng.shuffle(files)
        rows = []
        for f in files[:sample_per_class]:
            try:
                rows.append(image_features(f))
            except ValueError:
                continue
        features[c] = {k: [row[k] for row in rows] for k in
                       ("mean_r", "mean_g", "mean_b", "green_ratio", "brightness", "edge_density")}

    summary = {
        c: {k: (float(np.mean(v)) if v else None) for k, v in feats.items()}
        for c, feats in features.items()
    }
    bins = np.linspace(0, 1, 21)
    green_hist = {c: np.histogram(features[c]["green_ratio"], bins=bins)[0].tolist()
                  for c in class_names}
    insights = {
        "class_names": class_names,
        "display_names": {c: pretty_label(c) for c in class_names},
        "counts": counts,
        "feature_means": summary,
        "green_ratio_hist": {"bins": bins.round(2).tolist(), "counts": green_hist},
        "samples_used": {c: len(features[c]["green_ratio"]) for c in class_names},
    }
    if save:
        config.INSIGHTS_PATH.write_text(json.dumps(insights, indent=2))
    return insights
