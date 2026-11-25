"""Prediction utilities: load the saved custom model once and classify images."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from src import config
from src.preprocessing import load_class_names, preprocess_image, pretty_label


class Predictor:
    """Thread-safe wrapper around the saved Keras model.

    The model is loaded lazily on first use and can be hot-reloaded after a
    retraining run without restarting the API.
    """

    def __init__(self, model_path: Path | str = config.MODEL_PATH,
                 class_names_path: Path | str = config.CLASS_NAMES_PATH):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self._model = None
        self.class_names: list[str] = []
        self._lock = threading.Lock()
        self.loaded_at = None
        self._mtime = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def load(self):
        from datetime import datetime, timezone
        from src.model import load_model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train it with the notebook "
                "and copy potato_cnn.h5 + class_names.json into models/.")
        with self._lock:
            self._model = load_model(self.model_path, compile=False)
            self.class_names = load_class_names(self.class_names_path)
            self.loaded_at = datetime.now(timezone.utc).isoformat()
            self._mtime = self.model_path.stat().st_mtime
        return self

    def maybe_reload(self) -> bool:
        """Reload if the model file was replaced (e.g. retrained by another container)."""
        if self.model_path.exists() and self._mtime is not None \
                and self.model_path.stat().st_mtime != self._mtime:
            self.load()
            return True
        return False

    def reload(self):
        return self.load()

    def _ensure_loaded(self):
        if self._model is None:
            self.load()

    def predict_array(self, batch: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        with self._lock:
            return self._model(batch, training=False).numpy()

    def predict(self, source: bytes | str | Path) -> dict:
        """Predict the disease class of one leaf image (bytes or path)."""
        probs = self.predict_array(preprocess_image(source))[0]
        idx = int(np.argmax(probs))
        return {
            "class": self.class_names[idx],
            "label": pretty_label(self.class_names[idx]),
            "confidence": float(probs[idx]),
            "probabilities": {pretty_label(c): float(p)
                              for c, p in zip(self.class_names, probs)},
        }

    def predict_many(self, sources) -> list[dict]:
        return [self.predict(s) for s in sources]


_default: Predictor | None = None


def get_predictor() -> Predictor:
    global _default
    if _default is None:
        _default = Predictor()
    return _default


def predict_image(source: bytes | str | Path) -> dict:
    """Convenience function: predict_image('leaf.jpg')."""
    return get_predictor().predict(source)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(arg, "->", predict_image(arg))
