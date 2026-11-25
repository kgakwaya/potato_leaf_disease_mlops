"""Train the custom CNN from scratch outside the notebook (optional).

    python -m src.train --epochs 40

Uses data/train (split 85/15 into train/validation) and evaluates on data/test.
The notebook is the main, documented training path; this script reproduces it.
"""
from __future__ import annotations

import argparse
import json

from src import config
from src.model import build_model, evaluate_model, save_artifacts, train_model
from src.preprocessing import (compute_class_weights, compute_insights, discover_class_names,
                               list_images, make_dataset, stratified_split)


def main(epochs: int, optimizer: str, lr: float):
    class_names = discover_class_names(config.TRAIN_DIR)
    if not class_names:
        raise SystemExit(f"No class folders found in {config.TRAIN_DIR}")
    n = len(class_names)
    paths, labels = list_images([config.TRAIN_DIR], class_names)
    tr_p, tr_l, va_p, va_l = stratified_split(paths, labels, 0.15)
    train_ds = make_dataset(tr_p, tr_l, n, augment=True, shuffle=True)
    val_ds = make_dataset(va_p, va_l, n)
    te_p, te_l = list_images([config.TEST_DIR], class_names)
    test_ds = make_dataset(te_p, te_l, n) if te_p else val_ds

    model = build_model(n, optimizer=optimizer, learning_rate=lr)
    model.summary()
    train_model(model, train_ds, val_ds, epochs, compute_class_weights(tr_l, n),
                checkpoint_path=config.MODELS_DIR / "best_checkpoint.h5")
    metrics = evaluate_model(model, test_ds, class_names)
    save_artifacts(model, class_names, metrics)
    compute_insights(class_names)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    main(a.epochs, a.optimizer, a.lr)
