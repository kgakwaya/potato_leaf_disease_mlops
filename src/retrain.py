"""Retraining pipeline.

Triggered from the UI / API (POST /retrain) or the command line:

    python -m src.retrain --epochs 5

Steps:
    1. Read the newly uploaded images recorded in the database (already
       verified, converted to RGB and resized at upload time).
    2. Build a training set = original data/train + all uploads, with the same
       preprocessing + augmentation as the notebook.
    3. Load the SAVED CUSTOM MODEL (models/potato_cnn.h5) as the pre-trained
       starting point and fine-tune it with a lower learning rate and early
       stopping.
    4. Evaluate old vs new model on data/test with the same metrics as the
       notebook. The new model is promoted only if it is not worse, and the old
       one is archived in models/archive/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src import config, database
from src.preprocessing import (compute_class_weights, compute_insights, list_images,
                               load_class_names, make_dataset, stratified_split)

_lock = threading.Lock()

SUMMARY_KEYS = ("accuracy", "loss", "precision_macro", "recall_macro", "f1_macro")


def _summary(metrics: dict) -> dict:
    return {k: metrics[k] for k in SUMMARY_KEYS if k in metrics}


def is_running() -> bool:
    return _lock.locked()


def run_retraining(epochs: int = config.RETRAIN_EPOCHS, learning_rate: float = config.RETRAIN_LR,
                   batch_size: int = 16, run_id: int | None = None,
                   on_finished=None) -> dict:
    if not _lock.acquire(blocking=False):
        raise RuntimeError("A retraining job is already running.")
    database.init_db()
    pending = database.pending_uploads()
    try:
        if run_id is None:
            run_id = database.start_run(len(pending), epochs)
        if len(pending) < config.MIN_NEW_IMAGES:
            raise ValueError(f"Need at least {config.MIN_NEW_IMAGES} new uploaded image(s) "
                             f"to retrain, found {len(pending)}.")
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Base model not found at {config.MODEL_PATH}")

        from src.model import compile_model, evaluate_model, get_callbacks, load_model, save_artifacts

        class_names = load_class_names()
        n_cls = len(class_names)

        # 1-2. Data: original training images + uploaded images
        database.update_run_message(run_id, "Preprocessing data")
        paths, labels = list_images([config.TRAIN_DIR, config.UPLOAD_DIR], class_names)
        tr_p, tr_l, va_p, va_l = stratified_split(paths, labels, val_fraction=0.15)
        train_ds = make_dataset(tr_p, tr_l, n_cls, batch_size, augment=True, shuffle=True)
        val_ds = make_dataset(va_p, va_l, n_cls, batch_size)
        te_p, te_l = list_images([config.TEST_DIR], class_names)
        test_ds = make_dataset(te_p, te_l, n_cls, batch_size) if te_p else val_ds

        # 3. Load the saved custom model as the pre-trained model
        database.update_run_message(run_id, "Evaluating current model")
        old_model = load_model(config.MODEL_PATH)
        compile_model(old_model, "adam", learning_rate)
        old_metrics = evaluate_model(old_model, test_ds, class_names)

        database.update_run_message(run_id, f"Fine-tuning on {len(tr_p)} images")
        model = old_model  # continue training from the saved weights
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=compute_class_weights(tr_l, n_cls),
                  callbacks=get_callbacks(None, patience=3), verbose=2)

        # 4. Evaluate + promote
        database.update_run_message(run_id, "Evaluating retrained model")
        new_metrics = evaluate_model(model, test_ds, class_names)
        promoted = new_metrics["accuracy"] >= old_metrics["accuracy"] - 0.01

        archive_dir = config.MODELS_DIR / "archive"
        archive_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        candidate_path = archive_dir / f"potato_cnn_run{run_id}_{stamp}.h5"
        model.save(candidate_path)
        if promoted:
            shutil.copy2(config.MODEL_PATH, archive_dir / f"potato_cnn_before_run{run_id}.h5")
            save_artifacts(model, class_names, new_metrics, config.MODEL_PATH,
                           extra={"retrain_run": run_id, "previous_metrics": _summary(old_metrics)})
        database.mark_uploads_used([u["id"] for u in pending], run_id)
        try:
            compute_insights(class_names)
        except Exception:  # visual stats are best-effort
            pass

        msg = ("New model promoted." if promoted else
               "New model was not better; kept the previous model (candidate archived).")
        database.finish_run(run_id, "completed", _summary(old_metrics), _summary(new_metrics),
                            promoted, str(config.MODEL_PATH if promoted else candidate_path), msg)
        result = {"run_id": run_id, "status": "completed", "promoted": promoted,
                  "old_metrics": _summary(old_metrics), "new_metrics": _summary(new_metrics),
                  "n_new_images": len(pending), "message": msg}
        if on_finished and promoted:
            on_finished()
        return result
    except Exception as exc:
        if run_id is not None:
            database.finish_run(run_id, "failed", message=f"{exc}")
        traceback.print_exc()
        return {"run_id": run_id, "status": "failed", "message": str(exc)}
    finally:
        _lock.release()


def start_background_retraining(epochs: int = config.RETRAIN_EPOCHS,
                                learning_rate: float = config.RETRAIN_LR,
                                on_finished=None) -> int:
    """Create the run record and start retraining in a background thread."""
    if is_running() or database.running_run():
        raise RuntimeError("A retraining job is already running.")
    database.init_db()
    pending = database.pending_uploads()
    if len(pending) < config.MIN_NEW_IMAGES:
        raise ValueError(f"Upload at least {config.MIN_NEW_IMAGES} image(s) before retraining.")
    run_id = database.start_run(len(pending), epochs)
    threading.Thread(target=run_retraining,
                     kwargs=dict(epochs=epochs, learning_rate=learning_rate, run_id=run_id,
                                 on_finished=on_finished),
                     daemon=True).start()
    return run_id


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrain the potato leaf model on uploaded data")
    ap.add_argument("--epochs", type=int, default=config.RETRAIN_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.RETRAIN_LR)
    args = ap.parse_args()
    print(json.dumps(run_retraining(args.epochs, args.lr), indent=2))
