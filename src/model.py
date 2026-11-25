"""Model creation, training and evaluation.

The CNN is a custom architecture trained FROM SCRATCH (no ImageNet weights).
Optimisation techniques used:
    * L2 weight regularisation on conv + dense kernels
    * Batch normalisation after every convolution
    * Dropout (spatial blocks + dense head)
    * Data augmentation (see preprocessing.get_augmentation_layer)
    * Choice of optimiser (Adam / RMSprop / SGD-momentum) + tuned learning rate
    * Early stopping, ReduceLROnPlateau and best-checkpoint saving
    * Class weights for class imbalance, label smoothing
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import config


def get_optimizer(name: str = "adam", learning_rate: float = 1e-3):
    import tensorflow as tf
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer {name}")


def compile_model(model, optimizer: str = "adam", learning_rate: float = 1e-3,
                  label_smoothing: float = 0.05):
    import tensorflow as tf
    model.compile(
        optimizer=get_optimizer(optimizer, learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def build_model(num_classes: int, img_size: int = config.IMG_SIZE, l2: float = 1e-4,
                dropout: float = 0.4, filters=(32, 64, 128, 256),
                optimizer: str = "adam", learning_rate: float = 1e-3):
    """Custom CNN: 4 conv blocks [Conv-BN-ReLU x2, MaxPool, Dropout] -> GAP -> Dense."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(l2) if l2 else None
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    for i, f in enumerate(filters):
        for j in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False,
                              kernel_regularizer=reg, name=f"conv{i+1}_{j+1}")(x)
            x = layers.BatchNormalization(name=f"bn{i+1}_{j+1}")(x)
            x = layers.Activation("relu", name=f"relu{i+1}_{j+1}")(x)
        x = layers.MaxPooling2D(name=f"pool{i+1}")(x)
        x = layers.Dropout(min(0.1 + 0.05 * i, 0.3), name=f"drop{i+1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg, name="dense")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="potato_leaf_cnn")
    return compile_model(model, optimizer, learning_rate)


def build_vanilla_model(num_classes: int, img_size: int = config.IMG_SIZE):
    """Baseline with NO regularisation / BN / dropout, used for comparison."""
    import tensorflow as tf
    from tensorflow.keras import layers
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="vanilla_cnn")
    return compile_model(model, "sgd", 1e-2, label_smoothing=0.0)


def get_callbacks(checkpoint_path: Path | str | None = None, patience: int = 6):
    import tensorflow as tf
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=max(2, patience // 2),
                                             min_lr=1e-6, verbose=1),
    ]
    if checkpoint_path:
        cbs.append(tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path),
                                                      monitor="val_loss",
                                                      save_best_only=True, verbose=0))
    return cbs


def train_model(model, train_ds, val_ds, epochs: int = 40, class_weight=None,
                checkpoint_path=None, patience: int = 6):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        class_weight=class_weight,
                        callbacks=get_callbacks(checkpoint_path, patience), verbose=2)
    return history


def predict_dataset(model, dataset):
    """Return (y_true, y_pred, y_prob) for a one-hot labelled dataset."""
    y_true, y_prob = [], []
    for x, y in dataset:
        y_prob.append(model.predict(x, verbose=0))
        y_true.append(np.argmax(y.numpy(), axis=1))
    y_prob = np.concatenate(y_prob) if y_prob else np.zeros((0, 0))
    y_true = np.concatenate(y_true) if y_true else np.zeros(0, dtype=int)
    return y_true, y_prob.argmax(axis=1) if len(y_prob) else y_true, y_prob


def evaluate_model(model, dataset, class_names: list[str]) -> dict:
    """Accuracy, loss, precision, recall, F1 (macro + weighted + per class),
    ROC-AUC and the confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)
    results = model.evaluate(dataset, verbose=0, return_dict=True)
    y_true, y_pred, y_prob = predict_dataset(model, dataset)
    labels = list(range(len(class_names)))
    metrics = {
        "loss": float(results.get("loss", np.nan)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=labels,
                                           target_names=class_names, output_dict=True,
                                           zero_division=0),
        "n_samples": int(len(y_true)),
    }
    try:
        if len(set(y_true)) > 1:
            metrics["roc_auc_ovr"] = float(roc_auc_score(
                np.eye(len(class_names))[y_true], y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        pass
    return metrics


def save_artifacts(model, class_names: list[str], metrics: dict | None = None,
                   model_path: Path | str = config.MODEL_PATH, extra: dict | None = None):
    """Save the model (.h5), its class names and the evaluation metrics."""
    from src.preprocessing import save_class_names
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    save_class_names(class_names, model_path.parent / "class_names.json")
    if metrics is not None:
        payload = {"saved_at": datetime.now(timezone.utc).isoformat(),
                   "model_file": model_path.name, "metrics": metrics, **(extra or {})}
        (model_path.parent / "metrics.json").write_text(json.dumps(payload, indent=2))
    return model_path


def load_model(model_path: Path | str = config.MODEL_PATH, compile: bool = False):
    import tensorflow as tf
    return tf.keras.models.load_model(str(model_path), compile=compile)
