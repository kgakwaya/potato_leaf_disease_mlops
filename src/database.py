"""SQLite storage for uploaded images, training runs and prediction logs.

SQLite keeps the project free to run anywhere. Every uploaded file is recorded
here (path, label, size, status) so the retraining job knows which new data
it has to learn from.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    label TEXT NOT NULL,
    size_bytes INTEGER,
    uploaded_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | used
    used_in_run INTEGER
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,                     -- running | completed | failed
    n_new_images INTEGER,
    epochs INTEGER,
    old_metrics TEXT,
    new_metrics TEXT,
    promoted INTEGER,
    model_path TEXT,
    message TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT,
    predicted_class TEXT,
    confidence REAL,
    latency_ms REAL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as c:
        c.executescript(SCHEMA)


# ----- uploads -------------------------------------------------------------- #

def add_upload(filename: str, stored_path: str, label: str, size_bytes: int) -> int:
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO uploads (filename, stored_path, label, size_bytes, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)", (filename, stored_path, label, size_bytes, _now()))
        return cur.lastrowid


def pending_uploads() -> list[dict]:
    with get_conn() as c:
        rows = c.execute("SELECT * FROM uploads WHERE status='pending' ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def mark_uploads_used(ids: list[int], run_id: int):
    if not ids:
        return
    with get_conn() as c:
        c.executemany("UPDATE uploads SET status='used', used_in_run=? WHERE id=?",
                      [(run_id, i) for i in ids])


def upload_summary() -> dict:
    with get_conn() as c:
        rows = c.execute("SELECT label, status, COUNT(*) n FROM uploads "
                         "GROUP BY label, status").fetchall()
        recent = c.execute("SELECT id, filename, label, status, uploaded_at FROM uploads "
                           "ORDER BY id DESC LIMIT 20").fetchall()
    summary: dict = {"by_label": {}, "pending": 0, "used": 0}
    for r in rows:
        summary["by_label"].setdefault(r["label"], {})[r["status"]] = r["n"]
        summary[r["status"]] = summary.get(r["status"], 0) + r["n"]
    summary["recent"] = [dict(r) for r in recent]
    return summary


# ----- training runs -------------------------------------------------------- #

def start_run(n_new_images: int, epochs: int) -> int:
    with get_conn() as c:
        cur = c.execute("INSERT INTO training_runs (started_at, status, n_new_images, epochs) "
                        "VALUES (?, 'running', ?, ?)", (_now(), n_new_images, epochs))
        return cur.lastrowid


def finish_run(run_id: int, status: str, old_metrics=None, new_metrics=None,
               promoted: bool | None = None, model_path: str | None = None,
               message: str = ""):
    with get_conn() as c:
        c.execute("UPDATE training_runs SET finished_at=?, status=?, old_metrics=?, "
                  "new_metrics=?, promoted=?, model_path=?, message=? WHERE id=?",
                  (_now(), status, json.dumps(old_metrics) if old_metrics else None,
                   json.dumps(new_metrics) if new_metrics else None,
                   None if promoted is None else int(promoted), model_path, message, run_id))


def update_run_message(run_id: int, message: str):
    with get_conn() as c:
        c.execute("UPDATE training_runs SET message=? WHERE id=?", (message, run_id))


def list_runs(limit: int = 10) -> list[dict]:
    with get_conn() as c:
        rows = c.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?",
                         (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("old_metrics", "new_metrics"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def running_run() -> dict | None:
    with get_conn() as c:
        r = c.execute("SELECT * FROM training_runs WHERE status='running' "
                      "ORDER BY id DESC LIMIT 1").fetchone()
    return dict(r) if r else None


# ----- predictions ---------------------------------------------------------- #

def log_prediction(filename: str, predicted_class: str, confidence: float, latency_ms: float):
    with get_conn() as c:
        c.execute("INSERT INTO predictions (created_at, filename, predicted_class, confidence, "
                  "latency_ms) VALUES (?, ?, ?, ?, ?)",
                  (_now(), filename, predicted_class, confidence, latency_ms))


def prediction_stats() -> dict:
    with get_conn() as c:
        total = c.execute("SELECT COUNT(*), AVG(latency_ms) FROM predictions").fetchone()
        by_class = c.execute("SELECT predicted_class, COUNT(*) n FROM predictions "
                             "GROUP BY predicted_class").fetchall()
    return {"total": total[0], "avg_latency_ms": total[1],
            "by_class": {r["predicted_class"]: r["n"] for r in by_class}}


def fail_stale_runs():
    """Runs left 'running' by a crashed/restarted container can never finish."""
    with get_conn() as c:
        c.execute("UPDATE training_runs SET status='failed', finished_at=?, "
                  "message='Interrupted by a server restart' WHERE status='running'", (_now(),))
