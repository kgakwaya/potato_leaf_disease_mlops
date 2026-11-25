"""Flood-test the prediction API with Locust.

Interactive:
    locust -f locustfile.py --host http://localhost:8080
Headless (used for the README table), e.g. 100 users for 60 s:
    locust -f locustfile.py --host http://localhost:8080 --headless \
           -u 100 -r 10 -t 60s --csv locust_results/c3_u100 --html locust_results/c3_u100.html

Test images are taken from data/test/ (or SAMPLE_DIR); if none exist a
synthetic leaf image is generated so the test can always run.
"""
import io
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", Path(__file__).parent / "data" / "test"))
EXTS = {".jpg", ".jpeg", ".png"}


def _load_samples(limit: int = 50):
    files = [p for p in SAMPLE_DIR.rglob("*") if p.suffix.lower() in EXTS]
    random.shuffle(files)
    samples = [(p.name, p.read_bytes()) for p in files[:limit]]
    if not samples:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (256, 256), (70, 150, 60))
        ImageDraw.Draw(img).ellipse([90, 90, 140, 140], fill=(110, 80, 40))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        samples = [("synthetic.jpg", buf.getvalue())]
    return samples


SAMPLES = _load_samples()


class LeafUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(10)
    def predict(self):
        name, data = random.choice(SAMPLES)
        self.client.post("/predict?log=false", files={"file": (name, data, "image/jpeg")},
                         name="/predict")

    @task(2)
    def status(self):
        self.client.get("/api/status", name="/api/status")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
