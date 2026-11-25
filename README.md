# 🥔 Potato Leaf Doctor: Irish Potato Leaf Disease Detection (MLOps)

An end-to-end machine learning pipeline that classifies photos of Irish potato leaves as
**Early blight**, **Late blight** or **Healthy**, served through a Python API and a web dashboard,
containerised with Docker, deployed to the cloud and load-tested with Locust.

| | |
|---|---|
| 🎥 **Video demo** | `<ADD YOUTUBE LINK>` |
| 🌐 **Live app** | `<ADD DEPLOYED URL>` (API docs at `<URL>/docs`) |
| 📓 **Notebook** | [`notebook/potato_leaf_disease.ipynb`](notebook/potato_leaf_disease.ipynb) |
| 🧠 **Model file** | [`models/potato_cnn.h5`](models/) |

---

## 1. Project description

Late blight and early blight are the two most damaging leaf diseases of Irish potato, a staple
crop in Rwanda and the region. Late blight can destroy a field within days in cool, wet weather,
so early detection matters. This project lets a farmer or extension officer upload a leaf photo
and get an instant diagnosis, and lets the team keep improving the model with newly collected,
labelled photos.

**The model** is a custom convolutional neural network **trained from scratch** (no pretrained
backbone). For retraining, the saved custom model is loaded as the pre-trained starting point
and fine-tuned on the old data plus the newly uploaded images.

### Dataset

[PlantVillage on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) (`arjuntejaswi/plant-village`).
The download has 15 class folders for **pepper, potato and tomato**. This project uses **only the
3 potato classes**:

| Folder | Class | Images (approx.) |
|---|---|---|
| `Potato___Early_blight` | Early blight | 1,000 |
| `Potato___Late_blight` | Late blight | 1,000 |
| `Potato___healthy` | Healthy | 152 |

The notebook filters the folders automatically (`CLASS_PREFIX = "Potato"`), so you can upload
the full Kaggle zip as it is. Pepper and tomato folders are ignored and duplicate images are
removed. The healthy class is small, which is why training uses class weights and reports macro F1.

### Pipeline

```
 Dataset (your zip, uploaded in Colab) ─► Google Drive
        │
        ▼
 Data acquisition ─► Preprocessing ─► Custom CNN ─► Evaluation ─► models/potato_cnn.h5
 (verify, split)     (RGB, 128×128,    (L2, dropout,   (accuracy, loss,
                      /255, augment,    BN, early       precision, recall,
                      class weights)    stopping, opt.  F1, ROC-AUC,
                                        search)         confusion matrix)
        │
        ▼
 FastAPI service + web UI  ──►  /predict          one image → class + confidence
 (Docker, N containers     ──►  /upload           bulk images / zip → preprocessed → SQLite DB
  behind nginx)            ──►  /retrain          button → fine-tune saved model → compare → promote
                           ──►  /api/status       uptime, model version, metrics
                           ──►  /insights         feature visualisations
```

### Features

* **Prediction**: upload one leaf image and get the predicted class, confidence and class probabilities.
* **Visualisations**: class distribution, average colour per class, green-leaf pixel ratio and
  edge density (lesion texture), each with a written interpretation, plus model metrics and the
  confusion matrix.
* **Bulk upload**: many images at once, or a `.zip` with one folder per class. Every image is
  verified, converted to RGB, resized and stored in `data/uploads/<class>/`. Each file is recorded in the
  SQLite database (`uploads` table).
* **Retraining trigger**: a button (or `POST /retrain`) fine-tunes the saved model on old + new
  data, evaluates old vs new on the test set and promotes the new model only if it is not worse.
  Every run is logged in the `training_runs` table and shown in the dashboard.
* **Monitoring**: uptime, model version, requests served, average latency, run history.

---

## 2. Repository structure

```
potato_leaf_disease_mlops/
├── README.md
├── notebook/
│   └── potato_leaf_disease.ipynb   # Colab: data upload → preprocessing → training → evaluation → retraining
├── src/
│   ├── preprocessing.py            # image validation, resize, scaling, augmentation, tf.data, feature extraction
│   ├── model.py                    # custom CNN, optimisers, callbacks, evaluation metrics, saving
│   ├── prediction.py               # Predictor class (hot-reloads after retraining)
│   ├── retrain.py                  # retraining pipeline (uses the saved model as pre-trained model)
│   ├── train.py                    # train from scratch from the command line (same as notebook)
│   ├── database.py                 # SQLite: uploads, training runs, prediction log
│   └── config.py                   # paths & settings (overridable with env vars)
├── api/main.py                     # FastAPI app (predict, upload, retrain, status, insights)
├── ui/index.html                   # web dashboard (served by the API at /)
├── data/
│   ├── train/<class>/*.jpg
│   └── test/<class>/*.jpg
├── models/
│   ├── potato_cnn.h5               # trained model
│   ├── class_names.json, metrics.json, insights.json, plots/
├── Dockerfile, docker-compose.yml, nginx/nginx.conf
├── locustfile.py, scripts/flood_test.sh, locust_results/
└── deploy/huggingface/README.md    # Space config for cloud deployment
```

---

## 3. Model results

Fill in from `models/metrics.json` (printed in section 7 of the notebook) after training.

| Model | Accuracy | Loss | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| Vanilla CNN (baseline) | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |
| **Optimised custom CNN** | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |

Optimisation used: L2 regularisation, dropout, batch normalisation, data augmentation, label
smoothing, class weights, optimiser comparison (Adam / RMSprop / SGD-Nesterov), early stopping,
ReduceLROnPlateau and checkpointing.

---

## 4. Setup

### Step 1: Train the model in Google Colab (your dataset, saved to your Drive)

1. Open `notebook/potato_leaf_disease.ipynb` in Colab (*File → Upload notebook*, or open it from GitHub).
2. *Runtime → Change runtime type → T4 GPU*.
3. Provide the dataset. Pick one option in the configuration cell:
   * `UPLOAD_METHOD = "drive_zip"` (recommended): upload your dataset zip to Google Drive and set
     `DATASET_ZIP_ON_DRIVE` to its path, e.g. `/content/drive/MyDrive/potato_dataset.zip`.
   * `UPLOAD_METHOD = "upload_zip"`: a file picker appears and you upload the zip from your computer.
   * `UPLOAD_METHOD = "drive_folder"`: the dataset is already extracted on Drive.

   The zip can contain `class/image.jpg` folders, or existing splits such as
   `Training/`, `Validation/`, `Testing/`. Both layouts are detected automatically.
   Only folders starting with `Potato` are kept, so the full PlantVillage zip works as it is. If an
   earlier run saved pepper/tomato classes to Drive, the notebook detects it and rebuilds the split.
4. *Runtime → Run all* and allow Drive access. Everything is saved under
   `MyDrive/potato_leaf_mlops/` (`data/`, `models/`, `models/plots/`, `metrics.json`).
5. The last cell writes `MyDrive/potato_leaf_mlops/export_for_repo.zip`. Download it and unzip it
   into the root of this repository. That fills `models/` and `data/train`, `data/test`.
6. The first cell prints the TensorFlow version. Make sure `requirements.txt` uses the same
   version (`tensorflow-cpu==<version>`) so the `.h5` file loads cleanly.

### Step 2: Run locally

```bash
git clone <GITHUB_REPO_URL>
cd potato_leaf_disease_mlops
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000 for the dashboard and http://localhost:8000/docs for Swagger.

Optional command-line tools:

```bash
python -m src.train --epochs 40          # retrain from scratch on data/train
python -m src.retrain --epochs 5         # retrain on uploaded data (same as the button)
python -m src.prediction path/to/leaf.jpg
```

### Step 3: Run with Docker (scalable)

```bash
docker compose up --build -d --scale api=3     # 3 API containers behind nginx
open http://localhost:8080
docker compose restart nginx                   # run after changing the number of containers
docker compose down
```

The API containers share `./data` and `./models` volumes, so an upload, the database or a
retrained model in one container is visible to all of them. Each container hot-reloads the model
when the file changes.

### Step 4: Deploy to the cloud (Hugging Face Spaces, free)

Hugging Face Spaces gives a free Docker host with 2 vCPU and 16 GB RAM. That is enough to run
retraining in production. Render's free tier has only 512 MB RAM: in testing, retraining
used about 830 MB, so it would be killed there.

1. Create an account at https://huggingface.co, then *New Space* → SDK **Docker** → blank → public.
2. Push the repository to the Space:
   ```bash
   git clone https://huggingface.co/spaces/<hf-username>/potato-leaf-doctor hf-space
   rsync -av --exclude .git --exclude README.md ./ hf-space/
   cp deploy/huggingface/README.md hf-space/README.md     # Space config (sdk: docker, port 8000)
   cd hf-space
   git lfs install && git lfs track "*.h5" "*.jpg" "*.png" "*.JPG"
   git add . && git commit -m "Deploy Potato Leaf Doctor" && git push
   ```
   (Use a Hugging Face *access token* with write permission as the password.)
3. Wait for the build (about 5 to 10 minutes). The app is then live at
   `https://<hf-username>-potato-leaf-doctor.hf.space`. Put this URL at the top of this README.

**Alternative: Render.** *New → Web Service → Build from GitHub repo → Docker*. Prediction works on
the free plan. For retraining, choose an instance with at least 1 GB RAM.

**Evaluation in production.** After a retraining run, the dashboard shows the accuracy before and
after on the test set, the new metrics and the confusion matrix. `GET /metrics` returns the full
metrics of the deployed model.

---

## 5. Using the app

| Tab | What to do |
|---|---|
| **Dashboard** | Uptime, model version, test metrics, confusion matrix, retraining history |
| **Data insights** | 4 visualisations with interpretations, *Recompute* after new uploads |
| **Predict** | Drop one leaf image → *Predict* → class, confidence, probabilities |
| **Upload & retrain** | Pick a label (or upload a zip with class folders) → *Upload* → *Retrain model* → watch progress |

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/api/status` | uptime, model info, latest metrics, request counters |
| POST | `/predict` | `file`: one image → prediction |
| POST | `/upload` | `files`: many images or zip(s); `label` (optional for zips with class folders) |
| GET | `/uploads` | upload counts per class (pending / used) |
| POST | `/retrain?epochs=5` | start retraining in the background |
| GET | `/retrain/status` | run history and progress |
| GET | `/metrics` | full evaluation metrics of the deployed model |
| GET | `/insights?refresh=true` | dataset feature statistics used for the charts |

Example:

```bash
curl -F "file=@leaf.jpg" http://localhost:8000/predict
curl -F "files=@new_images.zip" http://localhost:8000/upload
curl -F "files=@a.jpg" -F "files=@b.jpg" -F "label=Late blight" http://localhost:8000/upload
curl -X POST "http://localhost:8000/retrain?epochs=5"
```

### Retraining pipeline (what happens when you press the button)

1. **Upload + save to database**: files are stored in `data/uploads/<class>/` and recorded in SQLite.
2. **Preprocessing of the uploaded data**: verify, convert to RGB, resize to 128×128 at upload time.
   During retraining the data is scaled to [0, 1] and augmented, exactly like in the notebook.
3. **Retraining with the custom model as pre-trained model**: `models/potato_cnn.h5` is loaded
   and fine-tuned (Adam, learning rate 1e-4, early stopping, class weights) on `data/train` + uploads.
4. **Evaluation + promotion**: old and new models are both scored on `data/test`. The new model
   replaces the old one only if its accuracy is not lower. The previous model is archived in
   `models/archive/`.

---

## 6. Flood request simulation (Locust)

```bash
pip install locust
# Interactive UI at http://localhost:8089
locust -f locustfile.py --host http://localhost:8080
# Automated: 1, 2 and 3 containers, 100 users each, 60 s, prints a results table
USERS=100 DURATION=60s bash scripts/flood_test.sh
```

Each simulated user mainly sends `/predict` requests with real test images, plus a few
status calls. HTML reports are saved in `locust_results/`.

### Results

Paste the table printed by `scripts/flood_test.sh` (values for `/predict`):

| Containers | Users | Requests | Failures | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Req/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |
| 2 | 100 | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |
| 3 | 100 | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` | `<>` |

**Observations:** `<describe how latency and throughput change as containers are added, e.g.
average latency drops and requests/second rise because nginx spreads the load across replicas,
until CPU on the host becomes the bottleneck>`.

---

## 7. Tech stack

TensorFlow/Keras · scikit-learn · FastAPI · SQLite · Chart.js · Docker + nginx · Locust · Google Colab · Hugging Face Spaces
