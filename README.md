# 🥔 Potato Leaf Doctor: Irish Potato Leaf Disease Detection (MLOps)

I built this end-to-end machine learning project to classify a photo of an Irish potato leaf as
**Early blight**, **Late blight** or **Healthy**. I trained a CNN, served it through a FastAPI API
and web dashboard, packaged the app with Docker, deployed it on Render, and tested its API with Locust.

| | |
|---|---|
| 🧑‍💻 **GitHub repository** | https://github.com/kgakwaya/potato_leaf_disease_mlops |
| 🎥 **Video demo (YouTube)** | [Watch my project demo](https://www.youtube.com/watch?v=KzKfwLNZ57Q) |
| 🌐 **Live app** | https://potato-leaf-disease-mlops.onrender.com |
| 📖 **API docs** | https://potato-leaf-disease-mlops.onrender.com/docs |
| 📓 **Notebook** | https://colab.research.google.com/drive/1uEENL-XzzKNE8Lqkom3KFhI2BL2pl4-V |
| 🧠 **Model file** | [`models/potato_cnn.h5`](models/potato_cnn.h5) |

[![Watch my project demo](https://img.youtube.com/vi/KzKfwLNZ57Q/0.jpg)](https://www.youtube.com/watch?v=KzKfwLNZ57Q)

> **Note:** the live app runs on Render's free plan. It goes to sleep when nobody uses it, so the
> first visit can take about a minute to load. After that it is fast.

---

## 1. Project description

Late blight and early blight are the two most damaging leaf diseases of Irish potato, a staple
crop in Rwanda and the region. Late blight can destroy a whole field within days in cool, wet
weather, so catching it early matters. With this app a farmer or extension officer can upload a
leaf photo and get an instant diagnosis. New labelled photos can also be uploaded so the model
keeps improving over time.

**The model** is a custom convolutional neural network (CNN) **trained from scratch**, with no
pretrained backbone. When the model is retrained, the saved custom model (`models/potato_cnn.h5`)
is loaded as the pre-trained starting point and fine-tuned on the old data plus the new uploads.

### Dataset

[PlantVillage on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
(`arjuntejaswi/plant-village`). The download has 15 class folders for pepper, potato and tomato.
This project uses **only the 3 potato classes**:

| Folder | Class | Images (approx.) |
|---|---|---|
| `Potato___Early_blight` | Early blight | 1,000 |
| `Potato___Late_blight` | Late blight | 1,000 |
| `Potato___healthy` | Healthy | 152 |

The notebook keeps only the folders that start with `Potato` (`CLASS_PREFIX = "Potato"`), so the
full Kaggle zip can be uploaded as it is. Duplicate images are removed. The healthy class is much
smaller than the others, so training uses class weights and the results report macro F1.

### Pipeline

```
 Dataset zip (uploaded in Colab) ─► Google Drive
        │
        ▼
 Data acquisition ─► Preprocessing ─► Custom CNN ─► Evaluation ─► models/potato_cnn.h5
 (verify, split)     (RGB, 128×128,    (L2, dropout,   (accuracy, loss,
                      /255, augment,    batch norm,     precision, recall,
                      class weights)    early stopping) F1, ROC-AUC,
                                                        confusion matrix)
        │
        ▼
 FastAPI service + web UI  ──►  /predict      one image → class + confidence
 (Docker, deployed on      ──►  /upload       many images / zip → preprocessed → SQLite DB
  Render)                  ──►  /retrain      fine-tune the saved model → compare → promote
                           ──►  /api/status   uptime, model version, metrics
                           ──►  /insights     data visualisations
```

### Features

* **Prediction:** upload one leaf image and get the predicted class, the confidence and the
  probability for each class.
* **Visualisations:** class distribution, average colour per class, green-leaf pixel ratio and
  edge density (lesion texture), each with a short explanation, plus model metrics and the
  confusion matrix.
* **Bulk upload:** upload many images at once, or a `.zip` with one folder per class. Every image
  is checked, converted to RGB, resized and saved in `data/uploads/<class>/`. Each file is
  recorded in the SQLite database (`uploads` table).
* **Retraining:** a button (or `POST /retrain`) fine-tunes the saved model on old + new data,
  compares the old and new models on the test set and keeps the new one only if it is not worse.
  Every run is logged in the `training_runs` table and shown on the dashboard.
* **Monitoring:** uptime, model version, number of requests, average response time and run history.

---

## 2. Repository structure

```
potato_leaf_disease_mlops/
├── README.md
├── notebook/
│   └── potato_leaf_disease.ipynb   # Colab: data → preprocessing → training → evaluation → retraining
├── src/
│   ├── preprocessing.py            # image checks, resize, scaling, augmentation, feature extraction
│   ├── model.py                    # custom CNN, optimisers, callbacks, evaluation metrics, saving
│   ├── prediction.py               # Predictor class (reloads the model after retraining)
│   ├── retrain.py                  # retraining pipeline (saved model used as the pre-trained model)
│   ├── train.py                    # train from scratch from the command line
│   ├── database.py                 # SQLite: uploads, training runs, prediction log
│   └── config.py                   # paths and settings
├── api/main.py                     # FastAPI app (predict, upload, retrain, status, insights)
├── ui/index.html                   # web dashboard (served at /)
├── data/
│   ├── train/<class>/*.jpg
│   └── test/<class>/*.jpg
├── models/
│   ├── potato_cnn.h5               # trained model
│   └── class_names.json, metrics.json, insights.json, plots/
├── Dockerfile, docker-compose.yml, nginx/nginx.conf
├── locustfile.py
└── locust_results/                 # Locust screenshots and reports
```

---

## 3. Model results

These are the saved evaluation results in [`models/metrics.json`](models/metrics.json). The
optimised model was evaluated on 323 held-out test images.

| Model | Accuracy | Loss | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|---|
| Vanilla CNN (baseline) | 0.4830 | 1.8776 | 0.6073 | 0.4080 | 0.3110 | 0.9122 |
| **Optimised custom CNN** | **0.9876** | **0.2936** | **0.9593** | **0.9911** | **0.9740** | **0.9998** |

I used L2 regularisation, dropout, batch normalisation, data augmentation, label smoothing, class
weights, optimiser comparison (Adam / RMSprop / SGD with Nesterov), early stopping,
ReduceLROnPlateau and model checkpointing. These results are from the saved model evaluation; they
are not a guarantee of performance on new field photos.

## 4. Notebook: preprocessing, training and prediction

I documented the training workflow in [`notebook/potato_leaf_disease.ipynb`](notebook/potato_leaf_disease.ipynb)
(also [open it in Google Colab](https://colab.research.google.com/drive/1uEENL-XzzKNE8Lqkom3KFhI2BL2pl4-V)).
The notebook covers these steps:

1. **Get and inspect the data:** I load the PlantVillage dataset from a zip or Google Drive. The
   notebook keeps the `Potato` folders, removes duplicate images, and prepares training,
   validation and test splits.
2. **Preprocess the images:** I verify each image, convert it to RGB, resize it to 128 × 128,
   and scale pixel values from 0–255 to 0–1. For training, I also apply random flips, rotations,
   zoom, contrast and brightness changes. I use class weights to account for the smaller healthy
   class.
3. **Train the model:** I build and train a custom CNN, compare optimisers, and use regularisation,
   early stopping, learning-rate scheduling and checkpointing to guide training.
4. **Test and evaluate:** I report accuracy, loss, macro precision, recall and F1, ROC-AUC, and a
   confusion matrix on held-out data.
5. **Predict:** the notebook includes functions that preprocess a leaf image, run the model, and
   return the predicted class and class probabilities.
6. **Save and retrain:** I save the model and class names for the API, then demonstrate fine-tuning
   the saved model with additional labelled data.

The trained Keras model is included at [`models/potato_cnn.h5`](models/potato_cnn.h5) (about
14.4 MB). Supporting class names and evaluation results are in `models/class_names.json` and
`models/metrics.json`.

---

## 5. Setup

### Step 1: Get the project

```bash
git clone https://github.com/kgakwaya/potato_leaf_disease_mlops.git
cd potato_leaf_disease_mlops
```

### Step 2: (Optional) Train the model in Google Colab

1. Open `notebook/potato_leaf_disease.ipynb` in Colab (*File → Upload notebook*, or use the Colab link above).
2. Go to *Runtime → Change runtime type → T4 GPU*.
3. Choose how to give the dataset in the configuration cell:
   * `UPLOAD_METHOD = "drive_zip"` (recommended): put the dataset zip in Google Drive and set
     `DATASET_ZIP_ON_DRIVE` to its path, e.g. `/content/drive/MyDrive/potato_dataset.zip`.
   * `UPLOAD_METHOD = "upload_zip"`: a file picker appears and you upload the zip from your computer.
   * `UPLOAD_METHOD = "drive_folder"`: the dataset is already unzipped on Drive.
4. Click *Runtime → Run all* and allow Drive access. Everything is saved under
   `MyDrive/potato_leaf_mlops/` (`data/`, `models/`, `models/plots/`, `metrics.json`).
5. The last cell creates `MyDrive/potato_leaf_mlops/export_for_repo.zip`. Download it and unzip it
   into the root of this repository. This fills `models/`, `data/train` and `data/test`.
6. The notebook records TensorFlow 2.21.0, while this API's `requirements.txt` currently pins
   TensorFlow 2.20.0. If I export a model trained with a different TensorFlow version, I align the
   serving dependency with the training version before deploying it.

### Step 3: Run locally with Docker (recommended)

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) running.

```bash
docker compose up --build
```

Then open:

* Dashboard: http://localhost:8080
* API docs: http://localhost:8080/docs

To stop it, press `Ctrl + C`, then run `docker compose down`.

**Optional: more containers.** To run 3 API containers behind nginx (to test scaling):

```bash
docker compose up --build -d --scale api=3
docker compose restart nginx
```

All containers share the `./data` and `./models` folders, so an upload or a retrained model is
seen by every container.

### Step 4 (optional): Run locally without Docker

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000 and http://localhost:8000/docs.

Command-line tools:

```bash
python -m src.train --epochs 40          # train from scratch on data/train
python -m src.retrain --epochs 5         # retrain on uploaded data (same as the button)
python -m src.prediction path/to/leaf.jpg
```

### Step 5: Deploy to the cloud (Render)

The app is deployed on [Render](https://render.com) as a Docker web service:
**https://potato-leaf-disease-mlops.onrender.com**

How to deploy it yourself:

1. Push this repository to GitHub. Make sure `models/potato_cnn.h5` and the other files in
   `models/` are committed, because the app needs them to start.
2. Sign in to Render with GitHub.
3. Click *New → Web Service* and choose this GitHub repository.
4. Set **Language / Runtime** to **Docker**. Render finds the `Dockerfile` by itself.
5. Choose the **Free** instance type and click *Create Web Service*.
6. Wait for the build to finish (about 5 to 10 minutes). Render then shows the public URL,
   e.g. `https://potato-leaf-disease-mlops.onrender.com`.

Every new push to GitHub starts a new deploy automatically.

**Important: memory limit.** Render's free plan has only **512 MB of RAM**. Prediction works fine,
but retraining needs more memory (about 830 MB in testing), so it can be stopped on the free plan.
To retrain, either run the app on localhost with Docker (Step 2), or choose a bigger Render
instance with at least 1 GB of RAM.

**Why not Hugging Face?** Hugging Face Spaces was the first plan, but Docker Spaces required a
paid plan on this account, so Render was used instead.

**Evaluation in production.** After a retraining run, the dashboard shows the accuracy before and
after on the test set, the new metrics and the confusion matrix. `GET /metrics` returns the full
metrics of the deployed model.

---

## 6. Using the app

| Tab | What to do |
|---|---|
| **Dashboard** | See uptime, model version, test metrics, confusion matrix and retraining history |
| **Data insights** | 4 visualisations with explanations. Click *Recompute* after new uploads |
| **Predict** | Drop one leaf image → *Predict* → class, confidence, probabilities |
| **Upload & retrain** | Pick a label (or upload a zip with class folders) → *Upload* → *Retrain model* → watch progress |

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | check that the app is running |
| GET | `/api/status` | uptime, model info, latest metrics, request counters |
| POST | `/predict` | `file`: one image → prediction |
| POST | `/upload` | `files`: many images or zip(s); `label` (optional for zips with class folders) |
| GET | `/uploads` | number of uploads per class (pending / used) |
| POST | `/retrain?epochs=5` | start retraining in the background |
| GET | `/retrain/status` | run history and progress |
| GET | `/metrics` | full evaluation metrics of the deployed model |
| GET | `/insights?refresh=true` | dataset statistics used for the charts |

Examples (local Docker):

```bash
curl -F "file=@leaf.jpg" http://localhost:8080/predict
curl -F "files=@new_images.zip" http://localhost:8080/upload
curl -F "files=@a.jpg" -F "files=@b.jpg" -F "label=Late blight" http://localhost:8080/upload
curl -X POST "http://localhost:8080/retrain?epochs=5"
```

For the live app, replace `http://localhost:8080` with `https://potato-leaf-disease-mlops.onrender.com`.

### What happens when you press "Retrain"

1. **Upload and save to the database:** files are saved in `data/uploads/<class>/` and recorded in SQLite.
2. **Preprocessing:** each image is checked, converted to RGB and resized to 128×128 at upload
   time. During retraining the images are scaled to [0, 1] and augmented, the same as in the notebook.
3. **Retraining from the saved model:** `models/potato_cnn.h5` is loaded and fine-tuned
   (Adam, learning rate 1e-4, early stopping, class weights) on `data/train` + the uploads.
4. **Evaluation and promotion:** the old and new models are both tested on `data/test`. The new
   model replaces the old one only if its accuracy is not lower. The old model is kept in
   `models/archive/`.

---

## 7. Flood request simulation (Locust)

I use Locust to send concurrent requests and observe how the API responds. Each simulated user
mostly sends `/predict` requests with test images, along with occasional status and health checks.

```bash
pip install locust
```

**Test on localhost (Docker running):**

```bash
python -m locust -f locustfile.py --host http://localhost:8080
```

**Test on Render:**

```bash
python -m locust -f locustfile.py --host https://potato-leaf-disease-mlops.onrender.com
```

Do not put a `/` at the end of the host URL.

Then I open http://localhost:8089, set the number of users and spawn rate, and start the run.
The screenshots below show my run against the deployed Render app. I saved the original Locust
CSV export in `locust_results/` as well.

### Results from my recorded run

| Endpoint | Requests | Failures | Average response time | p95 response time | Current RPS shown |
|---|---:|---:|---:|---:|---:|
| `POST /predict` | 9 | 0 (0%) | 1,847.22 ms | 2,600 ms | 0.57 |
| **All endpoints** | **17** | **0 (0%)** | **1,678.67 ms** | **3,000 ms** | **1.29** |

This short recorded run completed without failed requests. It includes only 17 requests, so I treat
it as a demonstration of the test setup, not as a large-scale capacity benchmark. The Render app
may also take longer to answer its first requests after going idle.



## 8. Tech stack

TensorFlow/Keras · scikit-learn · FastAPI · SQLite · Chart.js · Docker + nginx · Render · Locust · Google Colab
