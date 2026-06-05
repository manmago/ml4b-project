# Architecture Documentation — ML4B Gym Exercise Recognition
> Lightweight architecture overview. Full goal/scope rationale lives in
> [`business_understanding.md`](../business_understanding/business_understanding.md);
> every design decision is captured in [`docs/decisions/`](../decisions/) (ADRs).
> This document focuses on the parts that are *architectural*: context, building
> blocks, runtime pipelines, and the cross-cutting rules that hold them together.

---

## 1. Goals and Constraints

**Primary goal.** Classify gym exercises from **Apple Watch** sensor streams
(accelerometer + gyroscope) using supervised ML. The model distinguishes
**3 exercise classes** — `bicep_curl`, `tricep_extension`, `row` (ADR-016).
Three further outputs are produced **outside** the model: `rest` (energy gate,
ADR-017), `unknown` (novelty detector, ADR-024) and `uncertain` (confidence
threshold, ADR-020).

**Dataset anchor.** Training data is the Kaggle *Gym Workout IMU* dataset
(Apple Watch SE, left wrist, 100 Hz). Two earlier datasets (RecoFit, MM-Fit)
were dropped because of device-domain shift on real Apple-Watch uploads — see
ADR-013/016 for the full journey. Only **Wrist Motion** (accel + gyro) is used.

**Quality goals (architecturally relevant):**
| Priority | Quality Goal | Scenario |
|----------|-------------|----------|
| 1 | Reproducibility | Any team member runs the app with one command on a new machine (`uv run streamlit run …`) |
| 2 | Honest evaluation | Metrics are leakage-free (leave-one-set-out) and limitations are documented |
| 3 | Modularity / shared pipeline | Training and inference import the *same* preprocessing modules |
| 4 | Explainability | Predictions and confidence are shown per window in the app |

**Performance target:** macro F1 (leave-one-set-out CV) ≥ 0.80 — achieved 0.776
(single-subject ceiling, ADR-021).

**Key constraints:** Python 3.11 + `uv` (one-command run, ADR-022); Streamlit UI
(Python-native, no frontend work); scikit-learn (interpretable tabular models);
runs locally only (WSL/macOS/Windows), no cloud. No multi-subject Apple-Watch
data is collectable, which forces the single-subject anchor + augmentation
(ADR-019/021). Stakeholders and full scope: see `business_understanding.md`.

---

## 2. System Context

```
┌──────────────────────────────────────────────────────────────────┐
│                          External World                           │
│                                                                  │
│  Kaggle "Gym Workout IMU" dataset ──► 164 CSV files  [TRAIN]     │
│    Apple Watch SE, left wrist, 100 Hz, CoreMotion                │
│    Loaded via: src/ml4b/data/kaggle_loader.py (3 classes)        │
│                                                                  │
│  Apple Watch (Sensor Logger app) ──► WristMotion.csv or ZIP     │
│    Live app input (same CoreMotion format as the training data)  │
│    Loaded via: src/ml4b/data/apple_watch_loader.py               │
└──────────────────────┬───────────────────────────────────────────┘
                       │ data/raw/kaggle_gym_imu/   (training — ADR-016)
                       │ upload                       (inference)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                        ML4B System                                │
│  src/ml4b/   ──► shared pipeline package (training AND inference) │
│  scripts/    ──► train_model.py · inspect_kaggle_dataset.py       │
│  app/        ──► Streamlit prediction UI (3 pages)                │
│  models/     ──► best_model.joblib + model_metrics.json (committed)│
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
         Streamlit App (localhost:8501)
         Upload WristMotion.csv → per-window exercise + confidence
```

Both the Kaggle training data and Sensor Logger uploads are Apple CoreMotion
streams, so `src/ml4b/data/canonical.py` canonicalizes both identically (total
acceleration in g, gyro in rad/s). This device-domain match is the architectural
reason the Kaggle dataset was chosen (ADR-016).

---

## 3. Building Block View

```
src/ml4b/
├── data/
│   ├── canonical.py          # SHARED constants + CoreMotion canonicalization + 100 Hz resample
│   ├── kaggle_loader.py      # TRAIN: Kaggle CSV → long-format DataFrame (3 classes, ADR-016)
│   ├── apple_watch_loader.py # INFER: Sensor Logger CSV/ZIP → predict_from_sensor_logger()
│   ├── windowing.py          # Sliding window (200 @ 100 Hz, 50% overlap; carries recording_id)
│   ├── activity_gate.py      # Energy-threshold rest detection (ADR-017)
│   ├── novelty.py            # Open-set novelty detection → unknown (ADR-024)
│   ├── session.py            # Bout segmentation → per-set summary (ADR-025)
│   ├── features_invariant.py # 39 device-invariant features (ADR-018)
│   ├── augmentation.py       # rotation+time-warp+mirror+jitter (ADR-019)
│   ├── features.py           # LEGACY 47 per-axis features (abandoned)
│   ├── loader.py / mmfit_loader.py / splitting.py  # LEGACY (abandoned)
├── models/
│   ├── train.py              # train_random_forest() / xgboost / svm (ADR-009)
│   └── evaluate.py           # evaluate_model(), compare_models(), save_model()
├── feedback/                 # CONTINUAL LEARNING (ADR-027)
│   ├── store.py              # persist/load user label corrections (data/feedback/)
│   └── retrain.py            # rebuild model from base data + corrections (same pipeline)
└── utils/
    ├── config.py             # env-based path resolution (find_project_root, …)
    └── metrics.py            # load_model_metrics() — reads committed model_metrics.json
```

### Key Module Responsibilities (current pipeline)
| Module | Responsibility |
|--------|---------------|
| `canonical.py` | Single source of truth for pipeline constants (100 Hz, window 200, overlap 0.5, confidence 0.50) and CoreMotion canonicalization (total accel in g, gyro rad/s) + uniform resampling. Imported by both loaders so training and inference cannot drift. |
| `kaggle_loader.py` | Read the Kaggle Apple-Watch CSVs, keep the 3 target classes, drop sensor-lag, emit the long-format schema with one `recording_id` per set (ADR-016). |
| `apple_watch_loader.py` | Load Sensor Logger CSV/ZIP, auto-detect 5 column formats, run the full inference pipeline `predict_from_sensor_logger()` (resample → window → gate → invariant features → predict → confidence threshold). |
| `windowing.py` | Segment into 200-sample (2 s) windows, 50% overlap, never crossing a set; carries `recording_id` for leave-one-set-out grouping (ADR-006). |
| `activity_gate.py` | Energy-threshold rest detection (accel-std OR gyro-mean) so rest is not a learned class (ADR-017). |
| `features_invariant.py` | 39 orientation-/offset-robust features: magnitude stats + spectral, per-window z-normalized shape, axis-pair correlations, gyro/accel ratio (ADR-018). |
| `augmentation.py` | 6× synthetic variability (random rotation + time-warp + mirror + jitter) as a subject-diversity substitute (ADR-019). |
| `models/train.py` | Random Forest (`class_weight='balanced'`, seed 42) + XGBoost/SVM for comparison (ADR-009). |
| `models/evaluate.py` | Accuracy, macro F1, per-class F1, confusion matrix; saves plots. |
| `utils/metrics.py` | Load committed `model_metrics.json` for the app's Model Performance page. |
| `scripts/train_model.py` | End-to-end training with leave-one-set-out CV; writes model + metrics + feature names. |
| `feedback/store.py` | Persist the user's per-window label corrections (raw windows + label) to `data/feedback/feedback.jsonl`; read them back as a windowing-compatible frame (ADR-027). |
| `feedback/retrain.py` | Rebuild the model from base data + corrections through the *same* pipeline; backs up the shipped model; writes a manifest (ADR-027). |
| `scripts/update_model.py` | CLI for the retrain (and `--restore-base` to undo). |
| `app/pages/*.py` | `render()` for Home, Predict (incl. ✏️ Correct & Improve), Model Performance. |

---

## 4. Runtime View

### Training Pipeline (`scripts/train_model.py`)
```
data/raw/kaggle_gym_imu/*.csv   — Apple Watch, 100 Hz, 75 sets across 3 classes
    │
    ▼ kaggle_loader.load_kaggle_3class()
    │   canonicalize (total accel g, gyro rad/s), drop lag, recording_id per set
Long DataFrame: subject_id, exercise_name, recording_id, timestamp, ax..gz
    │
    ▼ windowing.apply_sliding_window(size=200, overlap=0.5)   — carries recording_id
    │
    ▼ augmentation.augment_windows(n_augment=5)   — rotation+time-warp+mirror+jitter (ADR-019)
    │
    ▼ features_invariant.extract_invariant_features()         — 39 features (ADR-018)
    │
    ▼ Leave-one-set-out CV (LeaveOneGroupOut on recording_id, ADR-021):
    │   train on other sets (+aug), test on held-out set's ORIGINAL windows only
    │   → aggregate macro F1 0.776 / acc 0.782, confusion matrix
    │
    ▼ Final model: RandomForest on ALL sets + augmentation (seed 42, balanced)
    │
    ▼ models/saved/best_model.joblib (compressed) + model_metrics.json
      data/processed/feature_names.txt
```

### Prediction Flow (Streamlit app)
```
User uploads WristMotion.csv OR a Sensor Logger ZIP
    │   (app/pages/prediction.py saves to a temp file)
    ▼ apple_watch_loader.predict_from_sensor_logger()
    │   load + detect_and_normalize_columns()  → [timestamp, ax..gz] (total accel g)
    │   canonical.resample_uniform(100 Hz)     — exact-rate windows
    │   windowing.apply_sliding_window(200, 0.5)        (SAME as training)
    │   activity_gate.gate_window_df()         — low-energy windows → rest
    │   features_invariant.extract_invariant_features() (SAME as training)
    │   novelty.NoveltyDetector.is_known()    — out-of-distribution → unknown (ADR-024)
    │   model.predict_proba() on KNOWN ACTIVE windows
    │   confidence < 0.50 → "uncertain" (ADR-020)
Per-window: predicted_class ∈ {3 exercises, rest, unknown, uncertain}, confidence, time
    │
    ▼ session.summarize_session()             — fold windows into per-set bouts (ADR-025)
    ▼ app/pages/prediction.py — detected rate, timeline, detected sets, pie, table, CSV
```

> Model + metrics are loaded once via `st.cache_resource` / `load_model_metrics()`
> and are committed to git, so the app runs after a fresh clone with no dataset.

---

## 5. Cross-Cutting Concepts

### Shared training/inference pipeline (the core rule)
Training (`kaggle_loader`) and inference (`apple_watch_loader`) both import
`canonical`, `windowing`, `activity_gate`, and `features_invariant`. Logic is
never duplicated, so predictions in the app match training exactly.

### Honest evaluation & limitations
- **Leave-one-set-out CV** (ADR-021): no same-set or augmented windows leak into
  the test fold.
- **Single-subject anchor:** cross-*person* performance cannot be measured and
  will be below the reported macro F1; augmentation (ADR-019) is the documented
  mitigation. This is stated in the README, project overview, and the app.

### Continual learning (human-in-the-loop)
The app captures the user's per-window label corrections (`feedback/store.py`)
and can rebuild the model from the base data **plus** those corrections through
the *same* pipeline (`feedback/retrain.py`) — the direct attack on the
single-subject limitation. Capture is decoupled from training (corrections are
always saved); retraining is explicit and reuses windowing/augmentation/features
so the model contract is unchanged. New labels become new classes (ADR-027).

### Reproducibility
- `uv.lock` pins every dependency; `uv run` provisions and runs in one step
  (ADR-022). Random seeds fixed at 42 everywhere.

### Path Handling
- All paths are `pathlib.Path` resolved via `src/ml4b/utils/config.py`
  (`find_project_root()`); no hardcoded absolute paths.

### Code Quality & Testing
- `ruff format` + `ruff check` (88-char lines); Google-style docstrings + type
  hints on all public functions.
- `pytest` with `tmp_path` fixtures; tests cover invariant features (incl.
  rotation invariance), the activity gate, augmentation, and the loader.
