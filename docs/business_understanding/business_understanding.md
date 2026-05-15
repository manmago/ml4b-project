# Business Understanding — ML4B Gym Exercise Recognition

> CRISP-DM Phase 1 | Status: **Done** | Completed: 2026-05-15

---

## 1. Project Background

### University Context

This project is part of the **ML4B (Machine Learning for Business)** course at **FAU Nürnberg, Lehrstuhl für Wirtschaftsinformatik**, Summer Semester 2026. The project follows the **CRISP-DM methodology** (Cross-Industry Standard Process for Data Mining) and is structured around its six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

### Domain: Sports & Fitness Technology

Wearable fitness devices — such as the Apple Watch — are equipped with inertial measurement units (IMUs) that continuously capture motion data via accelerometers and gyroscopes. These sensors record movement in three spatial axes (x, y, z), producing rich time-series data that encodes the kinematic signature of physical activity.

### Motivation

Automatically recognising which exercise a person is performing from raw sensor data has significant practical value:

- **Fitness tracking:** Automatic workout logging without manual input from the user
- **Coaching & form feedback:** Detecting exercise type is a prerequisite for further analysis of movement quality and repetition counting
- **Health monitoring:** Linking exercise type to physiological outcomes (heart rate, caloric expenditure) requires reliable activity recognition
- **Clinical rehabilitation:** Automated exercise detection supports patient compliance monitoring in physical therapy settings

Despite the availability of consumer-grade wearable devices, robust and generalizable exercise classification remains an open challenge — particularly when models trained on one dataset are applied to data from a different individual or device.

---

## 2. Research Question

### Primary Research Question

> **"Can machine learning models trained on publicly available wrist-worn sensor data accurately classify gym exercises, and how well do these models generalize to new data collected from an Apple Watch during real workout sessions?"**

This question deliberately separates two concerns:

1. **In-distribution performance:** How well does the trained model classify exercises when evaluated on a held-out portion of the same public dataset?
2. **Out-of-distribution generalization:** How well does the model transfer to data from a different individual, recorded with a different device (Apple Watch via Sensor Logger) in a real-world gym setting?

### Secondary Research Questions

1. **Feature importance:** Which sensor-derived features (e.g., mean acceleration magnitude, spectral energy, cross-axis correlations) are most informative for distinguishing between the 7 target exercise classes?
2. **Algorithm comparison:** Which ML algorithm (e.g., Random Forest, SVM, k-NN, Gradient Boosting) achieves the best classification performance on this type of high-frequency, multi-axis time-series data?
3. **Generalization gap:** By how much does model accuracy degrade when evaluated on self-recorded Apple Watch data compared to the public test set? What factors explain this gap (subject variability, device differences, recording conditions)?

---

## 3. Business Goals

### Classification Target

The model must classify sensor data windows into exactly **7 classes**:

| # | Class | Notes |
|---|-------|-------|
| 1 | Bicep Curl | Dumbbell, standing |
| 2 | Shoulder Press | Dumbbell or barbell, seated or standing |
| 3 | Lateral Raise | Dumbbell, standing |
| 4 | Squat | Bodyweight or barbell |
| 5 | Bench Press | Barbell or dumbbell, lying |
| 6 | Deadlift | Barbell, standing |
| 7 | Rest / No Exercise | Idle periods between sets |

The "Rest" class is critical — without it, a classifier would always predict an exercise even during idle periods, making it unusable in practice.

### Performance Target

- **Minimum classification accuracy:** ≥ 80% on the held-out public validation set (macro-averaged, to account for class imbalance)
- **Generalization target:** ≥ 65% accuracy on self-recorded Apple Watch data

### Deployment Goal

The trained model will be wrapped in a **Streamlit web application** (`app/main.py`) that:

- Accepts sensor data input (CSV file upload of a recording window)
- Displays the **predicted exercise class** with its label
- Shows the **model confidence** (predicted class probability)
- Shows overall **model performance metrics** (accuracy, per-class F1, confusion matrix)

The Streamlit app serves as the project's demonstration artifact and final deliverable.

---

## 4. Data Strategy

### Two-Dataset Validation Strategy

The project uses a deliberate two-phase data strategy to rigorously test model generalizability:

#### Phase A — Training & Initial Validation: Public Wrist-Sensor Datasets

| Aspect | Detail |
|--------|--------|
| Purpose | Train and validate the full ML pipeline |
| Source type | Publicly available datasets from academic repositories and Kaggle |
| Sensor placement | **Wrist/smartwatch** (priority) — hip/pocket placement datasets are avoided or used for secondary comparison only |
| Sensor modalities | Accelerometer (ax, ay, az) + Gyroscope (gx, gy, gz) — must be available |
| Split | Standard train/validation/test split within the public dataset |

**Candidate datasets (to be evaluated in Phase 2 — Data Understanding):**
- **MMFIT Dataset** — Multi-modal fitness exercise dataset with wrist sensor data
- **Exercise Recognition datasets on Kaggle** — Various community-contributed wrist IMU datasets
- **WHARF Dataset** — Wrist Hand Action Recognition Framework dataset

Datasets that rely on hip, pocket, or chest placement will be excluded or treated as secondary comparisons, as they produce fundamentally different motion signatures than wrist-worn sensors.

#### Phase B — Generalization Test: Self-Recorded Apple Watch Data

| Aspect | Detail |
|--------|--------|
| Purpose | Test how well the model generalizes to a new individual and device |
| Device | Apple Watch (Series 6 or later) |
| Recording app | **Sensor Logger** (free iOS app, App Store) |
| Used channels | Wrist Motion (Accelerometer + Gyroscope), optionally Heart Rate |
| Discarded channels | Location, Barometer, Magnetometer, Compass — irrelevant for exercise classification |

**Rationale for this strategy:**

Training on public data first allows building and validating the full pipeline before collecting personal data — a more efficient use of development time. Testing on personal Apple Watch data then provides a scientifically stronger generalization evaluation than a simple train/test split on the same dataset, because it introduces genuine subject-level and device-level distribution shift.

### Sampling Rate Alignment

Apple Watch via Sensor Logger delivers Wrist Motion data at approximately **50–100 Hz**. Public datasets must be resampled to a consistent sampling rate during Phase 3 (Data Preparation) to ensure the same sliding-window features are computed comparably. The target sampling rate will be determined in Phase 2 after inspecting the selected public dataset.

---

## 5. Success Criteria

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Classification Accuracy | ≥ 80% | Macro-averaged accuracy on held-out public test set |
| Generalization | ≥ 65% | Macro-averaged accuracy on self-recorded Apple Watch data |
| App Usability | Functional demo | Streamlit app runs without errors, predictions render correctly |
| Reproducibility | Full pipeline reproducible | `uv sync` + single command restores environment and runs pipeline |
| Documentation | Complete | arc42 architecture doc + CRISP-DM log up to date at project end |

---

## 6. Constraints & Assumptions

### Data Constraints

- **Limited personal test data:** Only one team member records personal Apple Watch data → the personal test set will be small (a single gym session per exercise class). This limits statistical power but is sufficient for a qualitative generalization assessment.
- **Sampling rate mismatch:** Apple Watch via Sensor Logger may deliver wrist motion at a different sampling rate than the selected public dataset. Resampling will be required during preprocessing (Phase 3).
- **Sensor channel availability:** Only Wrist Motion (Accelerometer + Gyroscope) and optionally Heart Rate will be used from Sensor Logger. Location, Barometer, Magnetometer, and Compass channels will be discarded, as they do not carry exercise-discriminative information.

### Project Constraints

- **Deadline:** End of lecture period, SoSe 2026
- **Tech stack is fixed:** Python 3.11, scikit-learn, Streamlit, uv — no alternatives
- **All 7 exercise classes must be performable:** The team member recording personal data must be able to safely perform and record all 7 exercises (including Deadlift, Squat, Bench Press) in a real gym session

### Assumptions

- Public wrist-sensor datasets exist with sufficient labeled data for all 7 (or a subset of) the target exercise classes
- The Sensor Logger app records IMU data at a consistent and documentable sampling rate
- scikit-learn with classical ML algorithms (Random Forest, SVM) is sufficient — deep learning approaches (CNNs on raw signal) are out of scope for this project

---

## 7. Related Work

> **Placeholder — to be completed as a CRISP-DM extension phase.**

A dedicated Related Work section will be added as a separate documentation entry (`docs/business_understanding/related_work.md`), covering:

- Existing academic papers on gym exercise recognition from wearable IMU sensors
- Benchmark results and commonly used datasets in the exercise recognition literature
- Comparison of classical ML vs. deep learning approaches for sensor-based activity recognition
- Relevant surveys on Human Activity Recognition (HAR) from wrist-worn devices

This section will be populated during or after Phase 2 (Data Understanding), once the relevant literature has been surveyed to inform dataset selection and feature engineering decisions.
