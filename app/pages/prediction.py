"""Prediction page — Streamlit sub-page for exercise recognition from uploaded CSV.

This page will implement the full end-to-end prediction pipeline for Phase 6:
1. Accept a WristMotion.csv upload from the Sensor Logger app (Apple Watch)
2. Run the same preprocessing pipeline used during training:
   - load_sensor_logger_csv() from src/ml4b/data/apple_watch_loader.py
   - Sliding-window segmentation (100 samples, 50% overlap)
   - Feature extraction (47 features per window)
3. Load models/saved/best_model.joblib and run inference
4. Display predicted exercise per time window with confidence score
5. Show a timeline plot of the workout session

Phase 6 implementation target: replace the placeholder in streamlit_app.py with
the completed logic from this module.
"""
