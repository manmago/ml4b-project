"""Data subpackage for ML4B gym exercise recognition.

Handles all raw data access: loading RecoFit .mat files via scipy.io.loadmat,
schema validation, and train/test splitting. No transformation or feature
engineering belongs here — that lives in src/ml4b/models/.
"""
