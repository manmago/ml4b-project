# RecoFit Dataset

> This folder is in `.gitignore` — data files are never committed to git.

## Download

Source: https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors

Download the following files and place them in this directory (`data/raw/recofit/`):

## Files Needed

| File | Size | Description |
|------|------|-------------|
| `exercise_data.50.0000_singleonly.mat` | ~2.5 GB | Single-activity traces — use this for training |
| `exercise_data.50.0000_multionly.mat` | varies | Multi-activity traces |

## Citation

Morris, D., Saponas, T. S., Guillory, A., & Kelner, I. (2014, April).
RecoFit: using a wearable sensor to find, recognize, and count repetitive exercises.
In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems* (pp. 3225–3234). ACM.

## Format

MATLAB `.mat` file. Load with Python:

```python
import scipy.io
data = scipy.io.loadmat("exercise_data.50.0000_singleonly.mat", simplify_cells=True)
```

Key fields:
- `data['subject_data']` — cell matrix (n_subjects × n_exercises), each cell contains recording(s)
- `data['exerciseConstants']['activities']` — list of exercise class name strings
- Each recording: `recording['data']['accelDataMatrix']` and `recording['data']['gyroDataMatrix']`
  - Column 0: timestamp (seconds)
  - Columns 1–3: X, Y, Z axes

## Sensor Specs

| Property | Value |
|----------|-------|
| Sampling rate | 50 Hz |
| Accelerometer unit | g (gravitational acceleration) |
| Gyroscope unit | dps (degrees per second) |
| Placement | Wrist-worn |
| Participants | 200+ |

## Contact

Ask Anshul Agrawal for access to the downloaded files, or download directly from the GitHub link above.
