# ML4B — Gym Exercise Recognition

**ML4B SoSe 2026 | FAU Nürnberg | Lehrstuhl für Wirtschaftsinformatik**

Erkennung von Gym-Übungen anhand von Sensordaten (z.B. Apple Watch) mittels Machine Learning.

---

## Projektstruktur

```
ml4b-project/
│
├── app/                          # Streamlit Web Application
│   ├── streamlit_app.py          # Hauptdatei der App (Einstiegspunkt)
│   └── pages/                    # Weitere Streamlit-Seiten (Multi-Page App)
│       ├── 01_data_exploration.py
│       ├── 02_model_performance.py
│       └── 03_live_prediction.py
│
├── data/                         # NICHT in git (siehe .gitignore)
│   ├── raw/                      # Rohdaten — niemals bearbeiten
│   ├── processed/                # Vorverarbeitete Daten
│   └── README.md                 # Beschreibung der Datenstruktur
│
├── notebooks/                    # Jupyter Notebooks (Exploration & Analyse)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                          # Wiederverwendbarer Python-Code
│   └── ml4b/                     # Package (importierbar als `from ml4b import ...`)
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py         # Datenladen (keine hardcodierten Pfade!)
│       │   └── preprocessing.py  # Feature Engineering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train.py          # Modell Training
│       │   └── evaluate.py       # Modell Evaluation
│       └── utils/
│           ├── __init__.py
│           └── config.py         # Konfiguration via Umgebungsvariablen
│
├── models/                       # Gespeicherte Modelle (NICHT in git)
│   └── saved/
│
├── tests/                        # Unit Tests
│   ├── fixtures/                 # Kleine Testdaten (in git OK)
│   └── test_preprocessing.py
│
├── .vscode/
│   ├── settings.json             # Workspace-Einstellungen (in git)
│   └── extensions.json           # Empfohlene Extensions (in git)
│
├── .gitignore
├── .python-version               # Legt Python 3.11 für uv fest
├── pyproject.toml                # Projektdefinition & Dependencies
├── uv.lock                       # Lock-File (in git — für Reproduzierbarkeit!)
└── README.md
```

---

## Schnellstart

```bash
# 1. Repo clonen
git clone git@github.com:<ORG>/ml4b-project.git
cd ml4b-project

# 2. Umgebung einrichten (ein Befehl!)
uv sync

# 3. Streamlit App starten
uv run streamlit run app/streamlit_app.py
```

→ Detaillierte OS-spezifische Anleitungen: `Setup_WSL_Windows.md`, `Setup_macOS.md`, `Setup_Windows.md`

---

## Team

| Name | OS | GitHub |
|------|----|--------|
| Anshul | WSL/Windows | @... |
| Person 2 | macOS | @... |
| Person 3 | Windows | @... |

---

## Kurs-Kontext

- **Kurs:** ML4B (Machine Learning for Business) SoSe 2026
- **Dozenten:** Colin Frank, Markus Walk, Martin Enders
- **Methodik:** CRISP-DM + Related Work
- **Deliverable:** Streamlit Web Application + Präsentation
- **Finale Konferenz:** Schaeffler, Nürnberg (Nordostpark), Datum: TBA
