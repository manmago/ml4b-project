# Setup Guide — WSL (Ubuntu) on Windows
> ML4B Gym Exercise Recognition Project | Python 3.11 | uv | VS Code

---

## 0. Voraussetzungen prüfen

Öffne ein **WSL-Terminal** (Ubuntu) und prüfe folgendes:

```bash
# Git vorhanden?
git --version           # sollte >= 2.x sein

# uv vorhanden?
uv --version            # falls nicht → Schritt 1

# Python vorhanden?
python3 --version       # wird von uv verwaltet, muss nicht systemweit sein
```

---

## 1. uv installieren (Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Shell neu laden (oder Terminal neu öffnen)
source ~/.bashrc   # oder: source ~/.zshrc

# Prüfen
uv --version
```

---

## 2. Python 3.11 mit uv installieren

```bash
# Verfügbare Versionen anzeigen
uv python list

# Python 3.11 installieren
uv python install 3.11

# Prüfen
uv python list --only-installed
```

---

## 3. Git konfigurieren & SSH-Key für GitHub

### 3a. Name und E-Mail setzen

```bash
git config --global user.name "Dein Name"
git config --global user.email "deine@email.com"

# Prüfen
git config --global user.name
git config --global user.email
```

### 3b. SSH-Key prüfen oder neu erstellen

```bash
# Vorhandene Keys prüfen
ls -la ~/.ssh/
# Suche nach: id_ed25519.pub oder id_rsa.pub
```

**Falls kein Key vorhanden:**

```bash
# Neuen Ed25519-Key erstellen (empfohlen)
ssh-keygen -t ed25519 -C "deine@email.com"
# Einfach Enter drücken für Standard-Pfad (~/.ssh/id_ed25519)
# Optional: Passphrase setzen (empfohlen)

# SSH-Agent starten und Key hinzufügen
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 3c. Public Key zu GitHub hinzufügen

```bash
# Public Key anzeigen und kopieren
cat ~/.ssh/id_ed25519.pub
```

→ Gehe zu GitHub: **Settings → SSH and GPG keys → New SSH key**  
→ Titel: z.B. `WSL Ubuntu`  
→ Key einfügen und speichern

### 3d. Verbindung testen

```bash
ssh -T git@github.com
# Erwartete Ausgabe: "Hi <username>! You've successfully authenticated..."
```

### 3e. SSH-Agent automatisch starten (optional, aber komfortabel)

Füge folgendes in `~/.bashrc` (oder `~/.zshrc`) ein:

```bash
# SSH-Agent auto-start
if [ -z "$SSH_AUTH_SOCK" ]; then
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519 2>/dev/null
fi
```

---

## 4. Repository clonen

```bash
# Navigiere zu deinem Projektordner
cd ~/projects

# Repo clonen (SSH, kein HTTPS → kein Passwort nötig)
git clone git@github.com:<DEIN_ORG_ODER_USER>/ml4b-project.git

# In den Ordner wechseln
cd ml4b-project
```

---

## 5. Projekt mit uv initialisieren

```bash
# Im Projektordner (ml4b-project/)
# Falls pyproject.toml noch nicht existiert:
uv init --python 3.11

# Python-Version im Projekt fixieren
echo "3.11" > .python-version

# Benötigte Pakete hinzufügen (erstellt automatisch .venv)
uv add pandas numpy scikit-learn streamlit ipykernel matplotlib seaborn plotly

# Dev-Dependencies (nur für Entwicklung, nicht in Production)
uv add --dev jupyter pytest black ruff mypy

# Umgebung synchronisieren (nach git pull / erster Einrichtung)
uv sync
```

---

## 6. VS Code einrichten

### 6a. VS Code Extensions installieren

Öffne VS Code und installiere diese Extensions:

- `ms-python.python` — Python Support
- `ms-toolsai.jupyter` — Jupyter Notebooks
- `ms-vscode-remote.remote-wsl` — WSL Integration (**wichtig für Windows!**)
- `charliermarsh.ruff` — Linter/Formatter
- `eamodio.gitlens` — Git Superpowers
- `christian-kohler.path-intellisense` — Pfad-Autovervollständigung

### 6b. VS Code aus WSL öffnen

```bash
# Im Projektordner
code .
# VS Code öffnet sich im WSL-Modus (erkennbar am grünen ">< WSL" Badge links unten)
```

### 6c. Python Interpreter setzen

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Wähle den Interpreter aus `.venv` → sollte automatisch erkannt werden
3. Pfad sollte aussehen wie: `./.venv/bin/python`

### 6d. VS Code Workspace Settings

Erstelle `.vscode/settings.json` im Projektordner:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
  },
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true
  }
}
```

---

## 7. Streamlit App testen

```bash
# App starten (aus Projektordner)
uv run streamlit run app/streamlit_app.py
```

Browser öffnet sich automatisch auf `http://localhost:8501`

---

## 8. Git Workflow (Tägliche Arbeit)

```bash
# Vor dem Arbeiten: aktuellen Stand holen
git pull origin main

# Neuen Feature-Branch erstellen
git checkout -b feature/dein-feature-name

# Änderungen committen
git add .
git commit -m "feat: kurze beschreibung der änderung"

# Branch pushen
git push origin feature/dein-feature-name

# → Danach: Pull Request auf GitHub erstellen
```

### Commit Message Konventionen

```
feat:     neues Feature
fix:      Bugfix
docs:     Dokumentation
style:    Formatierung (kein Code-Änderung)
refactor: Code-Umstrukturierung
data:     Daten hinzugefügt/geändert
model:    Modell-Änderungen
```

---

## 9. Häufige Probleme (WSL)

| Problem | Lösung |
|---------|--------|
| `uv: command not found` | `source ~/.bashrc` ausführen |
| SSH-Key wird nicht gefunden | `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519` |
| VS Code öffnet sich nicht im WSL-Modus | Remote-WSL Extension installieren, dann `code .` aus WSL |
| `.venv` nicht erkannt in VS Code | `Ctrl+Shift+P` → Python: Select Interpreter → `.venv` wählen |
| `uv sync` schlägt fehl | `uv lock --upgrade` und dann `uv sync` |

---

## Schnellreferenz

```bash
uv add <paket>          # Paket hinzufügen
uv remove <paket>       # Paket entfernen
uv sync                 # Umgebung synchronisieren (nach git pull!)
uv run python script.py # Script ausführen
uv run streamlit run app/streamlit_app.py  # Streamlit starten
uv run pytest           # Tests ausführen
uv run jupyter lab      # Jupyter Lab starten
```
