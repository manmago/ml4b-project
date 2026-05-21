# Setup Guide — macOS
> ML4B Gym Exercise Recognition Project | Python 3.11 | uv | VS Code

---

## 0. Voraussetzungen prüfen

Öffne das **Terminal** (`Cmd+Space` → "Terminal") und prüfe:

```bash
# Git vorhanden? (auf macOS meist vorinstalliert via Xcode CLI Tools)
git --version

# uv vorhanden?
uv --version            # falls nicht → Schritt 1

# Xcode Command Line Tools (falls git fehlt)
xcode-select --install
```

---

## 1. uv installieren (Package Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Shell neu laden
source ~/.zshrc   # Zsh (macOS Standard seit Catalina)
# oder: source ~/.bash_profile (falls Bash)

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

> **Hinweis:** Kein Homebrew-Python nötig — uv verwaltet Python vollständig isoliert.

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
# Standard-Pfad mit Enter bestätigen: ~/.ssh/id_ed25519
# Optional: Passphrase setzen (empfohlen)

# SSH-Agent starten und Key hinzufügen
eval "$(ssh-agent -s)"
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

### 3c. SSH-Config für macOS Keychain (dauerhaft)

Erstelle/ergänze `~/.ssh/config`:

```bash
cat >> ~/.ssh/config << 'EOF'
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
EOF
```

### 3d. Public Key zu GitHub hinzufügen

```bash
# Public Key in Zwischenablage kopieren
pbcopy < ~/.ssh/id_ed25519.pub
```

→ Gehe zu GitHub: **Settings → SSH and GPG keys → New SSH key**  
→ Titel: z.B. `MacBook Pro`  
→ Mit `Cmd+V` einfügen und speichern

### 3e. Verbindung testen

```bash
ssh -T git@github.com
# Erwartete Ausgabe: "Hi <username>! You've successfully authenticated..."
```

---

## 4. Repository clonen

```bash
# Navigiere zu deinem Projektordner
cd ~/projects   # oder wo auch immer du Projekte ablegst
mkdir -p ~/projects && cd ~/projects

# Repo clonen (SSH)
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

# Benötigte Pakete hinzufügen
uv add pandas numpy scikit-learn streamlit ipykernel matplotlib seaborn plotly

# Dev-Dependencies
uv add --dev jupyter pytest black ruff mypy

# Umgebung synchronisieren
uv sync
```

### 5b. Umgebungsvariablen konfigurieren (.env)

```bash
# .env.example als Vorlage kopieren
cp .env.example .env
```

Die `.env` ist **optional** — du brauchst sie nur, wenn deine RecoFit-Daten **nicht** unter `data/raw/` im Projektordner liegen (z.B. auf einer externen Festplatte):

```bash
# .env öffnen und Pfade anpassen (nur wenn nötig)
nano .env   # oder: open -e .env
```

```
# Beispiel: Daten auf externer Festplatte
ML4B_DATA_RAW=/Volumes/ExterneFestplatte/datasets/recofit
ML4B_DATA_PROCESSED=/Volumes/ExterneFestplatte/datasets/processed
ML4B_MODELS_DIR=/Volumes/ExterneFestplatte/datasets/models
```

| Variable | Standard | Bedeutung |
|----------|----------|-----------|
| `ML4B_DATA_RAW` | `data/raw` | Ordner mit der RecoFit `.mat`-Datei |
| `ML4B_DATA_PROCESSED` | `data/processed` | Ausgabe für Feature-CSVs |
| `ML4B_MODELS_DIR` | `models/saved` | Ausgabe für trainierte Modelle |

> **Tipp:** Wenn deine RecoFit-Datei unter `ml4b-project/data/raw/recofit/` liegt, musst du **nichts ändern** — die Standardpfade greifen automatisch.

---

## 6. VS Code einrichten

### 6a. VS Code installieren

Download: https://code.visualstudio.com/  
Oder via Homebrew: `brew install --cask visual-studio-code`

### 6b. VS Code Extensions installieren

- `ms-python.python` — Python Support
- `ms-toolsai.jupyter` — Jupyter Notebooks
- `charliermarsh.ruff` — Linter/Formatter
- `eamodio.gitlens` — Git Superpowers
- `christian-kohler.path-intellisense` — Pfad-Autovervollständigung

### 6c. Shell-Befehl `code` aktivieren

1. VS Code öffnen
2. `Cmd+Shift+P` → "Shell Command: Install 'code' command in PATH"

```bash
# Dann aus Terminal
cd ~/projects/ml4b-project
code .
```

### 6d. Python Interpreter setzen

1. `Cmd+Shift+P` → "Python: Select Interpreter"
2. Wähle: `./.venv/bin/python`

### 6e. VS Code Workspace Settings

Erstelle `.vscode/settings.json`:

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
uv run streamlit run app/streamlit_app.py
```

---

## 8. Git Workflow (Tägliche Arbeit)

```bash
git pull origin main
git checkout -b feature/dein-feature-name
git add .
git commit -m "feat: kurze beschreibung"
git push origin feature/dein-feature-name
# → Pull Request auf GitHub erstellen
```

---

## 9. Häufige Probleme (macOS)

| Problem | Lösung |
|---------|--------|
| `uv: command not found` | `source ~/.zshrc` ausführen |
| SSH-Passphrase jedes Mal nötig | `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` |
| `permission denied (publickey)` | SSH-Key nicht auf GitHub → Schritt 3c wiederholen |
| VS Code öffnet falsches Python | `Cmd+Shift+P` → Python: Select Interpreter → `.venv` wählen |

---

## Schnellreferenz

```bash
uv add <paket>          # Paket hinzufügen
uv sync                 # Umgebung synchronisieren (nach git pull!)
uv run streamlit run app/streamlit_app.py
uv run pytest
uv run jupyter lab
```
