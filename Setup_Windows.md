# Setup Guide — Windows (nativ, ohne WSL)
> ML4B Gym Exercise Recognition Project | Python 3.11 | uv | VS Code

---

## 0. Voraussetzungen prüfen

Öffne **PowerShell** (`Win+X` → "Windows PowerShell" oder "Terminal"):

```powershell
# Git vorhanden?
git --version

# uv vorhanden?
uv --version

# Python vorhanden (wird von uv verwaltet)?
python --version
```

---

## 1. Git installieren

Falls `git --version` fehlschlägt:

→ Download: https://git-scm.com/downloads/win  
→ Während Installation: **"Git Bash"** und **"Git from the command line"** auswählen

---

## 2. uv installieren (Package Manager)

In **PowerShell (als Administrator)**:

```powershell
winget install astral-sh.uv
```

Alternativ:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Terminal neu starten**, dann prüfen:

```powershell
uv --version
```

---

## 3. Python 3.11 mit uv installieren

```powershell
# Python 3.11 installieren
uv python install 3.11

# Prüfen
uv python list --only-installed
```

---

## 4. Git konfigurieren & SSH-Key für GitHub

### 4a. Name und E-Mail setzen (in PowerShell oder Git Bash)

```powershell
git config --global user.name "Dein Name"
git config --global user.email "deine@email.com"
```

### 4b. SSH-Key erstellen

In **Git Bash** (nicht PowerShell, da ssh-keygen dort manchmal fehlt):

```bash
# Vorhandene Keys prüfen
ls -la ~/.ssh/

# Neuen Key erstellen falls keiner vorhanden
ssh-keygen -t ed25519 -C "deine@email.com"
# Enter für Standard-Pfad, optional Passphrase

# SSH-Agent starten
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### 4c. Public Key zu GitHub hinzufügen

In Git Bash:

```bash
# Key anzeigen und manuell kopieren
cat ~/.ssh/id_ed25519.pub
```

→ GitHub: **Settings → SSH and GPG keys → New SSH key**  
→ Einfügen und speichern

### 4d. Verbindung testen

```bash
ssh -T git@github.com
```

---

## 5. Repository clonen

In **PowerShell** oder **Git Bash**:

```powershell
# Projektordner erstellen und navigieren
mkdir C:\Users\<DeinName>\projects
cd C:\Users\<DeinName>\projects

# Repo clonen
git clone git@github.com:<DEIN_ORG_ODER_USER>/ml4b-project.git

cd ml4b-project
```

> **Tipp:** Nutze keine Pfade mit Leerzeichen oder Sonderzeichen.

---

## 6. Projekt mit uv initialisieren

```powershell
# Im Projektordner
# Falls pyproject.toml noch nicht existiert:
uv init --python 3.11

# Python-Version fixieren
echo "3.11" > .python-version

# Pakete hinzufügen
uv add pandas numpy scikit-learn streamlit ipykernel matplotlib seaborn plotly

# Dev-Dependencies
uv add --dev jupyter pytest black ruff mypy

# Umgebung synchronisieren
uv sync
```

---

## 7. VS Code einrichten

### 7a. VS Code installieren

Download: https://code.visualstudio.com/  
→ Während Installation: **"Add to PATH"** aktivieren

### 7b. Extensions installieren

- `ms-python.python`
- `ms-toolsai.jupyter`
- `charliermarsh.ruff`
- `eamodio.gitlens`
- `christian-kohler.path-intellisense`

### 7c. VS Code öffnen

```powershell
cd C:\Users\<DeinName>\projects\ml4b-project
code .
```

### 7d. Python Interpreter setzen

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Wähle: `.venv\Scripts\python.exe`

### 7e. VS Code Workspace Settings

Erstelle `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
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

> **Hinweis:** Windows nutzt `\` statt `/` in Pfaden — das `${workspaceFolder}` wird aber von VS Code korrekt aufgelöst.

---

## 8. Streamlit App testen

```powershell
uv run streamlit run app/streamlit_app.py
```

---

## 9. Git Workflow

```powershell
git pull origin main
git checkout -b feature/dein-feature-name
git add .
git commit -m "feat: beschreibung"
git push origin feature/dein-feature-name
```

---

## 10. Häufige Probleme (Windows nativ)

| Problem | Lösung |
|---------|--------|
| `uv: command not found` | Terminal neu starten nach Installation |
| `ssh-keygen` nicht gefunden | Git Bash statt PowerShell verwenden |
| Zeilenenden-Probleme (CRLF vs LF) | `git config --global core.autocrlf input` setzen |
| `.venv` nicht erkannt | `Ctrl+Shift+P` → Python: Select Interpreter → `.venv\Scripts\python.exe` |
| Port 8501 blockiert | Windows Firewall-Ausnahme für Python hinzufügen |

### Wichtig: Zeilenenden (Line Endings)

Damit keine CRLF/LF-Konflikte mit Mac/Linux-Teammitgliedern entstehen:

```powershell
git config --global core.autocrlf input
```

---

## Schnellreferenz

```powershell
uv add <paket>
uv sync
uv run streamlit run app/streamlit_app.py
uv run pytest
uv run jupyter lab
```
