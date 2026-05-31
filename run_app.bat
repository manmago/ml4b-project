@echo off
REM One-click launcher for the ML4B Streamlit app (Windows).
REM
REM Usage:  double-click run_app.bat, or run it from a terminal:  run_app.bat
REM
REM Requires only `uv` to be installed (see docs\setup\Setup_Windows.md).
REM `uv run` automatically creates the virtual environment and installs every
REM dependency from pyproject.toml / uv.lock on first launch — no manual
REM `uv sync`, no pip, no conda, no separate Python install needed. The trained
REM model is committed, so no dataset download is required just to run the app.

REM Run from this script's own directory so relative paths resolve.
cd /d "%~dp0"

REM Fail early with a friendly message if uv is missing.
where uv >nul 2>nul
if errorlevel 1 (
  echo ERROR: 'uv' is not installed.
  echo Install it ^(PowerShell^):  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
  echo Then re-run:  run_app.bat
  pause
  exit /b 1
)

echo Starting the ML4B app - open http://localhost:8501 in your browser.
uv run streamlit run app/streamlit_app.py
pause
