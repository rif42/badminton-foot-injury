@echo off
setlocal

REM Change to the repository root (parent of scripts\).
cd /d "%~dp0\.."

REM Activate the virtual environment if it exists.
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Warning: .venv not found. Using system Python.
)

REM Make the source package importable.
set PYTHONPATH=src

REM Pass all command-line arguments through to the analyzer.
python -m badminton_risk.video_risk_analyzer %*

endlocal
