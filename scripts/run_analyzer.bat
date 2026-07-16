@echo off
setlocal

REM Change to the repository root (parent of scripts\).
cd /d "%~dp0\.."

REM Use the virtual environment Python if it exists, otherwise fall back.
set "VENV_PYTHON=%~dp0\..\.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    echo Using .venv Python.
) else (
    echo Warning: %VENV_PYTHON% not found. Using system Python.
    set "VENV_PYTHON=python"
)

REM Make the source package importable.
set PYTHONPATH=src

REM Pass all command-line arguments through to the analyzer.
%VENV_PYTHON% -m badminton_risk.video_risk_analyzer %*

endlocal
