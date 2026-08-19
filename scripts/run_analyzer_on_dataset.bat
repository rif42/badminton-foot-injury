@echo off
setlocal enabledelayedexpansion

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

REM Create results directory if it doesn't exist.
if not exist "data\results" mkdir data\results

REM Make the source package importable.
set PYTHONPATH=src

REM Process each MP4 file in the dataset folder.
for %%f in ("data\dataset\*.mp4") do (
    set "name=%%~nf"
    echo Processing %%f ...
    %VENV_PYTHON% -m badminton_risk.video_risk_analyzer "%%f" ^
        --output-csv "data\results\!name!_risk_report.csv" ^
        --output-video "data\results\!name!_annotated.mp4"
)

echo.
echo All files processed. Results are in data\results\

endlocal
