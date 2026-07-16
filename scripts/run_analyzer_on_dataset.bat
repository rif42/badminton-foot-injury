@echo off
setlocal enabledelayedexpansion

REM Change to the repository root (parent of scripts\).
cd /d "%~dp0\.."

REM Create results directory if it doesn't exist.
if not exist "data\results" mkdir data\results

REM Activate the virtual environment if it exists.
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Warning: .venv not found. Using system Python.
)

REM Make the source package importable.
set PYTHONPATH=src

REM Process each MP4 file in the dataset folder.
for %%f in ("data\dataset\*.mp4") do (
    set "name=%%~nf"
    echo Processing %%f ...
    python -m badminton_risk.video_risk_analyzer "%%f" ^
        --output-csv "data\results\!name!_risk_report.csv" ^
        --output-video "data\results\!name!_annotated.mp4"
)

echo.
echo All files processed. Results are in data\results\

endlocal
