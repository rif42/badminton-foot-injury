@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Run the badminton injury-risk analyzer over every video in
REM  the dataset folder. Outputs land in data\results\:
REM    <name>_risk_report.csv    per-frame risk report
REM    <name>_annotated.mp4      annotated video
REM    <name>_critical.json      JSON log of critical (risky) detections
REM
REM  Usage:
REM    run_analyzer_on_dataset.bat            (uses data\dataset)
REM    run_analyzer_on_dataset.bat <folder>   (analyze videos in <folder>)
REM ============================================================

REM Change to the repository root (parent of scripts\) so the
REM script works when double-clicked from anywhere.
cd /d "%~dp0\.."

REM Optional dataset folder override (default: data\dataset).
set "DATASET_DIR=data\dataset"
if not "%~1"=="" set "DATASET_DIR=%~1"

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

echo.
echo Dataset folder: !DATASET_DIR!
echo.

REM Process each MP4 file in the dataset folder, retrying once if the
REM video writer fails to open (transient file lock on Windows).
set "COUNT=0"
for %%f in ("!DATASET_DIR!\*.mp4") do (
    set "name=%%~nf"
    echo Processing %%f ...
    call :analyze "%%f"
    if errorlevel 1 (
        echo   Writer open failed - waiting 3s and retrying %%f ...
        timeout /t 3 /nobreak >nul <nul
        call :analyze "%%f"
    )
    if errorlevel 1 (
        echo   ERROR processing %%f
    ) else (
        set /a COUNT+=1
    )
)

echo.
if !COUNT!==0 (
    echo No MP4 files found in !DATASET_DIR!.
) else (
    echo !COUNT! files processed. Results are in data\results\
)
endlocal
exit /b 0

:analyze
%VENV_PYTHON% -m badminton_risk.video_risk_analyzer "%~1" ^
    --output-csv "data\results\!name!_risk_report.csv" ^
    --output-video "data\results\!name!_annotated.mp4" ^
    --output-log "data\results\!name!_critical.json"
exit /b %errorlevel%
