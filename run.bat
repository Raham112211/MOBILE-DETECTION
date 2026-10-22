@echo off
title RF-DETR Small Mobile Phone Detection Studio
color 0b

echo =========================================================================
echo    RF-DETR Small Mobile Phone Detection & ByteTrack Studio
echo    GPU Hardware Acceleration: NVDEC Decoding + FP16 + NVENC CQ18
echo =========================================================================
echo.

set PYTHON_VENV=C:\Users\raham\.gemini\antigravity-ide\scratch\mobile_detection\.venv\Scripts\python.exe

if not exist "%PYTHON_VENV%" (
    echo [ERROR] Virtual environment python not found at:
    echo "%PYTHON_VENV%"
    echo Using system python fallback...
    set PYTHON_VENV=python
)

echo Starting Flask Server at http://127.0.0.1:5001 ...
"%PYTHON_VENV%" backend\app.py

pause
