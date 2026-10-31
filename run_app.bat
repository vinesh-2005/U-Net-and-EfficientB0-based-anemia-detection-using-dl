@echo off
title Anemia Screening - AI Conjunctival Analysis Server
echo ========================================================
echo   ANEMEIA - AI Conjunctival Screening Server
echo ========================================================
echo.
cd /d "%~dp0"

if exist "C:\anemia_venv\Scripts\python.exe" (
    set "PYTHON_EXEC=C:\anemia_venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXEC=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

echo Starting server with %PYTHON_EXEC% ...
echo Open your browser at http://127.0.0.1:5000
echo.
%PYTHON_EXEC% app.py
pause
