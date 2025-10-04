@echo off
echo ========================================
echo NASA Space Apps Challenge 2025
echo Exoplanet Detection AI System
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Starting Exoplanet Detection AI...
echo The application will open in your default web browser
echo Press Ctrl+C to stop the application
echo.

streamlit run main.py

pause
