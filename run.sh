#!/bin/bash

echo "========================================"
echo "NASA Space Apps Challenge 2025"
echo "Exoplanet Detection AI System"
echo "========================================"
echo

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher from https://python.org"
    exit 1
fi

echo
echo "Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "Starting Exoplanet Detection AI..."
echo "The application will open in your default web browser"
echo "Press Ctrl+C to stop the application"
echo

streamlit run main.py
