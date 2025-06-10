#!/bin/bash
set -e  # Exit on any error

echo "[INFO] Starting script at $(date)"

# Setup virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and install dependencies
echo "[INFO] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run your main Python script
echo "[INFO] Running app.py..."
# python app.py
python app.py

# Deactivate and optionally clean temp files
echo "[INFO] Deactivating virtual environment..."
deactivate

echo "[INFO] Cleaning up Python cache files..."
rm -rf __pycache__
find . -name "*.pyc" -delete

echo "[INFO] Script finished at $(date)"
