#!/bin/bash

# Exit on error
set -e

# Step 1: Ensure Python installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 not found. Please install Python3 and rerun."
    exit
fi

# Step 2: Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Step 3: Install requirements
echo "Installing dependencies..."
pip install --upgrade pip

pip install -r requirements.txt
pip install onnxruntime-gpu==1.16.3 --extra-index-url https://download.pytorch.org/whl/cu117


# Step 4: Run Flask
echo "Starting Flask server..."
flask run --host=0.0.0.0 --port=8521
