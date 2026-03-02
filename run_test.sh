#!/bin/bash

# Helper script to run test_installation.py with virtual environment activated

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: sh setup.sh"
    exit 1
fi

# Activate venv and run test
source venv/bin/activate
python3 test_installation.py
