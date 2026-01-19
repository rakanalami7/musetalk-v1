#!/bin/bash
# Run MuseTalk API Server

# Set the script directory as working directory
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the server
python -m server.main

