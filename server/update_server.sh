#!/bin/bash

# Quick script to update the server files on RunPod
# This copies the fixed files to the server

echo "🔄 Updating MuseTalk server files..."

# Files that need to be updated
FILES=(
    "server/api/v1/session.py"
    "server/api/v1/generate.py"
    "server/config.py"
    "server/main.py"
)

echo "Files to update:"
for file in "${FILES[@]}"; do
    echo "  - $file"
done

echo ""
echo "To update your RunPod server:"
echo "1. Upload these files to your RunPod instance"
echo "2. Restart the server: python server/main.py"
echo ""
echo "Or use git to pull the latest changes:"
echo "  cd /workspace/musetalk-v1"
echo "  git pull"
echo "  python server/main.py"

