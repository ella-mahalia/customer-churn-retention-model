#!/bin/bash
set -e

python3 -m pip install -r requirements.txt
python3 src/download_data.py
python3 src/train_model.py

echo ""
echo "Real IBM dataset downloaded and model trained."
echo "Now run:"
echo "  git add ."
echo '  git commit -m "Replace synthetic churn data with IBM real dataset"'
echo "  git push"
