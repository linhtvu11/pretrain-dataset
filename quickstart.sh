#!/bin/bash
# Quick start script for LLM Pretraining Dataset Collection

set -e  # Exit on error

echo "========================================="
echo "LLM Pretraining Dataset - Quick Start"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run test cleaning
echo ""
echo "========================================="
echo "Running cleaning tests..."
echo "========================================="
python test_cleaning.py

# Check if test passed
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
    echo ""
    echo "========================================="
    echo "Ready to use!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Edit datasets_config.yaml to customize:"
    echo "   - Update 'hub_repo' with your HuggingFace username"
    echo "   - Select which datasets to include"
    echo "   - Adjust filtering parameters"
    echo ""
    echo "2. Login to HuggingFace (if you want to push):"
    echo "   huggingface-cli login"
    echo ""
    echo "3. Test with limited samples:"
    echo "   python merge_datasets.py --max-samples 100"
    echo ""
    echo "4. Run full merge (this may take a while):"
    echo "   python merge_datasets.py --output-dir ./merged_dataset"
    echo ""
    echo "5. Push to HuggingFace Hub:"
    echo "   python merge_datasets.py --push-to-hub --hub-repo your-username/dataset-name"
    echo ""
    echo "For more options, see: python merge_datasets.py --help"
    echo ""
else
    echo ""
    echo "✗ Tests failed. Please check the output above."
    exit 1
fi
