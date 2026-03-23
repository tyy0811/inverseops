#!/usr/bin/env bash
# Day 2: Data directory setup for FMD microscopy images
#
# This script creates the expected directory structure and provides
# instructions for placing the Fluorescence Microscopy Denoising (FMD)
# dataset files.

set -e

# Create directory structure
mkdir -p data/raw/fmd
mkdir -p data/processed
mkdir -p artifacts/samples

echo "========================================"
echo "InverseOps Data Directory Setup"
echo "========================================"
echo ""
echo "Created directories:"
echo "  data/raw/fmd/      - Place raw FMD microscopy images here"
echo "  data/processed/    - For processed/prepared data"
echo "  artifacts/samples/ - For sample outputs and visualizations"
echo ""
echo "Expected data layout:"
echo "  data/raw/fmd/"
echo "    ├── image_001.png"
echo "    ├── image_002.png"
echo "    └── ..."
echo ""
echo "Day 2 expects clean microscopy reference images (PNG, JPG, or TIFF)"
echo "to be placed under data/raw/fmd/. The dataset loader will recursively"
echo "discover image files and split them into train/val/test sets."
echo ""
echo "FMD Dataset:"
echo "  The Fluorescence Microscopy Denoising dataset can be obtained from:"
echo "  https://github.com/yinhaoz/denoising-fluorescence"
echo ""
echo "  After downloading, place the ground truth (GT) images in data/raw/fmd/"
echo ""
echo "Setup complete."
