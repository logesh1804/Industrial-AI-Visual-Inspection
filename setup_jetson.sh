#!/usr/bin/env bash
# Setup & Optimization Script for NVIDIA Jetson Nano
# Usage: bash setup_jetson.sh

set -e

echo "================================================================="
echo "   NVIDIA JETSON NANO - INDUSTRIAL AI INSPECTION SETUP"
echo "================================================================="

# 1. Maximize Jetson Performance Mode
echo "[1/5] Setting Maximum Clock Performance (10W Mode)..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0 || true
fi
if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks || true
fi

# 2. Add User to Video Group for Camera Access
echo "[2/5] Configuring camera permissions..."
sudo usermod -aG video $USER || true

# 3. System Packages
echo "[3/5] Installing required system libraries..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    zlib1g-dev \
    v4l-utils \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad

# 4. Python Dependencies
echo "[4/5] Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt || true
python3 -m pip install ultralytics opencv-python onnxruntime-gpu || python3 -m pip install onnxruntime

# 5. Verify Jetson Environment
echo "[5/5] Running Jetson Diagnostics..."
python3 src/test_jetson_nano.py

echo "================================================================="
echo "   JETSON NANO SETUP COMPLETE! Ready for inspection."
echo "================================================================="
