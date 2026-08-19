#!/bin/bash

# Exit on error
set -e

echo "=========================================================="
echo "      Jetson Setup Script for Visual Inspection App       "
echo "=========================================================="

# 1. Detect L4T / JetPack Version
echo "[1/5] Detecting Jetson / JetPack Environment..."
L4T_VERSION=""
JETPACK_VERSION=""

if [ -f /etc/nv_tegra_release ]; then
    L4T_LINE=$(head -n 1 /etc/nv_tegra_release)
    echo "Found L4T Release info: $L4T_LINE"
    
    # Parse L4T release and revision
    # E.g. "# R35 (release), REVISION: 2.1" -> R35.2.1
    R_RELEASE=$(echo "$L4T_LINE" | grep -o -E 'R[0-9]+' || echo "")
    R_REVISION=$(echo "$L4T_LINE" | grep -o -E 'REVISION: [0-9]+\.[0-9]+' | awk '{print $2}' || echo "")
    if [ ! -z "$R_RELEASE" ] && [ ! -z "$R_REVISION" ]; then
        L4T_VERSION="${R_RELEASE}.${R_REVISION}"
        echo "Parsed L4T Version: $L4T_VERSION"
    fi
fi

# Fallback check via dpkg if L4T release file parsing wasn't sufficient
if [ -z "$L4T_VERSION" ]; then
    L4T_VERSION=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core 2>/dev/null | cut -d'-' -f1 || echo "")
    if [ ! -z "$L4T_VERSION" ]; then
        echo "Detected L4T Version via dpkg: $L4T_VERSION"
    fi
fi

# Detect JetPack package version via apt
APT_JETPACK=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}' || echo "")
if [ ! -z "$APT_JETPACK" ]; then
    echo "Detected JetPack Version via apt: $APT_JETPACK"
    JETPACK_VERSION=$APT_JETPACK
fi

# Map L4T to standard JetPack version if not found via apt
if [ -z "$JETPACK_VERSION" ] && [ ! -z "$L4T_VERSION" ]; then
    case "$L4T_VERSION" in
        36.*)
            JETPACK_VERSION="6.x (JetPack 6)"
            ;;
        35.*)
            JETPACK_VERSION="5.1.x (JetPack 5.1)"
            ;;
        34.*)
            JETPACK_VERSION="5.0.x (JetPack 5.0)"
            ;;
        32.7*)
            JETPACK_VERSION="4.6.1+ (JetPack 4.6)"
            ;;
        *)
            JETPACK_VERSION="Unknown (L4T $L4T_VERSION)"
            ;;
    esac
fi

echo "Estimated JetPack Version: ${JETPACK_VERSION:-Unknown}"
echo "Python Version: $(python3 --version 2>&1)"

# 2. Select the Custom Index for PyTorch Installation
echo ""
echo "[2/5] Selecting PyTorch/Torchvision Package Index..."
echo "For Jetson, standard pip wheels do not support CUDA. We must use Jetson AI Lab wheels."

INDEX_URL=""
if [[ "$L4T_VERSION" == 36.* ]]; then
    # JetPack 6
    echo "JetPack 6 detected. Defaulting to: jp6/cu122"
    INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu122"
elif [[ "$L4T_VERSION" == 35.* ]]; then
    # JetPack 5.1
    echo "JetPack 5.1 detected. Defaulting to: jp5/cu118"
    INDEX_URL="https://pypi.jetson-ai-lab.io/jp5/cu118"
else
    # Prompt user or default to JetPack 6
    echo "Could not reliably auto-select index URL. Please choose from the options below:"
    echo "1) JetPack 6 (JP6 / CUDA 12.2) [Default]"
    echo "2) JetPack 5 (JP5 / CUDA 11.8)"
    read -p "Select option (1-2): " choice
    if [ "$choice" = "2" ]; then
        INDEX_URL="https://pypi.jetson-ai-lab.io/jp5/cu118"
    else
        INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu122"
    fi
fi

echo "Using Index URL: $INDEX_URL"

# 3. Install System Dependencies
echo ""
echo "[3/5] Installing System Prerequisites via apt..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libopenblas-base \
    libopenmpi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python3-pil \
    python3-matplotlib

# OpenCV Check
echo "Checking OpenCV installation..."
if python3 -c "import cv2; print('OpenCV Version:', cv2.__version__)" 2>/dev/null; then
    echo "System OpenCV is already installed and working."
else
    echo "System OpenCV Python bindings not found. Installing via apt..."
    sudo apt-get install -y python3-opencv
fi

# 4. Install PyTorch & Torchvision
echo ""
echo "[4/5] Installing Jetson-optimized PyTorch and Torchvision..."
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url="$INDEX_URL"

# 5. Install Ultralytics YOLOv8 & Other Pip requirements
echo ""
echo "[5/5] Installing Ultralytics YOLOv8..."
# Note: Ultralytics expects torch/torchvision. Since they are already installed, pip will see the requirements are met.
python3 -m pip install ultralytics

echo ""
echo "=========================================================="
echo "Installation complete. Testing environment..."
echo "=========================================================="
python3 src/check_environment.py
