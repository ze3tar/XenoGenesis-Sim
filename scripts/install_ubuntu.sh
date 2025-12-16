#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-venv python3-pip ffmpeg
if sudo apt-get install -y libfftw3-dev; then
  echo "FFTW3 installed"
else
  echo "FFTW3 unavailable; NumPy FFT fallback will be used" >&2
fi
