#!/usr/bin/env bash
# Fetch face-detection models (Stage 1). YuNet ONNX (primary) + Haar cascade (fallback).
set -u
cd "$(dirname "$0")/.."
mkdir -p assets/models
cd assets/models
curl -sL -o face_detection_yunet.onnx -w 'yunet %{http_code} %{size_download}\n' \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
curl -sL -o haarcascade_frontalface_default.xml -w 'haar %{http_code} %{size_download}\n' \
  "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
