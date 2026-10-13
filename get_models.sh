#!/bin/sh
# Face models from opencv_zoo: YuNet (MIT) finds faces, SFace (Apache 2.0) compares them. EmbeddingGemma 2 downloads itself on first run.
# ArcFace R50 (InsightFace buffalo_l, non-commercial weights) names people in the app ("find anyone by name").
set -e; mkdir -p models
curl -sL -o models/yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o models/sface.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
if [ ! -f models/arcface_w600k_r50.onnx ]; then
  tmp=$(mktemp -d); curl -sL -o "$tmp/buffalo_l.zip" https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  unzip -q -o -j "$tmp/buffalo_l.zip" "*w600k_r50.onnx" -d "$tmp" && mv "$tmp/w600k_r50.onnx" models/arcface_w600k_r50.onnx; rm -rf "$tmp"
fi
ls -la models
