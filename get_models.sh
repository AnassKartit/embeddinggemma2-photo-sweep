#!/bin/sh
# Face models from opencv_zoo: YuNet (MIT) finds faces, SFace (Apache 2.0) compares them. EmbeddingGemma 2 downloads itself on first run.
set -e; mkdir -p models
curl -sL -o models/yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -sL -o models/sface.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
ls -la models
