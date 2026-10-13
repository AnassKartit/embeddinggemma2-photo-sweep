"""Face fingerprints with two models, so they can be compared on the same photos.
  sface:   OpenCV SFace (128 numbers, 37 MB), what the video used.
  arcface: InsightFace ArcFace R50 trained on WebFace600K (512 numbers, 174 MB, buffalo_l pack). Stronger on hard
           faces (age gaps, side views, low light). Weights are for non-commercial use (InsightFace licence).
Both use the same finder (YuNet) and the same 5 face points; ArcFace gets the standard 112x112 alignment from them.
Get ArcFace: download https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip and copy
w600k_r50.onnx to models/arcface_w600k_r50.onnx."""
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def finder(score=0.85):
    return cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (320, 320), score, 0.3, 5000)


class Embedder:
    def __init__(self, model="arcface"):
        self.model = model
        if model == "sface": self.sf = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")
        else:
            import onnxruntime as ort
            self.ort = ort.InferenceSession(str(ROOT / "models/arcface_w600k_r50.onnx"), providers=["CPUExecutionProvider"])
            self.inp = self.ort.get_inputs()[0].name

    def crop(self, img, f):
        """112x112 aligned face (BGR) from a YuNet row."""
        if self.model == "sface": return self.sf.alignCrop(img, f)
        M, _ = cv2.estimateAffinePartial2D(np.array(f[4:14], np.float32).reshape(5, 2), REF, method=cv2.LMEDS)
        return cv2.warpAffine(img, M, (112, 112), borderValue=0)

    def vec(self, crop):
        if self.model == "sface": x = self.sf.feature(crop).flatten()
        else:
            b = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
            x = self.ort.run(None, {self.inp: b.transpose(2, 0, 1)[None]})[0].flatten()
        return x / np.linalg.norm(x)
