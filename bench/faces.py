"""Face matching for "keep my people": YuNet finds faces (MIT), SFace turns each face into a 128-number print (Apache 2.0).
Both from opencv_zoo, a few MB, run on iOS and Android through OpenCV or ONNX. Same WhatsApp-style copies as bench.py.

The user marks k family photos; every face in them is enrolled. A photo is KEPT as family if any face in it matches an
enrolled face at SFace's published cosine threshold (0.363, not tuned on this data). 20 random trials per k.
Usage: .venv/bin/python bench/faces.py
"""
import json, random, time
from pathlib import Path

import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE, OUT = ROOT / "data", ROOT / "data" / "_whatsapp", ROOT / "results"
TH = 0.363
random.seed(0)

det = cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (320, 320), 0.8, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")

def category(name):
    top = name.split("__")[0]
    if top in ("family", "other_family", "other_people"): return top
    return "junk"

def faces(path):
    img = cv2.imread(str(path)); h, w = img.shape[:2]; k = 1280 / max(h, w)
    if k < 1: img = cv2.resize(img, (int(w * k), int(h * k))); h, w = img.shape[:2]
    det.setInputSize((w, h)); _, found = det.detect(img)
    feats = []
    for f in (found if found is not None else []):
        if min(f[2], f[3]) < 24: continue                                   # too small to recognise
        feats.append(rec.feature(rec.alignCrop(img, f)).flatten())
    return [x / np.linalg.norm(x) for x in feats]

items = [{"name": p.stem, "cat": category(p.stem)} for p in sorted(CACHE.glob("*.jpg"))]
t = time.time(); F = [faces(CACHE / f"{i['name']}.jpg") for i in items]; ms = (time.time() - t) / len(items) * 1000
cats = sorted({i["cat"] for i in items})
print(len(items), "images,", round(ms, 1), "ms/img (CPU),", {c: sum(i["cat"] == c for i in items) for c in cats})
print("photos with >=1 face:", {c: round(np.mean([len(F[n]) > 0 for n, i in enumerate(items) if i["cat"] == c]), 2) for c in cats})

fam = [n for n, i in enumerate(items) if i["cat"] == "family"]
def trial(k):
    shots = random.sample(fam, k); enrolled = [f for n in shots for f in F[n]]
    E = np.stack(enrolled) if enrolled else np.zeros((0, 128))
    kept = {}
    for n, i in enumerate(items):
        if n in shots: continue
        s = max((float((E @ f).max()) for f in F[n]), default=-1) if len(E) else -1
        kept.setdefault(i["cat"], []).append(s >= TH)
    return {c: float(np.mean(v)) for c, v in kept.items()}

res = {"threshold": TH, "ms_per_image_cpu": round(ms, 1), "faces_found": {c: round(float(np.mean([len(F[n]) > 0 for n, i in enumerate(items) if i["cat"] == c])), 3) for c in cats}, "k": {}}
for k in (1, 3, 5):
    rows = [trial(k) for _ in range(20)]
    res["k"][k] = {c: round(float(np.mean([r[c] for r in rows])), 3) for c in cats}
    print(f"k={k} kept as family:", res["k"][k])
OUT.mkdir(exist_ok=True); json.dump(res, open(OUT / "faces.json", "w"), indent=1)
