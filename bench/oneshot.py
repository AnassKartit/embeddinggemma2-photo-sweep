"""One-shot: the user gives ONE group photo of the family (data/refs/us_family_sheet.jpg, not in the test set).
A) faces: enrol every face in that photo (YuNet + SFace), keep a photo if >= `need` of its faces match (all, if it has fewer).
B) EmbeddingGemma 2 alone: rank all photos by cosine to that one photo.
Usage: .venv/bin/python bench/oneshot.py"""
import json
from pathlib import Path
import numpy as np, torch, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT / "data/_whatsapp"; SHOT = ROOT / "data/refs/us_family_sheet.jpg"
det = cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (320, 320), 0.8, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")
def faces(path):
    img = cv2.imread(str(path)); h, w = img.shape[:2]; k = 1280 / max(h, w)
    if k < 1: img = cv2.resize(img, (int(w * k), int(h * k))); h, w = img.shape[:2]
    det.setInputSize((w, h)); _, found = det.detect(img); out = []
    for f in (found if found is not None else []):
        if min(f[2], f[3]) >= 24: x = rec.feature(rec.alignCrop(img, f)).flatten(); out.append(x / np.linalg.norm(x))
    return out
def cat(stem):
    top = stem.split("__")[0]; return top if top in ("family", "other_family", "other_people") else "junk"
files = sorted(CACHE.glob("*.jpg")); y = np.array([cat(p.stem) for p in files])
enrolled = np.stack(faces(SHOT)); print("faces enrolled from the one photo:", len(enrolled))
F = [faces(p) for p in files]; res = {"enrolled_faces": len(enrolled), "faces": {}}
for th in (0.363, 0.45, 0.5):
    for need in (1, 2):
        keep = np.array([(lambda h: h > 0 and h >= min(need, len(f)))(sum(float((enrolled @ x).max()) >= th for x in f)) for f in F])
        r = {c: round(float(keep[y == c].mean()), 3) for c in ("family", "other_family", "other_people", "junk")}
        res["faces"][f"th{th}_need{need}"] = r; print(f"faces th={th} need>={need}:", r)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("google/embeddinggemma-2", device="mps", config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
ip = m[0].processor; ip.image_processor.max_soft_tokens = ip.image_processor.image_seq_length = ip.image_seq_length = 70
E = np.load(ROOT / "results/emb_70tok.npy"); q = m.encode([{"image": Image.open(SHOT).convert("RGB")}], normalize_embeddings=True)[0]
s = E @ q; order = np.argsort(-s); fam = (y == "family"); n = fam.sum()
top = y[order[:n]]
res["embedding"] = {"family_in_top_29": int((top == "family").sum()), "of": int(n), "top_29_mix": {c: int((top == c).sum()) for c in set(top)},
                    "auc": round(float((np.argsort(np.argsort(s))[fam].sum() - n * (n - 1) / 2) / (n * (~fam).sum())), 3)}
print("embedding only:", res["embedding"]); json.dump(res, open(ROOT / "results/oneshot.json", "w"), indent=1)

# Fair version: the sheet above is what Codex drew every scene from. Here the ONE example is an ordinary family scene with
# 3+ faces, averaged over every such scene.
fam_idx = [n for n in range(len(files)) if y[n] == "family" and len(F[n]) >= 3]; res["fair_one_scene"] = {"scenes_tried": len(fam_idx)}
for th, need in ((0.45, 1), (0.5, 1), (0.45, 2)):
    out = {}
    for shot in fam_idx:
        Eh = np.stack(F[shot])
        for n in range(len(files)):
            if n == shot: continue
            h = sum(float((Eh @ x).max()) >= th for x in F[n]); out.setdefault(y[n], []).append(h > 0 and h >= min(need, len(F[n])))
    res["fair_one_scene"][f"th{th}_need{need}"] = {c: round(float(np.mean(v)), 3) for c, v in out.items()}
    print(f"fair one scene th={th} need>={need}:", res["fair_one_scene"][f"th{th}_need{need}"])
json.dump(res, open(ROOT / "results/oneshot.json", "w"), indent=1)
