"""Find each family member BY NAME from one photo of them, then check every photo.
Enroll: data/refs/us_family_sheet.jpg (five people side by side) -> one face fingerprint per name (YuNet finds, SFace fingerprints).
Test: the 29 family photos (who is in each was labelled by eye first: data/names_truth.json), the 12 photos of two other
families and the 40 real strangers (nobody should be found there). Each face in a photo goes to the closest name if the
similarity is at least TH, and at most one face per name per photo. Threshold 0.5 was fixed before this test (bench/oneshot.py).
Also: the same with 3 photos per person tagged (the "tag a few" trick), to see what more examples buy.
Usage: .venv/bin/python bench/names.py [sface|arcface] [threshold]   -> results/names.json (sface) or results/names_arcface_<th>.json"""
import json, random, statistics as st, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "agent"))
from faceid import Embedder, finder
MODEL = sys.argv[1] if len(sys.argv) > 1 else "sface"; TH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
TRUTH = json.load(open(ROOT / "data/names_truth.json")); NAMES = list(TRUTH["names"])
det = finder(0.8); emb = Embedder(MODEL)
DET = sys.argv[3] if len(sys.argv) > 3 else "yunet"   # "scrfd" = Immich's finder (det_10g, min score 0.7, 640 px)
if DET == "scrfd":
    from insightface.app import FaceAnalysis
    scrfd = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models/insightface"), allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    scrfd.prepare(ctx_id=-1, det_thresh=0.7, det_size=(640, 640))


def faces(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB"); img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]; k = 1280 / max(h, w)
    if k < 1: img = cv2.resize(img, (int(w * k), int(h * k))); h, w = img.shape[:2]
    if DET == "scrfd": found = [np.r_[f.bbox[:2], f.bbox[2:] - f.bbox[:2], f.kps.flatten()] for f in scrfd.get(img)]
    else: det.setInputSize((w, h)); _, found = det.detect(img)
    out = []
    for f in (found if found is not None else []):
        if min(f[2], f[3]) >= 24:
            out.append((float(f[0]), emb.vec(emb.crop(img, f))))
    return out


def who(F, gallery):
    """gallery: name -> array of fingerprints. Returns {name: score} for the names found in this photo."""
    cand = sorted(((float((g @ x).max()), n, i) for i, (_, x) in enumerate(F) for n, g in gallery.items()), reverse=True)
    got, used = {}, set()
    for sc, n, i in cand:                                   # best pairs first: one face per name, one name per face
        if sc < TH: break
        if n in got or i in used: continue
        got[n] = round(sc, 3); used.add(i)
    return got


def score(pred, truth):
    rows = {n: {"found": 0, "in": 0, "wrong": 0} for n in NAMES}
    for p, t in truth.items():
        for n in NAMES:
            if n in t: rows[n]["in"] += 1; rows[n]["found"] += n in pred[p]
            elif n in pred[p]: rows[n]["wrong"] += 1
    return rows


if __name__ == "__main__":
    t0 = time.time()
    sheet = sorted(faces(ROOT / "data/refs/us_family_sheet.jpg"), key=lambda f: f[0]); assert len(sheet) == 5, len(sheet)
    one = {n: np.stack([x]) for n, (_, x) in zip(NAMES, sheet)}
    fam = {p: faces(ROOT / "data/family" / p) for p in TRUTH["photos"]}
    neg = {p.name: faces(p) for d in ("other_family", "other_people") for p in sorted((ROOT / "data" / d).glob("*.jpg"))}
    secs = time.time() - t0; n_photos = len(fam) + len(neg) + 1
    pred = {p: who(F, one) for p, F in fam.items()}
    neg_pred = {p: who(F, one) for p, F in neg.items()}
    rows = score(pred, TRUTH["photos"])
    whole = [p for p, t in TRUTH["photos"].items() if len(t) == 5]
    res = {"model": MODEL, "finder": DET, "threshold": TH, "enroll": "one photo (the family sheet)", "photos": n_photos, "seconds_all_faces": round(secs, 1),
           "per_name": rows,
           "found_total": f'{sum(r["found"] for r in rows.values())}/{sum(r["in"] for r in rows.values())}',
           "wrong_in_family_photos": sum(r["wrong"] for r in rows.values()),
           "whole_family_photos_all_5_found": f'{sum(len(pred[p]) == 5 for p in whole)}/{len(whole)}',
           "names_found_in_other_families_or_strangers": {p: g for p, g in neg_pred.items() if g},
           "negatives_checked": len(neg), "misses": {p: sorted(set(t) - set(pred[p])) for p, t in TRUTH["photos"].items() if set(t) - set(pred[p])},
           "pred": pred}
    # tag a few: 3 random family photos per person become extra examples (label the face by its best match to the sheet), scored on the rest
    random.seed(0); runs = []
    for r in range(20):
        g = {n: [one[n][0]] for n in NAMES}; used = set()
        for n in NAMES:
            for p in random.sample([p for p, t in TRUTH["photos"].items() if n in t], 3):
                F = fam[p]
                if F: g[n].append(max(F, key=lambda f: float(one[n][0] @ f[1]))[1]); used.add(p)
        g = {n: np.stack(v) for n, v in g.items()}
        test = {p: t for p, t in TRUTH["photos"].items() if p not in used}
        rr = score({p: who(fam[p], g) for p in test}, test)
        runs.append((sum(x["found"] for x in rr.values()) / sum(x["in"] for x in rr.values()),
                     sum(1 for F in neg.values() if who(F, g))))
    res["tag_3_per_person"] = {"recall_median": round(st.median(a for a, _ in runs), 3), "negatives_with_a_name_median": st.median(b for _, b in runs), "runs": 20}
    json.dump(res, open(ROOT / ("results/names.json" if MODEL == "sface" else f"results/names_{MODEL}_{TH}{'_' + DET if DET != 'yunet' else ''}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "pred"}, indent=1))
