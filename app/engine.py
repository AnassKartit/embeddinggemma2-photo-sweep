"""Photo Sweep engine: everything runs on this machine.

EmbeddingGemma 2 (70 image tokens, 256 dims) sorts photos into piles and powers search; YuNet + SFace faces find the family from
one photo; a blur/darkness check catches accidental shots; cosine >= 0.93 marks near-duplicates. Numbers behind each choice:
REVIEW.md. Nothing is deleted: the app moves files into a Sweep folder next to the scanned folder and can put them back.
"""
import hashlib, json, time
from pathlib import Path

import cv2, numpy as np, torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
TOKENS, DIMS, DUP = 70, 256, 0.93
FACE_TH = 0.5                                             # one face at >= 0.5 (REVIEW.md, one-shot table)
FWD_MARGIN = 0.045                                        # "forwards" must beat "people" by this much, else the photo goes to review
EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic"}

LABELS = {                                                # zero-shot prototypes, averaged per pile
    "people":     ["a candid photo of a family", "a personal photo of people", "a smartphone photo of kids and parents"],
    "forwards":   ["a good morning greeting card image with decorative text", "a holiday or birthday e-card", "an internet meme with a caption",
                   "a funny comic or reaction meme", "a motivational or religious quote image", "a forwarded promotional flyer or advertisement",
                   "a viral infographic or chain message", "an invitation card"],
    "screenshots": ["a screenshot of a phone screen", "a screenshot of a website or chat"],
    "receipts":   ["a photo of a shop receipt", "a photo of a printed document"],
    "accidental": ["an accidental, blurry or black photo taken by mistake"],
}
PILES = [("family", "Family", "keep"), ("duplicates", "Duplicates", "clean"), ("forwards", "Forwards & memes", "clean"),
         ("screenshots", "Screenshots", "clean"), ("receipts", "Receipts & documents", "review"), ("accidental", "Blurry & accidental", "clean"),
         ("people", "Other people", "review")]


class Engine:
    def __init__(self, log=print):
        from sentence_transformers import SentenceTransformer
        t = time.time()
        self.model = SentenceTransformer("google/embeddinggemma-2", device="mps" if torch.backends.mps.is_available() else "cpu",
                                         config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
        ip = self.model[0].processor
        ip.image_processor.max_soft_tokens = ip.image_processor.image_seq_length = ip.image_seq_length = TOKENS
        self.det = cv2.FaceDetectorYN.create(str(ROOT / "models/yunet.onnx"), "", (320, 320), 0.8, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(ROOT / "models/sface.onnx"), "")
        names = list(LABELS)
        T = [self._norm(self._text(LABELS[n]).mean(0)) for n in names]
        self.label_names, self.label_vecs = names, np.stack(T)
        log(f"models ready in {time.time() - t:.1f} s")

    @staticmethod
    def _norm(v):
        v = v[..., :DIMS]; return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def _text(self, texts):
        return self.model.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)

    def text_vec(self, q):
        return self._norm(self._text([q])[0])

    def image_vecs(self, imgs):
        return self._norm(self.model.encode([{"image": im} for im in imgs], batch_size=8, normalize_embeddings=True, show_progress_bar=False))

    def faces(self, im: Image.Image):
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR); h, w = img.shape[:2]; k = 1280 / max(h, w)
        if k < 1: img = cv2.resize(img, (int(w * k), int(h * k))); h, w = img.shape[:2]
        self.det.setInputSize((w, h)); _, found = self.det.detect(img); out = []
        for f in (found if found is not None else []):
            if min(f[2], f[3]) >= 24:
                x = self.rec.feature(self.rec.alignCrop(img, f)).flatten(); out.append(x / np.linalg.norm(x))
        return out

    @staticmethod
    def accidental(im: Image.Image):
        g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F).var()) < 20     # no edges at all: pocket, black, smeared (real photos >= 408 here)


def load(path: Path):
    im = Image.open(path); im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((1600, 1600)); return im


def file_id(p: Path):
    return hashlib.sha1(str(p).encode()).hexdigest()[:16]


def scan(engine: Engine, folder: Path, family_faces, progress=lambda *a: None, cache_dir: Path = None):
    """Returns {'photos': [...], 'piles': {...}, 'stats': {...}}. Embeddings and faces are cached per file (path + size + mtime)."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXT and not p.name.startswith("."))
    cache_dir = cache_dir or folder / ".photo-sweep"; cache_dir.mkdir(exist_ok=True)
    cpath = cache_dir / "cache.npz"; meta_path = cache_dir / "cache.json"
    cache = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    vecs = dict(np.load(cpath, allow_pickle=True)) if cpath.exists() else {}
    photos, t0, done_new = [], time.time(), 0
    batch = []
    def flush():
        nonlocal done_new
        if not batch: return
        V = engine.image_vecs([b[1] for b in batch])
        for (key, im, rec), v in zip(batch, V):
            rec["faces"] = [f.tolist() for f in engine.faces(im)]; rec["accidental"] = bool(engine.accidental(im))
            vecs[key] = v; cache[key] = rec; done_new += 1
        batch.clear()
    for i, p in enumerate(files):
        st = p.stat(); key = f"{file_id(p)}_{st.st_size}_{int(st.st_mtime)}"
        if key not in cache:
            im = load(p); batch.append((key, im, {"w": im.width, "h": im.height}))
            if len(batch) == 8: flush()
        photos.append({"id": file_id(p), "name": p.name, "path": str(p), "bytes": st.st_size, "key": key})
        progress(i + 1, len(files), time.time() - t0)
    flush()
    np.savez(cpath, **vecs); meta_path.write_text(json.dumps(cache))
    E = np.stack([vecs[ph["key"]] for ph in photos]) if photos else np.zeros((0, DIMS))
    S = E @ engine.label_vecs.T; self_idx = {n: i for i, n in enumerate(engine.label_names)}
    fam = np.stack(family_faces) if len(family_faces) else np.zeros((0, 128))
    piles = {k: [] for k, _, _ in PILES}
    for n, ph in enumerate(photos):
        rec = cache[ph["key"]]; F = [np.array(f) for f in rec["faces"]]
        ph["faces"] = len(F); ph["label"] = engine.label_names[int(S[n].argmax())]
        ph["family_score"] = round(max((float((fam @ f).max()) for f in F), default=0.0), 3) if len(fam) else 0.0
        if ph["family_score"] >= FACE_TH: ph["pile"] = "family"
        elif rec["accidental"]: ph["pile"] = "accidental"
        elif ph["label"] == "people": ph["pile"] = "people"
        elif ph["label"] == "forwards" and S[n][self_idx["forwards"]] - S[n][self_idx["people"]] <= FWD_MARGIN: ph["pile"] = "people"  # unsure: review, not junk
        else: ph["pile"] = ph["label"]
    # near-duplicates: in each group keep the biggest file, the others go to the Duplicates pile (family copies too)
    sim = E @ E.T; seen = set()
    for a in range(len(photos)):
        if a in seen: continue
        group = [b for b in range(len(photos)) if b != a and sim[a, b] >= DUP and b not in seen]
        if not group: continue
        group = [a] + group; best = max(group, key=lambda g: photos[g]["bytes"])
        for g in group:
            seen.add(g)
            if g != best: photos[g]["dup_of"] = photos[best]["id"]; photos[g]["pile"] = "duplicates"
    for ph in photos: piles[ph["pile"]].append(ph["id"])
    secs = time.time() - t0
    stats = {"photos": len(photos), "new": done_new, "seconds": round(secs, 1),
             "ms_per_new_photo": round(secs / done_new * 1000, 1) if done_new else None,
             "bytes": {k: sum(p["bytes"] for p in photos if p["pile"] == k) for k in piles}}
    return {"photos": photos, "piles": piles, "stats": stats, "E": E}
