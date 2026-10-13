"""EmbeddingGemma 2 for on-device photo cleanup: one run, every number in results/.

Dataset (data/): family/ (fictional American family, KEEP), other_family/ (two other fictional families, same Codex look, CLEAN: other
people's photos), other_people/ (real Flickr30k photos of strangers), junk/ (forwards, memes, screenshots, receipts,
accidental shots). Every image is first re-saved like WhatsApp does (long side 1600, JPEG q70) so all sources look alike.

Tests:
  1. zero-shot: text labels only, no training (what the app can do on day one)
  2. "these are my people": the user marks k family photos; nearest-neighbour vs the rest (k = 1, 3, 5, 10)
  3. near-duplicates: burst-like variants and re-forwards vs different photos
  4. text search: plain-language queries, hit in the top 5
  5. Matryoshka sizes 768 / 256 / 128 on tests 1 and 2
  6. speed and memory
Usage: .venv/bin/python bench/bench.py [mps|cpu]
"""
import io, json, random, sys, time
from pathlib import Path

import numpy as np, psutil, torch
from PIL import Image, ImageEnhance
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA, CACHE, OUT = ROOT / "data", ROOT / "data" / "_whatsapp", ROOT / "results"
DEV = sys.argv[1] if len(sys.argv) > 1 else "mps"
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 280          # image tokens: 280 = library default, 70 = Google's on-device budget
random.seed(0); np.random.seed(0)

def category(p: Path):
    top = p.relative_to(DATA).parts[0]
    if top in ("family", "other_family", "other_people"): return top
    n = p.stem
    for pre, cat in (("codex_greet", "greeting"), ("codex_meme", "meme"), ("codex_quote", "quote"), ("codex_cartoon", "meme")):
        if n.startswith(pre): return cat
    if p.parent.name in ("screenshot", "receipt", "accidental"): return p.parent.name
    return "forward"                                  # promo, news, chain message, invitation, stickers...

def whatsapp(p: Path):
    out = CACHE / (str(p.relative_to(DATA)).replace("/", "__") + ".jpg")
    if not out.exists() or out.stat().st_mtime < p.stat().st_mtime:      # rebuild when the source file changes
        im = Image.open(p).convert("RGB"); im.thumbnail((1600, 1600)); out.parent.mkdir(parents=True, exist_ok=True); im.save(out, quality=70)
    return out

files = sorted(p for p in DATA.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg") and "_whatsapp" not in p.parts
               and "refs" not in p.parts and "tmp_html" not in p.parts)
items = [{"path": str(p.relative_to(DATA)), "cat": category(p), "img": whatsapp(p)} for p in files]
cats = sorted({i["cat"] for i in items})
print(len(items), "images", {c: sum(i["cat"] == c for i in items) for c in cats})

t = time.time()
model = SentenceTransformer("google/embeddinggemma-2", device=DEV, config_kwargs={"audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32})
load_s = time.time() - t
_ip = model[0].processor; _ip.image_processor.max_soft_tokens = TOKENS; _ip.image_processor.image_seq_length = TOKENS; _ip.image_seq_length = TOKENS
def enc_imgs(paths, bs=8):
    return model.encode([{"image": Image.open(p).convert("RGB")} for p in paths], batch_size=bs, normalize_embeddings=True,
                        show_progress_bar=False)
def enc_txt(texts):
    return model.encode(texts, prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)

# ---- 6. speed (warm, after a warm-up batch) ----
enc_imgs([items[0]["img"]] * 8)
t = time.time(); E = enc_imgs([i["img"] for i in items]); embed_s = time.time() - t
speed = {"device": DEV, "image_tokens": TOKENS, "images": len(items), "ms_per_image": round(embed_s / len(items) * 1000, 1), "load_s": round(load_s, 1),
         "rss_gb": round(psutil.Process().memory_info().rss / 1e9, 2)}
print("speed", speed)

def trunc(X, d):
    Y = X[:, :d]; return Y / np.linalg.norm(Y, axis=1, keepdims=True)

# ---- 1. zero-shot ----
LABELS = {
    "people_photo": ["a candid photo of a family", "a personal photo of people", "a smartphone photo of kids and parents"],
    "greeting":     ["a good morning greeting card image with decorative text", "an Eid or Jumuah greeting card", "a birthday e-card"],
    "meme":         ["an internet meme with a caption", "a funny comic or reaction meme"],
    "quote":        ["a motivational or religious quote image"],
    "forward":      ["a forwarded promotional flyer or advertisement", "a viral infographic or chain message", "an invitation card"],
    "screenshot":   ["a screenshot of a phone screen", "a screenshot of a website or chat"],
    "receipt":      ["a photo of a shop receipt", "a photo of a printed document"],
    "accidental":   ["an accidental, blurry or black photo taken by mistake"],
}
GROUP = {"family": "people_photo", "other_family": "people_photo", "other_people": "people_photo"}
def zero_shot(X, d):
    names = list(LABELS); T = np.stack([trunc(enc_txt(LABELS[n]), d).mean(0) for n in names])
    T = T / np.linalg.norm(T, axis=1, keepdims=True); pred = (trunc(X, d) @ T.T).argmax(1)
    truth = [GROUP.get(i["cat"], i["cat"]) for i in items]
    acc = float(np.mean([names[p] == t for p, t in zip(pred, truth)]))
    # the decision the app really makes: real photo of people vs everything else
    is_photo = np.array([t == "people_photo" for t in truth]); said_photo = np.array([names[p] == "people_photo" for p in pred])
    tp = int((is_photo & said_photo).sum()); fp = int((~is_photo & said_photo).sum()); fn = int((is_photo & ~said_photo).sum())
    per = {c: round(float(np.mean([names[pred[k]] == GROUP.get(c, c) for k in range(len(items)) if items[k]["cat"] == c])), 3) for c in cats}
    wrong = [{"path": items[k]["path"], "truth": truth[k], "pred": names[pred[k]]} for k in range(len(items)) if names[pred[k]] != truth[k]]
    return {"dim": d, "accuracy_8way": round(acc, 3), "photo_kept_recall": round(tp / max(1, tp + fn), 3),
            "junk_wrongly_kept": fp, "photos_wrongly_flagged": fn, "per_category": per, "wrong": wrong}

# ---- 2. "these are my people" ----
def enrol(X, d, k, trials=20):
    Y = trunc(X, d); fam = [n for n, i in enumerate(items) if i["cat"] == "family"]
    rows = []
    for _ in range(trials):
        shots = random.sample(fam, k); rest = [n for n in range(len(items)) if n not in shots]
        s = (Y[rest] @ Y[shots].T).max(1); y = np.array([items[n]["cat"] == "family" for n in rest])
        pos, neg = y.sum(), (~y).sum(); ranks = np.argsort(np.argsort(s)); auc = float((ranks[y].sum() - pos * (pos - 1) / 2) / (pos * neg))
        best = max(((float(np.mean((s >= th) == y)), th) for th in np.unique(s)))
        hard = [n for n in range(len(rest)) if items[rest[n]]["cat"] == "other_family"]
        th = best[1]; rows.append({"auc": auc, "acc": best[0], "other_family_wrongly_kept": float(np.mean(s[hard] >= th)) if hard else None,
                                   "family_missed": float(np.mean(s[y] < th))})
    agg = lambda key: round(float(np.mean([r[key] for r in rows if r[key] is not None])), 3) if any(r[key] is not None for r in rows) else None
    return {"dim": d, "k": k, "auc": agg("auc"), "best_accuracy": agg("acc"), "family_missed": agg("family_missed"),
            "other_family_wrongly_kept": agg("other_family_wrongly_kept"), "trials": trials}

# ---- 3. near-duplicates ----
def variants(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; out = []
    c = random.uniform(0.03, 0.08); out.append(im.crop((int(w*c), int(h*c), int(w*(1-c/2)), int(h*(1-c/2)))))   # burst: slight reframe
    out.append(ImageEnhance.Brightness(im).enhance(random.uniform(0.8, 1.2)))                                    # exposure change
    b = io.BytesIO(); small = im.copy(); small.thumbnail((800, 800)); small.save(b, "JPEG", quality=40); out.append(Image.open(b).convert("RGB"))  # re-forward
    return out
def dedupe():
    base = random.sample([i for i in items if i["cat"] in ("family", "other_people", "other_family")], min(20, len(items)))
    A = model.encode([{"image": Image.open(i["img"]).convert("RGB")} for i in base], normalize_embeddings=True, show_progress_bar=False)
    pos = []
    for n, i in enumerate(base):
        V = model.encode([{"image": v} for v in variants(i["img"])], normalize_embeddings=True, show_progress_bar=False); pos += list(V @ A[n])
    idx = {i["path"]: k for k, i in enumerate(items)}; B = E[[idx[i["path"]] for i in base]]
    S = B @ B.T; neg = S[np.triu_indices(len(base), 1)]                      # different photos (incl. same family, same day look)
    th = float(np.percentile(pos, 5))                                       # threshold that keeps 95% of true duplicates
    return {"duplicate_cos_min": round(float(min(pos)), 3), "duplicate_cos_median": round(float(np.median(pos)), 3),
            "different_cos_max": round(float(neg.max()), 3), "different_cos_median": round(float(np.median(neg)), 3),
            "threshold_95pct_recall": round(th, 3), "false_pairs_at_threshold": int((neg >= th).sum()), "pairs": int(len(neg))}

# ---- 4. text search ----
QUERIES = {"a birthday cake": "fam_bday", "receipt from a shop": "receipt/", "kids at the beach": "fam_beach", "a meme": "codex_meme",
           "good morning": "codex_greet_morning", "snow": "fam_snow", "a website screenshot": "screenshot/web", "a chat conversation": "screenshot/chat",
           "grandmother pouring tea": "fam_tea", "prayer times": "codex_prayer"}
def search():
    out = {}
    for q, want in QUERIES.items():
        s = E @ enc_txt([q])[0]; top = [items[n]["path"] for n in np.argsort(-s)[:5]]
        out[q] = {"hit_top1": want in top[0], "hit_top5": any(want in t for t in top), "top5": top}
    return {"top1": sum(v["hit_top1"] for v in out.values()), "top5": sum(v["hit_top5"] for v in out.values()), "of": len(out), "queries": out}

res = {"model": "google/embeddinggemma-2 (text+vision, audio not loaded)", "date": time.strftime("%Y-%m-%d %H:%M"),
       "counts": {c: sum(i["cat"] == c for i in items) for c in cats}, "speed": speed,
       "zero_shot": [zero_shot(E, d) for d in (768, 256, 128)],
       "my_people": [enrol(E, d, k) for d in (768, 128) for k in (1, 3, 5, 10) if k <= sum(i["cat"] == "family" for i in items) - 2],
       "dedupe": dedupe(), "search": search(),
       "storage_bytes_per_photo": {"768_float32": 768 * 4, "128_int8": 128}}
OUT.mkdir(exist_ok=True); json.dump(res, open(OUT / f"bench_{DEV}_{TOKENS}tok.json", "w"), indent=1)
for z in res["zero_shot"]: print("zero-shot", {k: v for k, v in z.items() if k not in ("wrong", "per_category")})
print("per category (768)", res["zero_shot"][0]["per_category"])
for m in res["my_people"]: print("my people", m)
print("dedupe", res["dedupe"]); print("search", res["search"]["top1"], res["search"]["top5"], "/", res["search"]["of"])
