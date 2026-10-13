"""Unsupervised clustering + one cleanup score for EmbeddingGemma 2 (same WhatsApp-style copies as bench.py).
Clusters are scored against the true categories: ARI and NMI (1.0 = perfect, 0 = random) and purity (share of photos whose
cluster's majority category is their own). Usage: .venv/bin/python bench/cluster.py [tokens]"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT / "data/_whatsapp"; TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 70
def cat(stem):
    top, name = stem.split("__")[0], stem.split("__")[-1]
    if top in ("family", "other_family", "other_people"): return top
    if top == "junk" and "__" in stem and stem.split("__")[1] in ("screenshot", "receipt", "accidental"): return stem.split("__")[1]
    return "forward"                                   # greetings, memes, quotes, flyers, chain messages
files = sorted(CACHE.glob("*.jpg")); y = [cat(p.stem) for p in files]
m = SentenceTransformer("google/embeddinggemma-2", device="mps", config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
ip = m[0].processor; ip.image_processor.max_soft_tokens = ip.image_processor.image_seq_length = ip.image_seq_length = TOK
E = m.encode([{"image": Image.open(p).convert("RGB")} for p in files], batch_size=8, normalize_embeddings=True, show_progress_bar=False)
np.save(ROOT / f"results/emb_{TOK}tok.npy", E)
labels = sorted(set(y)); Y = np.array([labels.index(c) for c in y]); K = len(labels)
def score(pred):
    pur = sum(np.bincount(Y[pred == c]).max() for c in set(pred)) / len(Y)
    fam = Y == labels.index("family"); best = np.bincount(pred[fam]).argmax()
    return {"ARI": round(adjusted_rand_score(Y, pred), 3), "NMI": round(normalized_mutual_info_score(Y, pred), 3), "purity": round(pur, 3),
            "family_in_one_cluster": round(float((pred[fam] == best).mean()), 3), "that_cluster_is_family": round(float(fam[pred == best].mean()), 3)}
out = {"tokens": TOK, "n": len(Y), "categories": {c: y.count(c) for c in labels}}
for d in (768, 256):
    X = E[:, :d] / np.linalg.norm(E[:, :d], axis=1, keepdims=True)
    out[f"kmeans_{d}"] = score(KMeans(K, n_init=20, random_state=0).fit_predict(X))
    out[f"agglo_{d}"] = score(AgglomerativeClustering(K, metric="cosine", linkage="average").fit_predict(X))
print(json.dumps(out, indent=1)); json.dump(out, open(ROOT / f"results/cluster_{TOK}tok.json", "w"), indent=1)
