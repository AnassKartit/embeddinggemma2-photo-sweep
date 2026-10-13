"""'Tag a few photos': logistic regression on EmbeddingGemma 2 vectors (results/emb_70tok.npy from bench/cluster.py 70).
50 random draws of k junk + k keep photos, scored on the rest. Usage: .venv/bin/python decide/tagged.py"""
import json, random
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parent.parent
E = np.load(ROOT / "results/emb_70tok.npy"); files = sorted((ROOT / "data/_whatsapp").glob("*.jpg")); assert len(files) == len(E)
y = np.array([0 if f.stem.split("__")[0] in ("family", "other_family", "other_people") else 1 for f in files]); res = {}
for d in (768, 256):
    X = E[:, :d] / np.linalg.norm(E[:, :d], axis=1, keepdims=True)
    for k in (2, 5, 10, 20):
        accs, lost, missed = [], [], []
        for t in range(50):
            rng = random.Random(t); tr = rng.sample(list(np.where(y == 1)[0]), k) + rng.sample(list(np.where(y == 0)[0]), k)
            te = np.array([i for i in range(len(y)) if i not in tr]); p = LogisticRegression(C=10, max_iter=2000).fit(X[tr], y[tr]).predict(X[te])
            accs.append((p == y[te]).mean()); lost.append(((p == 1) & (y[te] == 0)).sum() / (y[te] == 0).sum()); missed.append(((p == 0) & (y[te] == 1)).sum() / (y[te] == 1).sum())
        res[f"{d}d_k{k}"] = {"tagged": 2 * k, "accuracy": round(float(np.mean(accs)), 3), "people_photos_lost": round(float(np.mean(lost)), 3), "junk_missed": round(float(np.mean(missed)), 3)}
        print(d, "dims,", 2 * k, "tagged:", res[f"{d}d_k{k}"])
json.dump(res, open(ROOT / "decide/tagged.json", "w"), indent=1)
