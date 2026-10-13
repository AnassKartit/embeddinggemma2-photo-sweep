"""Do you need a vector database? Same EmbeddingGemma 2 vectors, same 30 questions (rag/eval.py), four places to keep them:
  numpy      the vectors in memory, compared with every question (what the app does)
  chroma     ChromaDB 1.5.9, PersistentClient on disk, cosine HNSW index
  sqlite-vec sqlite-vec 0.1.6, one SQLite file, exact cosine search
  pgvector   PostgreSQL 17 + pgvector, local server, exact search and an HNSW index
Each store returns its top 20 vectors; files get the score of their best vector and "latest" re-sorts near ties by date
(same rules as rag/ask.py), so only the store differs. Timed: the store's search only (the question is embedded once, before).
Part 2: 50,000 random 768-number vectors grouped around 300 centres (embedding-like), insert time and query time per store.
Usage: .venv/bin/python rag/stores.py   (Postgres: brew install postgresql@17 pgvector; a throwaway cluster is created and removed)
"""
import json, os, shutil, statistics as st, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import ask, eval as ev

ROOT = Path(__file__).resolve().parent.parent; TMP = Path(tempfile.mkdtemp(prefix="stores_")); K = 20
PG_BIN = Path("/opt/homebrew/opt/postgresql@17/bin"); PG_PORT = 54329

def rank(idx, q, ids, scores, k=5):
    """ids/scores of the top vectors -> files, same rules as ask.Index.search."""
    best = {}
    for i, s in zip(ids, scores):
        f = idx.rows[int(i)]; best[f] = max(best.get(f, -9), float(s))
    order = sorted(best, key=lambda f: -best[f])
    if ask.RECENT.search(q) and order:
        top = best[order[0]]; near = [f for f in order if best[f] >= top - ask.NEAR]
        order = sorted(near, key=lambda f: -idx.files[f]["mtime"]) + [f for f in order if f not in near]
    return [idx.files[f]["name"] for f in order[:k]]

class Numpy:
    name = "numpy (in memory)"
    def __init__(self, E): self.E = E
    def search(self, v): s = self.E @ v; top = np.argsort(-s)[:K]; return top, s[top]

class Chroma:
    name = "ChromaDB"
    def __init__(self, E):
        import chromadb
        self.c = chromadb.PersistentClient(path=str(TMP / "chroma")).create_collection(f"files{len(E)}", metadata={"hnsw:space": "cosine"})
        for a in range(0, len(E), 5000): self.c.add(ids=[str(i) for i in range(a, min(a + 5000, len(E)))], embeddings=E[a:a + 5000].tolist())
    def search(self, v):
        r = self.c.query(query_embeddings=[v.tolist()], n_results=K)
        return [int(x) for x in r["ids"][0]], [1 - d for d in r["distances"][0]]

class SqliteVec:
    name = "sqlite-vec"
    def __init__(self, E):
        import sqlite3, sqlite_vec
        self.db = sqlite3.connect(TMP / f"vec{len(E)}.db"); self.db.enable_load_extension(True); sqlite_vec.load(self.db)
        self.db.execute(f"create virtual table v using vec0(emb float[{E.shape[1]}] distance_metric=cosine)")
        self.db.executemany("insert into v(rowid, emb) values (?, ?)", ((i, e.astype(np.float32).tobytes()) for i, e in enumerate(E))); self.db.commit()
    def search(self, v):
        r = self.db.execute("select rowid, distance from v where emb match ? and k = ?", (v.astype(np.float32).tobytes(), K)).fetchall()
        return [x[0] for x in r], [1 - x[1] for x in r]

class Pg:
    def __init__(self, E, hnsw=False):
        import psycopg
        self.name = "Postgres + pgvector" + (" (HNSW index)" if hnsw else " (exact)")
        self.cx = psycopg.connect(f"host=127.0.0.1 port={PG_PORT} dbname=postgres user=postgres", autocommit=True)
        self.t = f"v{len(E)}_{'hnsw' if hnsw else 'exact'}"            # one table per store, so the HNSW one never replaces the exact one
        self.cx.execute("create extension if not exists vector"); self.cx.execute(f"drop table if exists {self.t}")
        self.cx.execute(f"create table {self.t} (id int primary key, emb vector({E.shape[1]}))")
        with self.cx.cursor().copy(f"copy {self.t} (id, emb) from stdin") as cp:
            for i, e in enumerate(E): cp.write_row((i, "[" + ",".join(f"{x:.6f}" for x in e) + "]"))
        if hnsw: self.cx.execute(f"create index on {self.t} using hnsw (emb vector_cosine_ops)")
        self.cx.execute(f"analyze {self.t}")
    def search(self, v):
        r = self.cx.execute(f"select id, 1 - (emb <=> %s::vector) from {self.t} order by emb <=> %s::vector limit %s", ("[" + ",".join(f"{x:.6f}" for x in v) + "]",) * 2 + (K,)).fetchall()
        return [x[0] for x in r], [x[1] for x in r]

def pg_up():
    if not (PG_BIN / "initdb").exists(): return False
    d = TMP / "pg"; subprocess.run([PG_BIN / "initdb", "-D", d, "-U", "postgres", "--auth=trust"], check=True, capture_output=True)
    subprocess.run([PG_BIN / "pg_ctl", "-D", d, "-o", f"-p {PG_PORT} -k {TMP}", "-l", TMP / "pg.log", "-w", "start"], check=True, capture_output=True)
    return True

def pg_down(): subprocess.run([PG_BIN / "pg_ctl", "-D", TMP / "pg", "-m", "fast", "stop"], capture_output=True)

def build(E, kinds):
    out = []
    for k in kinds:
        t = time.time(); s = k(E); out.append((s, round(time.time() - t, 2)))
    return out

if __name__ == "__main__":
    has_pg = pg_up(); res = {"questions": len(ev.Q), "has_postgres": has_pg}
    try:
        m = ask.load_model(); idx = ask.Index(m, ev.FOLDER)
        QV = m.encode([q for q, _ in ev.Q], prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)
        kinds = [Numpy, Chroma, SqliteVec] + ([lambda E: Pg(E), lambda E: Pg(E, hnsw=True)] if has_pg else [])
        res["real"] = []
        for store, ins in build(idx.E, kinds):
            hits, ms = 0, []
            for (q, want), v in zip(ev.Q, QV):
                t = time.perf_counter(); ids, sc = store.search(v); ms.append((time.perf_counter() - t) * 1000)
                hits += rank(idx, q, ids, sc)[0] in want
            res["real"].append({"store": store.name, "right_file_first": hits, "median_ms": round(st.median(ms), 2), "build_s": ins})
            print("real", res["real"][-1], flush=True)
        # part 2: 50,000 embedding-like vectors (300 centres + noise), 768 numbers each
        rng = np.random.default_rng(0); C = rng.normal(size=(300, 768)); X = C[rng.integers(0, 300, 50000)] + 0.35 * rng.normal(size=(50000, 768))
        X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32); Q = X[rng.integers(0, 50000, 50)] + 0.05 * rng.normal(size=(50, 768)).astype(np.float32)
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True); truth = [set(np.argsort(-(X @ q))[:10]) for q in Q]
        res["scale_50k"] = []
        for store, ins in build(X, kinds):
            ms, rec = [], []
            for q, tr in zip(Q, truth):
                t = time.perf_counter(); ids, _ = store.search(q); ms.append((time.perf_counter() - t) * 1000); rec.append(len(set(int(i) for i in ids[:10]) & tr) / 10)
            res["scale_50k"].append({"store": store.name, "median_ms": round(st.median(ms), 2), "recall_at_10": round(float(np.mean(rec)), 3), "build_s": ins})
            print("50k ", res["scale_50k"][-1], flush=True)
    finally:
        if has_pg: pg_down()
        shutil.rmtree(TMP, ignore_errors=True)
    json.dump(res, open(ROOT / "results/stores.json", "w"), indent=1)
