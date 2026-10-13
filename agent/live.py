"""Live index: a local agent's memory of your Mac. Watches folders, re-embeds only what changed, keeps the vectors on disk.

Two levels, so it stays fast on a whole disk:
  repo cards   one unit per git repo: its name, README start, manifest description and top-level files ("which repo does X?")
  documents    the rag/ask.py units (700-character chunks, photos at 70 image tokens) for folders you opt in to (deep)
Never read: secrets (.env*, keys, credentials, ~/.ssh), dependencies and build output, files over 2 MB.
Store: one SQLite file (metadata + float32 vectors as blobs); search runs on a numpy matrix in memory, the fastest exact
store in rag/stores.py. A saved, created, moved or deleted file is re-indexed about a second later (FSEvents via watchdog).

    .venv/bin/python agent/live.py --repos ~/Projects --deep ~/Downloads "where's my latest electricity bill?"
    .venv/bin/python agent/live.py --deep demo/Downloads --watch          # stays running, prints every re-index
"""
import argparse, fnmatch, json, re, sqlite3, sys, threading, time
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag")); import ask

STORE = Path.home() / ".photosweep" / "live.sqlite"
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "dist", "build", ".next", "DerivedData", "Pods", "target", ".cache",
             ".ssh", ".gnupg", "Library", ".Trash"}
SECRETS = [".env", ".env.*", "*.pem", "*.p8", "*.p12", "*.key", "id_rsa*", "id_ed25519*", "*credential*", "*secret*", "*.keychain*",
           "*.sqlite", "*.db", "*token*"]
MANIFESTS = ("package.json", "pyproject.toml", "Cargo.toml", "Package.swift", "go.mod", "setup.py")
MAX_BYTES = 2_000_000


def private(p: Path):
    """True for anything the agent must never read."""
    return any(part in SKIP_DIRS for part in p.parts) or any(fnmatch.fnmatch(p.name.lower(), g) for g in SECRETS)


def repo_card(repo: Path):
    """One short text per repo: what a person would remember about it."""
    readme = next((repo / n for n in ("README.md", "readme.md", "README.txt", "README") if (repo / n).is_file()), None)
    desc = []
    for m in MANIFESTS:
        f = repo / m
        if not f.is_file(): continue
        t = f.read_text(errors="ignore")[:4000]
        if m == "package.json":
            try: d = json.loads(t); desc.append(" ".join(str(d.get(k, "")) for k in ("name", "description", "keywords")))
            except json.JSONDecodeError: pass
        else: desc += re.findall(r'(?:description|name)\s*[=:]\s*"([^"]+)"', t)[:3]
    top = sorted(x.name for x in repo.iterdir() if not x.name.startswith(".") and not private(x))[:40]
    body = " ".join(filter(None, [(readme.read_text(errors="ignore")[:1500] if readme else ""), " ".join(desc), "files: " + " ".join(top)]))
    return f"title: {repo.name} | text: {re.sub(r'\s+', ' ', body)}"


class LiveIndex:
    def __init__(self, model, store=STORE):
        self.m = model; store.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(store, check_same_thread=False); self.lock = threading.RLock()
        self.db.execute("create table if not exists units (path text, kind text, mtime real, vec blob)")
        self.db.execute("create index if not exists units_path on units(path)"); self._load()

    def _load(self):
        rows = self.db.execute("select rowid, path, kind, mtime, vec from units").fetchall()
        self.ids = [r[0] for r in rows]; self.meta = [(r[1], r[2], r[3]) for r in rows]
        self.E = np.vstack([np.frombuffer(r[4], np.float32) for r in rows]) if rows else np.zeros((0, 768), np.float32)

    def _put(self, path: Path, kind, texts=(), images=()):
        V = []
        if texts: V.append(self.m.encode(list(texts), normalize_embeddings=True, batch_size=16, show_progress_bar=False))
        if images: V.append(self.m.encode([{"image": im} for im in images], normalize_embeddings=True, batch_size=8, show_progress_bar=False))
        mt = path.stat().st_mtime
        with self.lock:
            self.db.execute("delete from units where path = ?", (str(path),))
            for v in (np.vstack(V) if V else []):
                self.db.execute("insert into units values (?, ?, ?, ?)", (str(path), kind, mt, np.asarray(v, np.float32).tobytes()))
            self.db.commit(); self._load()

    def forget(self, path: Path):
        with self.lock:
            self.db.execute("delete from units where path = ? or path like ?", (str(path), str(path) + "/%")); self.db.commit(); self._load()

    def fresh(self, path: Path):
        r = self.db.execute("select mtime from units where path = ? limit 1", (str(path),)).fetchone()
        return r is not None and abs(r[0] - path.stat().st_mtime) < 1e-3

    def add_file(self, p: Path):
        """Index one document or photo (same units as rag/ask.py). Returns True if it was (re)indexed."""
        if not p.is_file() or private(p) or p.name.startswith(".") or p.stat().st_size > MAX_BYTES or self.fresh(p): return False
        ext = p.suffix.lower()
        if ext in ask.IMG:
            from PIL import Image, ImageOps
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im.thumbnail((1600, 1600)); self._put(p, "image", images=[im])
        else:
            parts = ask.chunks(ask.text_of(p)) if ext in ask.DOC else []
            self._put(p, "document" if parts else "other", texts=[f"title: {p.name} | text: {c}" for c in (parts or ["file"])])
        return True

    def add_repo(self, repo: Path):
        if self.fresh(repo): return False
        self._put(repo, "repo", texts=[repo_card(repo)]); return True

    def scan(self, repos=(), deep=()):
        t, n = time.time(), 0
        for root in repos:
            for g in sorted(Path(root).expanduser().glob("**/.git")):
                if g.is_dir() and len(g.relative_to(Path(root).expanduser()).parts) <= 3 and not private(g.parent): n += self.add_repo(g.parent)
        for root in deep:
            for p in sorted(Path(root).expanduser().rglob("*")):
                if p.is_file() and not private(p): n += self.add_file(p)
        return n, time.time() - t

    def search(self, q, k=5, kind=None):
        t = time.time(); v = self.m.encode([q], prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)[0]
        with self.lock: s, meta = self.E @ v, list(self.meta)
        best = {}
        for (path, kd, mt), sc in zip(meta, s):
            if kind and kd != kind: continue
            if sc > best.get(path, (-9,))[0]: best[path] = (float(sc), kd, mt)
        order = sorted(best, key=lambda p: -best[p][0])
        if ask.RECENT.search(q) and order:
            top = best[order[0]][0]; near = [p for p in order if best[p][0] >= top - ask.NEAR]
            order = sorted(near, key=lambda p: -best[p][2]) + [p for p in order if p not in near]
        return [{"path": p, "score": round(best[p][0], 3), "kind": best[p][1]} for p in order[:k]], (time.time() - t) * 1000

    def watch(self, roots, on_change=None):
        """Re-index a file (or its repo card) about a second after it is saved; forget it when deleted."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
        pending, idx = {}, self

        class H(FileSystemEventHandler):
            def on_any_event(self, e):
                for p in filter(None, (getattr(e, "src_path", None), getattr(e, "dest_path", None))):
                    p = Path(p)
                    if not private(p): pending[p] = time.time()

        def worker():
            while True:
                time.sleep(.3)
                for p, t0 in list(pending.items()):
                    if time.time() - t0 < 1.0: continue                       # wait until the save has settled
                    pending.pop(p, None); t = time.time()
                    repo = next((r for r in [p.parent, *p.parents] if (r / ".git").is_dir()), None)
                    if not p.exists(): idx.forget(p); what = "forgot"
                    elif repo and p.name in ("README.md",) + MANIFESTS: idx.add_repo(repo); what = "repo card"
                    elif p.is_file() and idx.add_file(p): what = "indexed"
                    else: continue
                    if on_change: on_change(what, p, time.time() - t)

        obs = Observer()
        for r in roots: obs.schedule(H(), str(Path(r).expanduser()), recursive=True)
        obs.start(); threading.Thread(target=worker, daemon=True).start(); return obs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repos", nargs="*", default=[]); ap.add_argument("--deep", nargs="*", default=[])
    ap.add_argument("--store", default=str(STORE)); ap.add_argument("--watch", action="store_true"); ap.add_argument("q", nargs="*")
    a = ap.parse_args(); idx = LiveIndex(ask.load_model(), Path(a.store))
    n, s = idx.scan(a.repos, a.deep); print(f"{len(idx.meta)} units in the index, {n} (re)indexed in {s:.1f} s")
    for q in a.q:
        hits, ms = idx.search(q); print(f"\n{q}  ({ms:.0f} ms)"); [print(f"  {h['score']:.3f}  {h['kind']:8s} {h['path']}") for h in hits]
    if a.watch:
        idx.watch(a.repos + a.deep, lambda what, p, s: print(f"{time.strftime('%H:%M:%S')}  {what:9s} {p.name}  ({s:.2f} s)", flush=True))
        print("watching, Ctrl-C to stop"); threading.Event().wait()
