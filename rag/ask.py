"""Ask your files: index a folder (PDF, Word, text, photos, screenshots, installers) with EmbeddingGemma 2 and search it by
meaning. Text and images share one embedding space, so "the Thanksgiving photo" finds a JPEG and "where's my latest electricity bill" finds a
PDF. Everything stays on this machine.

Index unit: a document chunk ("title: <file name> | text: <chunk>", the model's Document prompt), a photo (70 image tokens), or
just the file name for files without readable text. A file's score = its best unit. Queries use the SearchQuery prompt.
"latest / newest / most recent / last" in a question: among files within 0.05 of the best score, the newest wins.
Usage:  .venv/bin/python rag/ask.py <folder> "where's my latest electricity bill?"
"""
import json, re, sys, time
from pathlib import Path

import numpy as np, torch
from PIL import Image, ImageOps

IMG = {".jpg", ".jpeg", ".png", ".webp", ".heic"}; DOC = {".pdf", ".docx", ".txt", ".md"}
CHUNK, TOKENS, NEAR = 700, 70, 0.05
RECENT = re.compile(r"\b(latest|newest|most recent|last|recent)\b", re.I)


def load_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("google/embeddinggemma-2", device="mps" if torch.backends.mps.is_available() else "cpu",
                            config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
    ip = m[0].processor; ip.image_processor.max_soft_tokens = ip.image_processor.image_seq_length = ip.image_seq_length = TOKENS
    return m


def text_of(p: Path):
    try:
        if p.suffix == ".pdf":
            import pypdf; return "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(p).pages)
        if p.suffix == ".docx":
            import docx; return "\n".join(par.text for par in docx.Document(p).paragraphs)
        return p.read_text(errors="ignore")
    except Exception:
        return ""


def chunks(t):
    t = re.sub(r"\s+", " ", t).strip()
    return [t[i:i + CHUNK] for i in range(0, max(len(t), 1), CHUNK - 100)] if t else []


class Index:
    def __init__(self, model, folder: Path):
        self.m, self.folder = model, folder; self.files, self.rows, V = [], [], []
        t0 = time.time(); imgs = []
        for p in sorted(x for x in folder.iterdir() if x.is_file() and not x.name.startswith(".")):
            f = len(self.files); self.files.append({"name": p.name, "mtime": p.stat().st_mtime, "kind": "other"})
            ext = p.suffix.lower()
            if ext in IMG:
                self.files[f]["kind"] = "image"; im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im.thumbnail((1600, 1600)); imgs.append((f, im))
                continue
            parts = chunks(text_of(p)) if ext in DOC else []
            if parts: self.files[f]["kind"] = "document"
            for c in (parts or ["file"]):
                self.rows.append(f); V.append(f"title: {p.name} | text: {c}")
        TV = self.m.encode(V, prompt_name=None, normalize_embeddings=True, batch_size=16, show_progress_bar=False) if V else np.zeros((0, 768))
        IV = self.m.encode([{"image": im} for _, im in imgs], batch_size=8, normalize_embeddings=True, show_progress_bar=False) if imgs else np.zeros((0, 768))
        self.rows += [f for f, _ in imgs]; self.E = np.vstack([TV, IV]).astype(np.float32)
        self.index_s = time.time() - t0

    def search(self, q, k=5):
        t = time.time(); v = self.m.encode([q], prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)[0]
        s = self.E @ v; best = {}
        for r, sc in zip(self.rows, s): best[r] = max(best.get(r, -1), float(sc))
        order = sorted(best, key=lambda f: -best[f])
        if RECENT.search(q):   # "latest": among near-ties with the best match, newest first
            top = best[order[0]]; near = [f for f in order if best[f] >= top - NEAR]
            order = sorted(near, key=lambda f: -self.files[f]["mtime"]) + [f for f in order if f not in near]
        return [{"name": self.files[f]["name"], "score": round(best[f], 3), "kind": self.files[f]["kind"]} for f in order[:k]], (time.time() - t) * 1000


if __name__ == "__main__":
    idx = Index(load_model(), Path(sys.argv[1]))
    print(f"indexed {len(idx.files)} files ({len(idx.E)} vectors) in {idx.index_s:.1f} s")
    for q in sys.argv[2:]:
        hits, ms = idx.search(q); print(f"\n{q}  ({ms:.0f} ms)"); [print(f"  {h['score']:.3f}  {h['name']}") for h in hits]
