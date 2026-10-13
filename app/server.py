"""Photo Sweep: local web app. Run: .venv/bin/python app/server.py [folder]  then open http://127.0.0.1:8765
Binds to 127.0.0.1 only. Photos never leave the machine; "Sweep" moves files to <folder>/_Sweep/<pile>/ and Undo puts them back."""
import io, json, shutil, sys, threading, time
from pathlib import Path

import numpy as np, uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as eng

HERE = Path(__file__).resolve().parent
app = FastAPI()
S = {"engine": None, "folder": Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else HERE.parent / "demo" / "Phone",
     "family_faces": [], "family_photo": None, "result": None, "progress": {"done": 0, "total": 0, "seconds": 0, "state": "idle"},
     "moved": [], "lock": threading.Lock()}

ENGINE_LOCK = threading.Lock()
def engine():
    with ENGINE_LOCK:                                   # warm-up thread and first scan must not load the model twice
        if S["engine"] is None: S["engine"] = eng.Engine()
    return S["engine"]

@app.get("/", response_class=HTMLResponse)
def index(): return (HERE / "static/index.html").read_text()

@app.get("/api/state")
def state():
    r = S["result"]
    return {"folder": str(S["folder"]).replace(str(Path.home()), "~", 1), "family_set": bool(S["family_faces"]), "family_faces": len(S["family_faces"]),
            "progress": S["progress"], "moved": len(S["moved"]),
            "result": None if r is None else {"photos": r["photos"], "piles": r["piles"], "stats": r["stats"],
                                              "piles_meta": [{"key": k, "title": t, "kind": kd} for k, t, kd in eng.PILES]}}

@app.post("/api/folder")
def set_folder(body: dict):
    p = Path(body["folder"]).expanduser()
    if not p.is_dir(): raise HTTPException(400, f"not a folder: {p}")
    S["folder"], S["result"] = p, None; return {"ok": True}

@app.post("/api/family")
async def family(file: UploadFile = File(...)):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(await file.read()))).convert("RGB"); im.thumbnail((1600, 1600))
    faces = engine().faces(im)
    if not faces: raise HTTPException(400, "no face found in that photo")
    S["family_faces"] = faces; b = io.BytesIO(); im.thumbnail((512, 512)); im.save(b, "JPEG", quality=85); S["family_photo"] = b.getvalue()
    return {"faces": len(faces)}

@app.post("/api/family_from/{pid}")
def family_from(pid: str):
    ph = next((p for p in (S["result"] or {}).get("photos", []) if p["id"] == pid), None)
    if not ph: raise HTTPException(404)
    im = eng.load(Path(ph["path"])); faces = engine().faces(im)
    if not faces: raise HTTPException(400, "no face found in that photo")
    S["family_faces"] = faces; b = io.BytesIO(); im.thumbnail((512, 512)); im.save(b, "JPEG", quality=85); S["family_photo"] = b.getvalue()
    return {"faces": len(faces)}

@app.get("/api/family.jpg")
def family_jpg():
    if not S["family_photo"]: raise HTTPException(404)
    return Response(S["family_photo"], media_type="image/jpeg")

def run_scan():
    with S["lock"]:
        S["progress"] = {"done": 0, "total": 0, "seconds": 0, "state": "loading models"}
        e = engine(); S["progress"]["state"] = "scanning"
        def prog(d, t, s): S["progress"].update(done=d, total=t, seconds=round(s, 1))
        S["result"] = eng.scan(e, S["folder"], S["family_faces"], prog)
        S["progress"]["state"] = "done"

@app.post("/api/scan")
def scan():
    if S["lock"].locked(): return {"ok": False, "busy": True}
    threading.Thread(target=run_scan, daemon=True).start(); return {"ok": True}

THUMBS = {}
@app.get("/api/thumb/{pid}")
def thumb(pid: str):
    if pid in THUMBS: return Response(THUMBS[pid], media_type="image/jpeg")
    ph = next((p for p in (S["result"] or {}).get("photos", []) if p["id"] == pid), None)
    if not ph or not Path(ph["path"]).exists(): raise HTTPException(404)
    im = ImageOps.exif_transpose(Image.open(ph["path"])).convert("RGB"); im.thumbnail((360, 360)); b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    THUMBS[pid] = b.getvalue(); return Response(THUMBS[pid], media_type="image/jpeg")

@app.get("/api/photo/{pid}")
def photo(pid: str):
    ph = next((p for p in (S["result"] or {}).get("photos", []) if p["id"] == pid), None)
    if not ph: raise HTTPException(404)
    return FileResponse(ph["path"])

@app.get("/api/search")
def search(q: str, k: int = 12):
    r = S["result"]
    if r is None or not len(r["E"]): return {"hits": []}
    t = time.time(); v = engine().text_vec(q); s = r["E"] @ v; top = np.argsort(-s)[:k]
    return {"ms": round((time.time() - t) * 1000, 1), "hits": [{"id": r["photos"][n]["id"], "score": round(float(s[n]), 3)} for n in top]}

@app.post("/api/sweep")
def sweep(body: dict):
    r = S["result"]; ids = set(body["ids"]); moved = 0; freed = 0
    for ph in r["photos"]:
        if ph["id"] in ids and not ph.get("moved"):
            src = Path(ph["path"]); dst = S["folder"] / "_Sweep" / ph["pile"] / src.name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst)); S["moved"].append((str(dst), str(src))); ph["moved"] = True; ph["path"] = str(dst)
            moved += 1; freed += ph["bytes"]
    return {"moved": moved, "freed_bytes": freed}

@app.post("/api/undo")
def undo():
    n = 0
    while S["moved"]:
        dst, src = S["moved"].pop(); shutil.move(dst, src); n += 1
        for ph in S["result"]["photos"]:
            if ph["path"] == dst: ph["path"], ph["moved"] = src, False
    return {"restored": n}

# ── Ask your files: EmbeddingGemma 2 finds the files, Gemma 4 E4B reads them and answers (rag/ask.py, rag/answer.py) ──
sys.path.insert(0, str(HERE.parent / "rag"))
import ask as rag_ask, answer as rag_answer
LLM_REPO = "mlx-community/gemma-4-E4B-it-qat-4bit"          # 10/10 on rag/answer.py, 6.3 GB peak
A = {"folder": HERE.parent / "demo" / "Downloads", "index": None, "llm": None, "state": "idle", "lock": threading.Lock()}

def build_index():
    with A["lock"]:
        A["state"] = "indexing"; A["index"] = rag_ask.Index(engine().model, A["folder"]); A["state"] = "ready"

@app.post("/api/ask/index")
def ask_index(body: dict = None):
    if body and body.get("folder"):
        p = Path(body["folder"]).expanduser()
        if not p.is_dir(): raise HTTPException(400, "not a folder")
        A["folder"] = p
    if not A["lock"].locked(): threading.Thread(target=build_index, daemon=True).start()
    return {"ok": True}

@app.get("/api/ask/state")
def ask_state():
    i = A["index"]
    home = str(Path.home())                                   # show ~/... so a screen recording never shows the account name
    return {"folder": str(A["folder"]).replace(home, "~", 1), "state": A["state"], "files": len(i.files) if i else 0, "index_s": round(i.index_s, 1) if i else None}

@app.get("/api/ask")
def ask_q(q: str, answer: int = 1):
    i = A["index"]
    if i is None: raise HTTPException(400, "index the folder first")
    hits, ms = i.search(q, k=6); out = {"q": q, "search_ms": round(ms, 1), "hits": hits}
    if answer and rag_answer.photo_answer(hits):
        out["answer"], out["answer_s"] = rag_answer.photo_answer(hits), 0.0
    elif answer:
        from mlx_lm import load, generate
        if A["llm"] is None: A["llm"] = load(LLM_REPO)
        model, tok = A["llm"]; ctx = []
        for h in hits[:3]: ctx.append(rag_answer.context_of(A["folder"], h))
        msgs = [{"role": "system", "content": rag_answer.SYSTEM}, {"role": "user", "content": "\n\n".join(ctx) + f"\n\nQUESTION: {q}"}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t = time.time(); out["answer"] = generate(model, tok, prompt=prompt, max_tokens=80, verbose=False).strip(); out["answer_s"] = round(time.time() - t, 2)
    return out

@app.get("/api/ask/thumb/{name}")
def ask_thumb(name: str):
    p = A["folder"] / Path(name).name
    if not p.exists() or p.suffix.lower() not in eng.EXT: raise HTTPException(404)
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im.thumbnail((360, 360)); b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return Response(b.getvalue(), media_type="image/jpeg")

@app.post("/api/ask/reveal/{name}")
def ask_reveal(name: str):
    import subprocess; subprocess.run(["open", "-R", str(A["folder"] / Path(name).name)]); return {"ok": True}

if __name__ == "__main__":
    threading.Thread(target=engine, daemon=True).start()        # warm the models while the page loads
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
