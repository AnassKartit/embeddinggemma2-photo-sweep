"""Junk-or-keep on all 158 test photos: the 6 decision models that really read images (decide/control.json) vs EmbeddingGemma 2
zero-shot (local). Truth: junk = forwards, greetings, memes, quotes, screenshots, receipts, accidental; keep = photos of people.
(Who the people are is a face-matching question: no model here gets family info.) Photos sent as 512 px JPEG, synthetic/public
test images only. Spend cap $1.50 from usage.cost. Usage: .venv/bin/python decide/full.py"""
import base64, io, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent)); import probe as P
ROOT = Path(__file__).resolve().parent.parent; LOG = ROOT / "decide/full.jsonl"; CAP = 1.5; lock = threading.Lock()
MODELS = [r["model"] for r in json.load(open(ROOT / "decide/control.json")) if r["sees_image"]]
Q = {"type": "noul", "instructions": "Is this image junk the user can delete (a forwarded greeting, meme, flyer, screenshot, receipt or an accidental blurry shot), not a personal photo of people?"}
files = sorted(p for p in (ROOT / "data").rglob("*") if p.suffix.lower() in (".jpg", ".png") and not {"refs", "_whatsapp", "prompts"} & set(p.parts))
def truth(p): return "keep" if p.relative_to(ROOT / "data").parts[0] in ("family", "other_family", "other_people") else "junk"
done = {(r["model"], r["file"]) for r in map(json.loads, LOG.read_text().splitlines())} if LOG.exists() else set()
spent = sum(json.loads(l).get("cost") or 0 for l in LOG.read_text().splitlines()) if LOG.exists() else 0.0
def url(p):
    im = Image.open(p).convert("RGB"); im.thumbnail((512, 512)); b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
def job(mp):
    global spent
    m, p = mp; rel = str(p.relative_to(ROOT / "data"))
    if (m, rel) in done or spent > CAP: return
    d, s = P.call(m, [{"type": "text", "text": "A photo from a phone gallery."}, {"type": "image_url", "image_url": {"url": url(p)}}], Q)
    a = ((d.get("answers") or {}).get("q") or {}); c = (d.get("usage") or {}).get("cost") or 0
    with lock:
        spent += c; LOG.open("a").write(json.dumps({"model": m, "file": rel, "truth": truth(p), "p_junk": a.get("noul"), "s": s, "cost": c, "err": d.get("error", "")[:120] if "error" in d else None}) + "\n")
with ThreadPoolExecutor(8) as ex: list(ex.map(job, [(m, p) for p in files for m in MODELS]))
print("spent $", round(spent, 4))
