"""Control: do the models that 'accepted' an image really see it? Same question on 3 junk and 3 family photos (synthetic test set).
A model that reads pixels gives high p on junk and low p on family. Usage: .venv/bin/python decide/control.py"""
import base64, io, json, os, sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent)); import probe as P            # reuses key loading + call()
ROOT = Path(__file__).resolve().parent.parent
MODELS = [r["model"] for r in json.load(open(ROOT / "decide/probe.json")) if r["ok"]]
def url(p):
    im = Image.open(p).convert("RGB"); im.thumbnail((512, 512)); b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
junk = [next((ROOT / "data/junk").glob(f"codex_{n}*")) for n in ("meme_cat", "promo_flyer", "greet_bday")]
fam = [ROOT / f"data/family/fam_{n}.jpg" for n in ("thanksgiving", "beach", "camping")]
Q = {"type": "noul", "instructions": "Is this image junk the user can delete (a forwarded greeting, meme, flyer, screenshot or receipt), not a personal photo of people?"}
rows = []
for m in MODELS:
    ps = {}
    for tag, files in (("junk", junk), ("family", fam)):
        for p in files:
            body_state = [{"type": "text", "text": "A photo from a phone gallery."}, {"type": "image_url", "image_url": {"url": url(p)}}]
            d, s = P.call(m, body_state, Q)
            a = ((d.get("answers") or {}).get("q") or {}); ps.setdefault(tag, []).append(a.get("noul"))
    r = {"model": m, "p_junk_on_junk": ps["junk"], "p_junk_on_family": ps["family"]}
    ok = all(x is not None for x in ps["junk"] + ps["family"])
    r["sees_image"] = ok and min(ps["junk"]) > 0.5 and max(ps["family"]) < 0.5
    rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(ROOT / "decide/control.json", "w"), indent=1)
