"""Build a realistic 'WhatsApp Images' folder from the test set, for the app demo.
WhatsApp-style names (IMG-YYYYMMDD-WA####.jpg), dates spread over 3 months, every file re-saved like WhatsApp (long side 1600,
JPEG q70), and the forwards arriving 2 to 4 times from different groups (re-compressed copies), the way junk piles up.
Usage: .venv/bin/python app/make_demo.py [out_dir]   (default demo/WhatsApp Images)"""
import io, os, random, shutil, sys, time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "WhatsApp Images"
random.seed(42)
# only images that may appear in a published video: the fictional American family, Codex forwards, CORD receipts (CC BY 4.0),
# our own chat screenshots, plain black pocket shots. Left out: other families (not shown on camera), Flickr30k strangers and
# the blurred shots made from them (research licence), screenshots of third-party websites.
def shareable(p):
    r = p.relative_to(DATA).parts
    if r[0] in ("other_family", "other_people", "refs", "_whatsapp", "prompts", "tmp_html"): return False
    if r[0] == "junk" and len(r) > 2 and r[1] == "screenshot": return p.name.startswith("chat_")
    if r[0] == "junk" and len(r) > 2 and r[1] == "accidental": return p.name in ("acc_00.jpg", "acc_01.jpg", "acc_02.jpg")
    return True
src = [p for p in DATA.rglob("*") if p.suffix.lower() in (".jpg", ".png") and shareable(p)]
forwards = [p for p in src if "codex_" in p.name]
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)
start = time.mktime((2026, 7, 10, 8, 0, 0, 0, 0, -1)); span = 92 * 86400
items = [(p, 1) for p in src] + [(p, random.randint(1, 3)) for p in random.sample(forwards, 14)]   # extra re-forwards
events = []
for p, copies in items:
    for _ in range(copies): events.append((start + random.random() * span, p))
events.sort()
counter = {}
for ts, p in events:
    day = time.strftime("%Y%m%d", time.localtime(ts)); counter[day] = counter.get(day, 0) + 1
    name = f"IMG-{day}-WA{counter[day]:04d}.jpg"
    im = Image.open(p).convert("RGB"); im.thumbnail((1600, 1600))
    q = random.choice([70, 65, 60]) if "codex_" in p.name else 70                                # re-forwards lose a bit more
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); (OUT / name).write_bytes(b.getvalue()); os.utime(OUT / name, (ts, ts))
tot = sum(f.stat().st_size for f in OUT.iterdir())
print(f"{len(events)} files, {tot / 1e6:.1f} MB in {OUT}")
