"""Pull small labelled samples from public HF datasets via the datasets-server rows API (no full download).
receipts: naver-clova-ix/cord-v2 (CC BY 4.0). strangers: nlphuji/flickr30k test (research use; never shown on camera)."""
import json, random, requests, io, sys
from pathlib import Path
from PIL import Image
API = "https://datasets-server.huggingface.co/rows"
def rows(ds, config, split, offset, n):
    r = requests.get(API, params=dict(dataset=ds, config=config, split=split, offset=offset, length=n), timeout=60); r.raise_for_status()
    return [x["row"] for x in r.json()["rows"]]
def save(url, out):
    im = Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB"); im.save(out, quality=92); return im.size
meta = []
d = Path("data/junk/receipt"); d.mkdir(parents=True, exist_ok=True)
for i, r in enumerate(rows("naver-clova-ix/cord-v2", "default", "test", 0, 25)):
    out = d / f"cord_{i:02d}.jpg"; save(r["image"]["src"], out); meta.append({"path": str(out), "label": "receipt", "source": "cord-v2"})
random.seed(7)
people_words = ("family", "children", "kids", "group", "friends", "wedding", "people", "man", "woman", "girl", "boy")
d = Path("data/other_people"); d.mkdir(parents=True, exist_ok=True); n = 0
for off in (0, 100, 200, 300, 400):
    for r in rows("nlphuji/flickr30k", "TEST", "test", off, 100):
        cap = " ".join(r["caption"]).lower()
        if any(w in cap for w in people_words) and random.random() < 0.12 and n < 40:
            out = d / f"flickr_{n:02d}.jpg"; save(r["image"]["src"], out); meta.append({"path": str(out), "label": "other_people", "source": "flickr30k", "caption": r["caption"][0]}); n += 1
json.dump(meta, open("data/public_meta.json", "w"), indent=1); print(len(meta), "images")
