"""Rebuild the parts of the test set that are not shipped in the repo:
  screenshots (15 mobile web pages + 5 fake chats, headless Chrome) and accidental shots (from the Flickr photos).
Run after fetch_public.py. Usage: .venv/bin/python bench/make_local.py"""
import os, random, subprocess, glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data/junk"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
URLS = ["https://en.wikipedia.org/wiki/Morocco", "https://en.wikipedia.org/wiki/Mint_tea", "https://www.kartit.net", "https://github.com/trending",
        "https://news.ycombinator.com", "https://www.bbc.com/news", "https://www.timeanddate.com/weather/morocco/casablanca", "https://www.python.org",
        "https://developer.apple.com/documentation/photokit", "https://huggingface.co/google/embeddinggemma-2", "https://www.lemonde.fr",
        "https://www.hespress.com/", "https://openrouter.ai/models", "https://www.wikipedia.org", "https://www.booking.com"]
def shot(url, out):
    try: subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=390,844", "--force-device-scale-factor=3",
                    "--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
                    f"--screenshot={out}", url], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired: print("skipped (timeout):", url)
def chat_html(seed):
    random.seed(seed)
    msgs = [("Did you see the match yesterday??", 0), ("Yes!! what a goal", 1), ("Send me the PDF for the meeting", 0), ("ok sending now", 1),
            ("Who is coming Friday?", 0), ("Me + 2", 1), ("Don't forget the bread", 0), ("👍", 1), ("Call me when you are free", 0),
            ("Traffic is crazy on the highway", 1), ("Price went up again 😅", 0), ("Happy birthday bro!!! 🎉", 1), ("Meeting moved to 3pm", 0)]
    random.shuffle(msgs); name = random.choice(["Mike", "Family ❤️", "Work team", "Jenny", "Neighbors"])
    rows = "".join(f'<div class="m {"me" if s else "them"}">{t}<span>{random.randint(8,22)}:{random.randint(10,59)}</span></div>' for t, s in msgs[:9])
    return f"""<html><body style="margin:0;font-family:-apple-system,Helvetica;background:#efe7de">
<div style="background:#f6f6f6;padding:52px 14px 10px;border-bottom:1px solid #ddd;font-weight:600;font-size:17px">‹ &nbsp; {name}</div>
<style>.m{{max-width:72%;margin:8px 10px;padding:8px 10px;border-radius:9px;font-size:16px;box-shadow:0 1px 1px #0002}}
.me{{background:#dcf8c6;margin-left:auto}}.them{{background:#fff}}.m span{{display:block;text-align:right;font-size:11px;color:#888}}</style>
<div style="padding-top:6px">{rows}</div></body></html>"""
(D / "screenshot").mkdir(parents=True, exist_ok=True); tmp = ROOT / "data/tmp_html"; tmp.mkdir(exist_ok=True)
for i, u in enumerate(URLS, 1): shot(u, D / f"screenshot/web_{i:02d}.png")
for i in range(1, 6): (tmp / f"c{i}.html").write_text(chat_html(i)); shot(f"file://{tmp / f'c{i}.html'}", D / f"screenshot/chat_{i}.png")
random.seed(3); np.random.seed(3); src = sorted(glob.glob(str(ROOT / "data/other_people/*.jpg"))); (D / "accidental").mkdir(exist_ok=True)
for i in range(10):
    if i < 3: im = Image.fromarray(np.clip(np.random.normal(12 + i * 6, 6, (1440, 1080, 3)), 0, 255).astype("uint8"))   # pocket: near black
    else:
        im = Image.open(random.choice(src)).convert("RGB").rotate(random.choice([90, 180, 13, -21]), expand=True)            # floor / motion blur
        im = ImageEnhance.Brightness(im.filter(ImageFilter.GaussianBlur(random.choice([18, 25, 35])))).enhance(random.choice([0.25, 0.4, 1.6]))
    im.save(D / f"accidental/acc_{i:02d}.jpg", quality=85)
print("screenshots:", len(list((D / "screenshot").glob("*.png"))), "accidental:", len(list((D / "accidental").glob("*.jpg"))))
