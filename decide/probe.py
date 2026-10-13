"""Which OpenRouter decision models accept an IMAGE in the state? One call per model and state format, synthetic test photo only.
Key: ~/.config/akatech/image_bench.env (never printed). Usage: .venv/bin/python decide/probe.py"""
import base64, io, json, os, time, urllib.request, urllib.error
from pathlib import Path
from PIL import Image
for ln in (Path.home() / ".config/akatech/image_bench.env").read_text().splitlines():
    if "=" in ln and not ln.startswith("#"): k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"'))
ROOT = Path(__file__).resolve().parent.parent
MODELS = ["microsoft/microsoft-decision-1", "openai/gpt-6-luna-decisions", "perplexity/pplx-decider-v1.1-27b", "cloudflare/clef-flash",
          "cloudflare/clef", "cloudflare/clef-omni", "inception/mercury-decide:free", "typesafe/jev-1.13", "nace-ai/drex-v1.5", "liquid/d1",
          "upstage/solar-decide-flash", "upstage/solar-decide", "togethercomputer/tev1-4b-experimental", "jaredpalmer/kev-4b"]
im = Image.open(next((ROOT / "data/junk").glob("codex_greet_morning1*"))).convert("RGB"); im.thumbnail((512, 512))
b = io.BytesIO(); im.save(b, "JPEG", quality=80); url = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
Q = {"type": "noul", "instructions": "Is this image a forwarded greeting card (junk), not a personal photo?"}
STATES = {"parts": [{"type": "text", "text": "A photo from a phone gallery."}, {"type": "image_url", "image_url": {"url": url}}],
          "messages": [{"role": "user", "content": [{"type": "text", "text": "A photo from a phone gallery."}, {"type": "image_url", "image_url": {"url": url}}]}]}
def call(model, state, q=None):
    body = json.dumps({"model": model, "state": state, "questions": {"q": q or Q}}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", body, {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
          "Content-Type": "application/json", "HTTP-Referer": "https://www.kartit.net", "X-Title": "photo-sweep probe"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r: d = json.loads(r.read())
    except urllib.error.HTTPError as e: d = {"error": e.read().decode()[:220], "status": e.code}
    except Exception as e: d = {"error": str(e)[:220]}
    return d, round(time.time() - t, 2)
def main():
  out = []
  for m in MODELS:
    for name, st in STATES.items():
        d, s = call(m, st); a = ((d.get("answers") or {}).get("q") or {})
        r = {"model": m, "state": name, "ok": "error" not in d, "p_yes": a.get("noul"), "s": s, "cost": (d.get("usage") or {}).get("cost"),
             "err": d.get("error", "")[:160] if "error" in d else None}
        out.append(r); print(json.dumps(r), flush=True)
        if r["ok"]: break
  json.dump(out, open(ROOT / "decide/probe.json", "w"), indent=1)
if __name__ == "__main__": main()
