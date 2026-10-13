"""Record the app demo for the video: real clicks in real Chrome, captured as crisp 1920x1080 frames (8 fps), then
encoded to MP4. Steps and their times go to <out>.json so the video build can label them.
Needs the app running: .venv/bin/python app/server.py "demo/WhatsApp Images"  (delete demo/WhatsApp Images/.photo-sweep first
for a fresh, timed scan). Usage: .venv/bin/python app/record_demo.py out_dir"""
import json, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

MODE = sys.argv[2] if len(sys.argv) > 2 else "clean"          # clean | ask | names (needs a scanned folder and no people.json yet)
TEACH = "f238ea6401d404d0"                                     # names mode: the airport photo, one click per person, left to right
NAMES = ["Walt", "Sam", "Emma", "Lily", "Max"]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/recording"); FR = OUT / f"frames_{MODE}"; FR.mkdir(parents=True, exist_ok=True)
for f in FR.glob("*.jpg"): f.unlink()
FPS, steps, t0, frames = 8, [], [None], [0]

def mark(name):   # frame index = position in the final video at FPS (screenshots can be slower than real time)
    steps.append({"step": name, "t_real": round(time.time() - t0[0], 2), "frame": frames[0], "t_video": round(frames[0] / FPS, 2)})
    print(f"{steps[-1]['t_real']:6.1f}s real, {steps[-1]['t_video']:6.1f}s video  {name}", flush=True)

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    ZOOM = "document.body.style.zoom='1.35'"          # bigger UI so it reads on a phone screen in the video
    pg.goto("http://127.0.0.1:8765"); pg.evaluate(ZOOM); pg.wait_for_timeout(1500)
    # Playwright pages are not thread-safe: grab frames from the main thread between actions instead
    def hold(sec):
        end = time.time() + sec
        while time.time() < end:
            ts = time.time(); pg.screenshot(path=str(FR / f"{frames[0]:05d}.jpg"), type="jpeg", quality=92); frames[0] += 1
            time.sleep(max(0, 1 / FPS - (time.time() - ts)))
    def type_slow(sel, text):   # capture frames while typing, so the video shows the typing
        pg.click(sel)
        for ch in text: pg.type(sel, ch); hold(0.07)
    t0[0] = time.time()
    if MODE == "names":
        mark("app"); hold(1.2)
        pg.click('.pile[data-k="people"]'); hold(1.0)
        card = f'.ph[data-id="{TEACH}"]'; pg.locator(card).scroll_into_view_if_needed(); pg.hover(card); hold(0.6)
        pg.click(card + " .who"); mark("namer"); hold(1.2)
        order = pg.evaluate(f"(()=>{{const p=R.photos.find(x=>x.id==='{TEACH}');return p.face_boxes.map((f,i)=>[f[0],i]).sort((a,b)=>a[0]-b[0]).map(x=>x[1])}})()")
        for name, i in zip(NAMES, order):
            box = pg.locator("#mbox .fbox").nth(i); box.click(); hold(0.3)
            for ch in name: pg.keyboard.type(ch); hold(0.08)
            pg.keyboard.press("Enter"); mark("named_" + name.lower()); hold(1.3)
        hold(0.8); pg.mouse.click(30, 1060); pg.evaluate("scrollTo(0,0)"); pg.click('.pile[data-k="family"]'); mark("names_done"); hold(2.2)
        for q, sec in (("Lily", 4.5), ("Max birthday", 4.0), ("whole family", 4.5)):
            pg.evaluate("scrollTo(0,0)"); pg.fill("#search", ""); mark("search_" + q.replace(" ", "_")); type_slow("#search", q)
            hold(sec); pg.mouse.wheel(0, 380); hold(1.4); pg.mouse.wheel(0, -380); hold(0.4)
        mark("end"); b.close()
    elif MODE == "ask":
        pg.click('.tab[data-v="askv"]'); mark("app"); hold(1.2)
        pg.click("#aindex"); mark("index")
        while pg.evaluate("fetch('/api/ask/state').then(r=>r.json()).then(s=>s.state)") != "ready": hold(0.3)
        hold(1.0)
        for name, q in (("dentist", "how much do I owe the dentist?"), ("lease", "when does my lease end?"), ("photo", "the Thanksgiving photo")):
            pg.fill("#aq", ""); mark("ask_" + name); type_slow("#aq", q); pg.keyboard.press("Enter")
            while True:   # keep filming while Gemma 4 E4B thinks
                hold(0.25); txt = pg.evaluate("document.querySelector('#atext').textContent")
                if txt and txt != "Thinking…": break
            mark("answered_" + name); hold(3.0)
        mark("end"); b.close()
    else:
        mark("app"); hold(1.5)
        pg.click("#scan"); mark("scan")
        while True:      # the progress bar only: 2 frames a second, so filming does not slow the scan much
            pg.screenshot(path=str(FR / f"{frames[0]:05d}.jpg"), type="jpeg", quality=92); frames[0] += 1; time.sleep(0.5)
            st = pg.evaluate("fetch('/api/state').then(r=>r.json()).then(s=>s.progress)")
            if st["state"] == "done": break
        hold(0.5); mark("scanned"); pg.click('.pile[data-k="people"]'); hold(2.0)
        pg.fill("#search", ""); type_slow("#search", "turkey dinner at the table"); hold(1.4)
        pg.hover("#grid .ph:nth-child(1)"); hold(0.6)
        pg.click("#grid .ph:nth-child(1) .heart"); mark("heart")
        while not pg.evaluate("fetch('/api/state').then(r=>r.json()).then(s=>s.progress.state==='done' && s.result && s.result.piles.family.length>0)"): hold(0.3)
        hold(0.8); pg.reload(); pg.evaluate(ZOOM); hold(1.2); mark("family"); pg.mouse.wheel(0, 500); hold(2.2); pg.mouse.wheel(0, -500); hold(0.6)
        for k in ("duplicates", "forwards", "screenshots", "accidental"):
            pg.click(f'.pile[data-k="{k}"]'); mark("pile_" + k); hold(1.6)
        pg.click('.pile[data-k="forwards"]'); hold(0.6); pg.click("#selall"); mark("select"); hold(1.2)
        pg.click("#sweep"); mark("sweep"); hold(2.6)
        pg.click("#undo"); mark("undo"); hold(2.2)
        for q in ("receipt", "birthday cake"):
            pg.evaluate("scrollTo(0,0)"); pg.fill("#search", ""); mark("search_" + q.replace(" ", "_")); type_slow("#search", q); hold(2.4)
        mark("end"); b.close()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(FR / "%05d.jpg"), "-vf", "fps=30,format=yuv420p",
                "-c:v", "libx264", "-crf", "16", "-preset", "slow", str(OUT / f"{MODE}.mp4")], check=True)
(OUT / f"{MODE}.json").write_text(json.dumps({"fps": FPS, "frames": frames[0], "steps": steps}, indent=1))
print("frames", frames[0], "->", OUT / f"{MODE}.mp4")
