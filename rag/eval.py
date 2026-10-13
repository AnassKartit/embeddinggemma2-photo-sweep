"""30 plain questions about the fictional Downloads folder (rag/make_downloads.py), right answers written down first.
Three ways to find the file, same questions:
  1. EmbeddingGemma 2 (rag/ask.py)
  2. macOS Spotlight: `mdfind -onlyin <folder> "<question>"` exactly as typed, and again with the stop words removed
  3. file-name search: files whose name contains a word of the question (what most file pickers do)
Hit@1 = the right file first; hit@5 = in the first five. Usage: .venv/bin/python rag/eval.py"""
import json, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import ask

ROOT = Path(__file__).resolve().parent.parent; FOLDER = ROOT / "demo/Downloads"
FAM = {p.stem.replace("fam_", ""): f"IMG_{1001 + i}.jpg" for i, p in enumerate(sorted((ROOT / "data/family").glob("*.jpg")))}
RECEIPTS = [f"IMG_{2000 + i}.jpg" for i in range(1, 6)]
CHATS = [p.name for p in FOLDER.glob("Screenshot*")]
MEMES = [p.name for p in FOLDER.glob("IMG-2026*WA*")][:3]
Q = [("where's my latest electricity bill?", ["ebill_download.pdf"]), ("my electricity bill from 2024", ["power_bill_march_2024.pdf"]),
     ("the dentist bill", ["invoice_8812.pdf"]), ("how much was my internet bill", ["internet_bill_sept.pdf"]),
     ("my flight to New York", ["boarding_pass_PDX_JFK.pdf"]), ("when does my lease end", ["apartment_lease_2026.pdf"]),
     ("my rental contract", ["apartment_lease_2026.pdf"]), ("tax document", ["tax_summary_2025.pdf"]),
     ("car insurance", ["car_insurance_card.pdf"]), ("grandpa's lasagna recipe", ["recipe_lasagna.docx"]),
     ("notes from the planning meeting", ["meeting_notes_q3.docx"]), ("when are the school holidays", ["kids_school_calendar.pdf"]),
     ("dishwasher warranty", ["dishwasher_warranty.pdf"]), ("concert tickets", ["concert_tickets.pdf"]),
     ("doctor appointment", ["doctor_appointment.pdf"]), ("wifi password", ["home_wifi_setup.docx"]),
     ("the Zoom installer", ["Zoom.pkg"]), ("the Thanksgiving photo", [FAM["thanksgiving"]]),
     ("grandpa fishing", [FAM["grandpa_fishing"]]), ("the snowman photo", [FAM["snow"]]),
     ("birthday cake", [FAM["bday"]]), ("graduation photo", [FAM["graduation"]]), ("the dog in the park", [FAM["dog_park"]]),
     ("camping trip", [FAM["camping"]]), ("day at the beach", [FAM["beach"]]), ("fourth of july barbecue", [FAM["july4"]]),
     ("a shop receipt", RECEIPTS), ("screenshot of a chat", CHATS), ("a funny meme", MEMES), ("christmas morning", [FAM["christmas"]])]
STOP = set("where's where is my the a an from of to when are how much was does do in at what".split())

def spotlight(q):
    out = subprocess.run(["mdfind", "-onlyin", str(FOLDER), q], capture_output=True, text=True).stdout.split("\n")
    return [Path(x).name for x in out if x.strip()]
def by_name(q):
    words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if w not in STOP and len(w) > 2]
    return [p.name for p in sorted(FOLDER.iterdir()) if any(w in p.name.lower() for w in words)]
def hit(got, want, k): return any(g in want for g in got[:k])

if __name__ == "__main__":
    m = ask.load_model(); idx = ask.Index(m, FOLDER); rows = []
    for q, want in Q:
        g, ms = idx.search(q); g = [h["name"] for h in g]
        sp = spotlight(q); sk = spotlight(" ".join(w for w in re.findall(r"[A-Za-z0-9']+", q) if w.lower() not in STOP)); nm = by_name(q)
        rows.append({"q": q, "want": want, "gemma": g, "gemma_ms": round(ms, 1), "spotlight": sp[:5], "spotlight_keywords": sk[:5], "name": nm[:5],
                     **{f"{k}@{n}": hit(v, want, n) for k, v in (("gemma", g), ("spotlight", sp), ("spotlight_keywords", sk), ("name", nm)) for n in (1, 5)}})
    res = {"files": len(idx.files), "vectors": int(len(idx.E)), "index_s": round(idx.index_s, 1), "questions": len(Q),
           "median_query_ms": sorted(r["gemma_ms"] for r in rows)[len(rows) // 2],
           **{k: sum(r[k] for r in rows) for k in rows[0] if "@" in k}, "rows": rows}
    json.dump(res, open(ROOT / "results/ask_eval.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})
    for r in rows:
        if not r["gemma@1"]: print("MISS@1", r["q"], "->", r["gemma"][:3], "want", r["want"])
