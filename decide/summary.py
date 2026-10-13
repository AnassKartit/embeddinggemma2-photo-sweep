"""Score decide/full.jsonl (6 image-reading decision models) next to EmbeddingGemma 2 zero-shot (results/bench_mps_70tok.json).
Usage: .venv/bin/python decide/summary.py"""
import json, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "decide/full.jsonl")]; out = []
for m in sorted({r["model"] for r in rows}):
    R = [r for r in rows if r["model"] == m]; ok = [r for r in R if r["p_junk"] is not None]
    junk = [r for r in ok if r["truth"] == "junk"]; keep = [r for r in ok if r["truth"] == "keep"]
    caught = sum(r["p_junk"] >= .5 for r in junk); lost = sum(r["p_junk"] >= .5 for r in keep)
    out.append({"model": m, "answered": f"{len(ok)}/{len(R)}", "accuracy": round((caught + len(keep) - lost) / len(ok), 3), "junk_caught": f"{caught}/{len(junk)}",
                "people_photos_lost": f"{lost}/{len(keep)}", "median_s": round(st.median(r["s"] for r in ok), 2), "cost_per_1000_photos": round(sum(r["cost"] for r in R) / len(R) * 1000, 3)})
b = json.load(open(ROOT / "results/bench_mps_70tok.json")); nj = sum(v for k, v in b["counts"].items() if k not in ("family", "other_family", "other_people")); nk = len(rows) // len(out) - nj
for z in b["zero_shot"][:2]:
    out.append({"model": f"EmbeddingGemma 2 zero-shot, local ({z['dim']} dims, 70 tokens)", "answered": f"{nj + nk}/{nj + nk}", "accuracy": round(1 - (z["junk_wrongly_kept"] + z["photos_wrongly_flagged"]) / (nj + nk), 3),
                "junk_caught": f"{nj - z['junk_wrongly_kept']}/{nj}", "people_photos_lost": f"{z['photos_wrongly_flagged']}/{nk}", "median_s": round(b["speed"]["ms_per_image"] / 1000, 3), "cost_per_1000_photos": 0})
json.dump(out, open(ROOT / "decide/summary.json", "w"), indent=1); [print(o) for o in out]
