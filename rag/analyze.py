"""Analyze your bills locally: find -> extract -> calculate -> explain. All on this Mac.
  find       EmbeddingGemma 2 ranks the folder for "electricity bill" / "bank statement"; files close to the best match are kept
  extract    Gemma 4 E4B reads ONE file at a time and returns JSON (month, total, kWh / every transaction with a category)
  calculate  plain Python adds, compares and finds the trend (small language models are bad at adding many numbers)
  explain    Gemma 4 E4B turns the calculated numbers (not the raw files) into a short answer
Measured against rag/bills_truth.json, written when the fictional files were made, and against the naive way: one prompt with
all 12 bills, "what is the total?". Usage: .venv/bin/python rag/analyze.py   -> results/analyze.json"""
import json, re, statistics as st, sys, time, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import ask

ROOT = Path(__file__).resolve().parent.parent; FOLDER = ROOT / "demo/Documents"; LLM = "mlx-community/gemma-4-E4B-it-qat-4bit"
TRUTH = json.load(open(ROOT / "rag/bills_truth.json")); MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                                                              "September", "October", "November", "December"]
CATS = ["groceries", "eating out", "fuel", "subscription", "rent", "electricity", "salary"]


class Analyst:
    def __init__(self, idx):
        from mlx_lm import load
        self.idx = idx; self.model, self.tok = load(LLM)

    def llm(self, system, user, max_tokens=600):
        from mlx_lm import generate
        p = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        return generate(self.model, self.tok, prompt=p, max_tokens=max_tokens, verbose=False).strip()

    def find(self, what, margin=0.08):
        """Every file within `margin` of the best match for `what` (one search, no fixed k)."""
        hits, ms = self.idx.search(what, k=len(self.idx.files)); top = hits[0]["score"]
        return [h["name"] for h in hits if h["score"] >= top - margin], ms

    @staticmethod
    def json_of(txt):
        m = re.search(r"[\[{].*[\]}]", txt, re.S)
        try: return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError: return None

    def electricity(self):
        t = time.time(); files, ms = self.find("electricity bill"); rows = []
        for f in files:
            txt = ask.text_of(FOLDER / f)[:2500]
            out = self.json_of(self.llm("Extract data from one utility bill. Reply with JSON only: "
                                        '{"is_electricity_bill": true/false, "month": "Month YYYY", "total_due": number, "kwh": number}', txt, 120))
            if out and out.get("is_electricity_bill"): rows.append({"file": f, **out})
        # calculate in Python
        def key(r):
            mth, yr = (r.get("month") or " 0").rsplit(" ", 1); return int(yr) * 12 + (MONTHS.index(mth) if mth in MONTHS else 0)
        rows.sort(key=key); tot = round(sum(float(r["total_due"]) for r in rows), 2)
        last, prev = rows[-1], rows[-2]; half = len(rows) // 2
        calc = {"bills_found": len(rows), "files_searched": len(files), "total": tot, "average": round(tot / len(rows), 2),
                "highest": max(rows, key=lambda r: r["total_due"])["month"], "lowest": min(rows, key=lambda r: r["total_due"])["month"],
                "last_month": last["month"], "last_total": last["total_due"], "previous_month": prev["month"], "previous_total": prev["total_due"],
                "last_vs_previous_pct": round((last["total_due"] / prev["total_due"] - 1) * 100, 1),
                "first_half_avg": round(st.mean(r["total_due"] for r in rows[:half]), 2), "second_half_avg": round(st.mean(r["total_due"] for r in rows[half:]), 2),
                "series": [(r["month"], r["total_due"], r.get("kwh")) for r in rows], "search_ms": round(ms, 1)}
        calc["trend_pct"] = round((calc["second_half_avg"] / calc["first_half_avg"] - 1) * 100, 1)
        calc["answer"] = self.llm("You explain a household's electricity bills in 2 short sentences, using only these calculated numbers. "
                                  "Sentence 1: the last bill compared to the one before. Sentence 2: the trend, using first_half_avg, second_half_avg and trend_pct.",
                                  json.dumps({k: v for k, v in calc.items() if k not in ("series", "search_ms")}) +
                                  "\nQuestion: what was my last electricity bill compared to the one before, and is it going up?", 120)
        calc["seconds"] = round(time.time() - t, 1); return calc

    def bank(self):
        t = time.time(); files, ms = self.find("bank account statement"); spend, n_tx, subs = {}, 0, {}
        for f in files:
            txt = ask.text_of(FOLDER / f)[:4000]
            out = self.json_of(self.llm("Extract every transaction from one bank statement. Reply with a JSON list only: "
                                        '[{"description": text, "amount": number (negative = money out), "category": one of ' + json.dumps(CATS) + "}]", txt, 1200))
            for tx in (out or []) if isinstance(out, list) else []:
                try: a = float(tx["amount"])
                except (KeyError, TypeError, ValueError): continue
                n_tx += 1
                if a < 0:
                    c = tx.get("category", "other"); spend[c] = round(spend.get(c, 0) + -a, 2)
                    if c == "subscription": subs[unicodedata.normalize("NFKC", tx.get("description", "?"))] = -a   # PDF text has ligatures (ﬂ)
        months = len(files)
        calc = {"statements": months, "transactions": n_tx, "spend_by_category": dict(sorted(spend.items(), key=lambda kv: -kv[1])),
                "monthly_average_by_category": {c: round(v / max(months, 1), 2) for c, v in spend.items()}, "search_ms": round(ms, 1)}
        mo = calc["monthly_average_by_category"]                  # code works out the savings; the model only phrases them
        calc["possible_savings_per_month"] = {
            "eat out half as often": round(mo.get("eating out", 0) / 2, 2),
            "cancel the subscriptions " + ", ".join(sorted(subs)): round(sum(subs.values()), 2),
            "cut fuel by a fifth (car-share or walk short trips)": round(mo.get("fuel", 0) / 5, 2)}
        sub_names = " and ".join(sorted(subs))                    # the sentences come from the calculated numbers, not from the model
        calc["answer"] = "\n".join([
            f"Eat out half as often: save about ${mo.get('eating out', 0) / 2:,.2f} a month (you spend ${mo.get('eating out', 0):,.2f} a month eating out).",
            f"Cancel {sub_names}: save ${sum(subs.values()):,.2f} a month.",
            f"Drive a fifth less: save about ${mo.get('fuel', 0) / 5:,.2f} a month (fuel is ${mo.get('fuel', 0):,.2f} a month)."])
        calc["seconds"] = round(time.time() - t, 1); return calc

    def naive_total(self, files):
        """The naive way: all bills in one prompt, the model adds them up itself."""
        txt = "\n\n".join(f"FILE {f}\n" + ask.text_of(FOLDER / f)[:900] for f in files)
        out = self.llm("Answer with one number only.", txt + "\n\nWhat is the total of all electricity bills above, in dollars?", 40)
        m = re.search(r"[\d,]+\.?\d*", out.replace("$", "")); return out, (float(m.group(0).replace(",", "")) if m else None)


if __name__ == "__main__":
    m = ask.load_model(); idx = ask.Index(m, FOLDER); a = Analyst(idx); A = TRUTH["answers"]
    el = a.electricity(); bank = a.bank()
    naive_out, naive_val = a.naive_total(a.find("electricity bill")[0])
    def ok(x, y, tol=0.011): return x is not None and abs(float(x) - float(y)) <= tol
    score = {"bills_found": f'{el["bills_found"]}/12', "total_exact": ok(el["total"], A["total_electricity_12_months"]),
             "per_bill_exact": sum(ok(t, TRUTH["electricity"][mth]["total"]) for mth, t, _ in el["series"] if mth in TRUTH["electricity"]),
             "highest_right": el["highest"] == A["highest_month"], "last_vs_previous_right": ok(el["last_vs_previous_pct"], A["last_vs_previous_pct"], .05),
             "naive_total": naive_val, "naive_total_exact": ok(naive_val, A["total_electricity_12_months"]),
             "bank_categories_exact": {c: ok(bank["spend_by_category"].get(c), v, .02) for c, v in A["bank_6_month_spend_by_category"].items()}}
    res = {"llm": LLM, "electricity": el, "bank": bank, "score": score, "naive_answer": naive_out, "truth": A}
    json.dump(res, open(ROOT / "results/analyze.json", "w"), indent=1)
    print(json.dumps(score, indent=1)); print("ELECTRICITY:", el["answer"]); print("COSTS:", bank["answer"]); print("seconds", el["seconds"], bank["seconds"])
