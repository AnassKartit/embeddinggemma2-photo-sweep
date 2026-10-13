"""A fictional 'Documents' folder for the 'analyze your bills' test: 12 monthly electricity bills (Nov 2025 to Oct 2026, made-up
utility 'Cascade Power'), 6 monthly bank statements (Apr to Sep 2026, made-up 'Pinecrest Bank'), plus water, phone and
internet bills, all for the made-up Sam Carter. Every amount is drawn from a fixed seed and the right answers are written to
rag/bills_truth.json BEFORE any model sees the files. PDFs printed from HTML with headless Chrome.
Usage: .venv/bin/python rag/make_bills.py [out]   (default demo/Documents)"""
import json, os, random, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "Documents"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
TMP = ROOT / "data/tmp_html"; TMP.mkdir(parents=True, exist_ok=True)
random.seed(2026)
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True)
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
CSS = ("body{font-family:Helvetica,Arial;margin:44px;color:#111;font-size:14px;line-height:1.45}h1{font-size:24px;margin:0 0 4px}"
       "table{border-collapse:collapse;width:100%;margin-top:12px}td,th{border:1px solid #bbb;padding:5px 8px;text-align:left}.r{text-align:right}.muted{color:#666}")

def pdf(name, html, y, m, d):
    h = TMP / (Path(name).stem + ".html"); h.write_text(f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body>{html}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT / name}", f"file://{h}"], capture_output=True, timeout=90)
    ts = time.mktime((y, m, d, 9, 0, 0, 0, 0, -1)); os.utime(OUT / name, (ts, ts))

truth = {"electricity": {}, "bank": {}}
# electricity: winter heating and summer air conditioning, so the yearly shape is realistic
base = {1: 162, 2: 148, 3: 121, 4: 96, 5: 84, 6: 109, 7: 141, 8: 156, 9: 118, 10: 92, 11: 117, 12: 158}
PERIOD = [(2025, 11), (2025, 12)] + [(2026, m) for m in range(1, 11)]          # the 12 months up to today (12 Oct 2026)
for yr, m in PERIOD:
    kwh = int(base[m] / 0.142 + random.randint(-40, 40)); energy = round(kwh * 0.142, 2); fee = 12.50; tax = round((energy + fee) * 0.04, 2)
    total = round(energy + fee + tax, 2); key = f"{MONTHS[m - 1]} {yr}"; truth["electricity"][key] = {"kwh": kwh, "total": total, "order": len(truth["electricity"])}
    name = f"cascade_power_{yr}_{m:02d}.pdf" if m % 3 else f"electricity_bill_{MONTHS[m - 1].lower()}_{yr}.pdf"     # inconsistent names, like real downloads
    pdf(name, f"""<h1>Cascade Power</h1><div class=muted>Account 4410-2287 · Service address: 18 Elm Street, Portland</div>
<h2>Electricity statement · {MONTHS[m - 1]} {yr}</h2><div>Customer: Sam Carter</div>
<table><tr><th>Item</th><th class=r>Amount</th></tr><tr><td>Energy used: {kwh} kWh at $0.142</td><td class=r>${energy:,.2f}</td></tr>
<tr><td>Basic service fee</td><td class=r>${fee:,.2f}</td></tr><tr><td>State energy tax 4%</td><td class=r>${tax:,.2f}</td></tr>
<tr><td><b>Total due</b></td><td class=r><b>${total:,.2f}</b></td></tr></table><div>Due on {MONTHS[m % 12]} 15. Autopay is on.</div>""",
        yr, m, 8 if (yr, m) == (2026, 10) else 25)
# bank statements: groceries, subscriptions, rent, salary
SHOPS = [("Green Basket Market", "groceries"), ("FreshFare Grocery", "groceries"), ("Netflix", "subscription"), ("Spotify", "subscription"),
         ("Shell fuel", "fuel"), ("Riverside Cafe", "eating out"), ("Cascade Power autopay", "electricity"), ("Elm Street Properties rent", "rent")]
for m in range(4, 10):
    rows, bal = [], 4200.0; spend = {}
    tx = [(1, "Elm Street Properties rent", "rent", -2150.00), (3, "Northwind Logistics salary", "salary", 6480.00)]
    for d in sorted(random.sample(range(2, 29), 10)):
        shop, cat = random.choice(SHOPS[:2] + SHOPS[4:6]); amt = -round(random.uniform(18, 140), 2); tx.append((d, shop, cat, amt))
    tx += [(12, "Netflix", "subscription", -15.49), (14, "Spotify", "subscription", -11.99), (15, "Cascade Power autopay", "electricity", -truth["electricity"][f"{MONTHS[m - 2]} 2026"]["total"])]
    tx.sort()
    for d, shop, cat, amt in tx:
        bal = round(bal + amt, 2); rows.append(f"<tr><td>{MONTHS[m - 1][:3]} {d:02d}</td><td>{shop}</td><td class=r>{amt:,.2f}</td><td class=r>{bal:,.2f}</td></tr>")
        if amt < 0: spend[cat] = round(spend.get(cat, 0) + -amt, 2)
    truth["bank"][MONTHS[m - 1]] = {"spend": spend, "closing_balance": bal}
    pdf(f"statement_{2026}-{m:02d}.pdf", f"""<h1>Pinecrest Bank</h1><div class=muted>Checking account ending 7731 · Sam Carter</div>
<h2>Monthly statement · {MONTHS[m - 1]} 2026</h2><table><tr><th>Date</th><th>Description</th><th class=r>Amount</th><th class=r>Balance</th></tr>{''.join(rows)}</table>
<div>Closing balance: ${bal:,.2f}</div>""", 2026, m, 28)
for m, (amt) in ((3, 46.20), (6, 51.85), (9, 49.10)):
    pdf(f"water_bill_q{(m - 1) // 3 + 1}_2026.pdf", f"<h1>Portland Water</h1><h2>Quarterly water bill · Q{(m - 1) // 3 + 1} 2026</h2><div>Sam Carter</div><div>Total due: ${amt:.2f}</div>", 2026, m, 20)
for m in (7, 8, 9):
    pdf(f"phone_{MONTHS[m - 1].lower()}.pdf", f"<h1>Summit Mobile</h1><h2>Phone bill · {MONTHS[m - 1]} 2026</h2><div>Unlimited plan</div><div>Total: $45.00</div>", 2026, m, 18)
el = truth["electricity"]; tot = round(sum(v["total"] for v in el.values()), 2); keys = list(el)
hi = max(el, key=lambda k: el[k]["total"]); lo = min(el, key=lambda k: el[k]["total"])
last, prev, year_ago = keys[-1], keys[-2], None
subs = sorted({(k.split()[0]) for mth in truth["bank"].values() for k in mth["spend"] if mth["spend"].get("subscription")})
truth["answers"] = {
    "total_electricity_12_months": tot, "highest_month": hi, "highest_total": el[hi]["total"], "lowest_month": lo, "lowest_total": el[lo]["total"],
    "average_electricity": round(tot / 12, 2), "last_month": last, "last_total": el[last]["total"], "previous_month": prev, "previous_total": el[prev]["total"],
    "last_vs_previous_pct": round((el[last]["total"] / el[prev]["total"] - 1) * 100, 1),
    "first_half_avg": round(sum(el[k]["total"] for k in keys[:6]) / 6, 2), "second_half_avg": round(sum(el[k]["total"] for k in keys[6:]) / 6, 2),
    "groceries_september": truth["bank"]["September"]["spend"].get("groceries", 0),
    "bank_6_month_spend_by_category": {c: round(sum(mth["spend"].get(c, 0) for mth in truth["bank"].values()), 2) for c in sorted({c for mth in truth["bank"].values() for c in mth["spend"]})}}
json.dump(truth, open(ROOT / "rag/bills_truth.json", "w"), indent=1)
print(len(list(OUT.iterdir())), "files in", OUT); print(json.dumps(truth["answers"], indent=1))
