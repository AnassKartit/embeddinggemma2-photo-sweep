"""A fictional 'Downloads' folder for the 'ask your files' test: one made-up person (Sam Carter, the father of the test
family), electricity bills from three different years, other bills, a boarding pass, a lease, notes, installers, family photos, screenshots,
receipts and forwards, with realistic modified dates. Every document is fictional (no real people, numbers or companies).
PDFs are printed from HTML with headless Chrome; Word files with python-docx.
Usage: .venv/bin/python rag/make_downloads.py [out]   (default demo/Downloads)"""
import json, os, random, shutil, subprocess, sys, time
from pathlib import Path
from docx import Document
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "demo" / "Downloads"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
random.seed(11)
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True); TMP = ROOT / "data/tmp_html"; TMP.mkdir(exist_ok=True)

def when(y, m, d): return time.mktime((y, m, d, 10, random.randint(0, 59), 0, 0, 0, -1))
def stamp(p, ts): os.utime(p, (ts, ts))

CSS = "body{font-family:Helvetica,Arial;margin:48px;color:#111;font-size:14px;line-height:1.5}h1{font-size:26px;margin:0 0 6px}h2{font-size:16px;margin:22px 0 6px;border-bottom:1px solid #999}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px;text-align:left}.r{text-align:right}"
def pdf(name, html, ts):
    h = TMP / (Path(name).stem + ".html"); h.write_text(f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body>{html}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT / name}", f"file://{h}"],
                   capture_output=True, timeout=90)
    stamp(OUT / name, ts)
def docx(name, title, paras, ts):
    d = Document(); d.add_heading(title, 1)
    for p in paras: d.add_paragraph(p)
    d.save(OUT / name); stamp(OUT / name, ts)

power = lambda month, kwh: (lambda energy: (lambda tax: f"""<h1>Cascade Power</h1><div>Account 4410-2287 · Service address: 18 Elm Street, Portland · Customer: Sam Carter</div>
<h2>Electricity statement · {month}</h2><table><tr><th>Item</th><th class=r>Amount</th></tr><tr><td>Energy used: {kwh} kWh at $0.142</td><td class=r>${energy:,.2f}</td></tr>
<tr><td>Basic service fee</td><td class=r>$12.50</td></tr><tr><td>State energy tax 4%</td><td class=r>${tax:,.2f}</td></tr>
<tr><td><b>Total due</b></td><td class=r><b>${energy + 12.5 + tax:,.2f}</b></td></tr></table><div>Autopay is on.</div>""")(round((energy + 12.5) * .04, 2)))(round(kwh * .142, 2))
# three electricity bills: dull names, only the dates say which is the latest (the 'latest' test)
pdf("power_bill_march_2024.pdf", power("March 2024", 861), when(2024, 3, 2))
pdf("CascadePower_eBill.pdf", power("May 2025", 640), when(2025, 5, 14))
pdf("ebill_download.pdf", power("September 2026", 802), when(2026, 10, 3))
pdf("invoice_8812.pdf", """<h1>Riverside Family Dental</h1><div>Invoice 8812 · 12 September 2026</div><h2>Patient</h2><div>Sam Carter</div>
<table><tr><th>Service</th><th class=r>Amount</th></tr><tr><td>Check-up and cleaning</td><td class=r>$145.00</td></tr><tr><td>X-ray</td><td class=r>$40.00</td></tr>
<tr><td><b>Total due</b></td><td class=r><b>$185.00</b></td></tr></table><div>Please pay within 30 days.</div>""", when(2026, 9, 12))
pdf("internet_bill_sept.pdf", """<h1>Cascade Fiber</h1><div>Monthly statement · September 2026 · Account 55-1043</div>
<table><tr><td>Fiber 1 Gbps plan</td><td class=r>$69.99</td></tr><tr><td>Router rental</td><td class=r>$10.00</td></tr><tr><td><b>Total</b></td><td class=r><b>$79.99</b></td></tr></table>
<div>Autopay on 28 September 2026.</div>""", when(2026, 9, 20))
pdf("boarding_pass_PDX_JFK.pdf", """<h1>Boarding pass</h1><div>Passenger: CARTER / SAM</div><table><tr><td>From</td><td>Portland (PDX)</td></tr><tr><td>To</td><td>New York (JFK)</td></tr>
<tr><td>Date</td><td>21 November 2026</td></tr><tr><td>Flight</td><td>NW 482 · boarding 07:15 · gate C9 · seat 14A</td></tr></table>""", when(2026, 10, 5))
pdf("apartment_lease_2026.pdf", """<h1>Residential Lease Agreement</h1><div>Tenant: Sam Carter · Landlord: Elm Street Properties LLC</div>
<h2>Term</h2><div>The lease starts on 1 February 2026 and ends on 31 January 2027.</div><h2>Rent</h2><div>$2,150 per month, due on the 1st. Security deposit $2,150.</div>
<h2>Pets</h2><div>One dog allowed.</div>""", when(2026, 1, 20))
pdf("tax_summary_2025.pdf", """<h1>2025 Tax Summary</h1><div>Prepared for Sam and Emma Carter</div><table><tr><td>Total income</td><td class=r>$148,200</td></tr>
<tr><td>Federal tax withheld</td><td class=r>$24,310</td></tr><tr><td>Refund</td><td class=r>$1,120</td></tr></table>""", when(2026, 3, 30))
pdf("car_insurance_card.pdf", """<h1>Auto Insurance ID Card</h1><div>Insured: Sam Carter · Vehicle: 2019 family SUV</div><div>Policy valid 1 June 2026 to 1 June 2027.</div>
<div>Roadside assistance: call the number on your policy.</div>""", when(2026, 6, 1))
pdf("kids_school_calendar.pdf", """<h1>Maple Grove Elementary, 2026-2027 calendar</h1><table><tr><td>First day</td><td>2 September 2026</td></tr>
<tr><td>Thanksgiving break</td><td>25 to 27 November 2026</td></tr><tr><td>Winter break</td><td>21 December 2026 to 1 January 2027</td></tr>
<tr><td>Spring break</td><td>22 to 26 March 2027</td></tr></table>""", when(2026, 8, 25))
pdf("dishwasher_warranty.pdf", """<h1>Limited Warranty</h1><div>Product: built-in dishwasher, model DW-300</div><div>Purchased 14 April 2025. Parts and labour covered for 2 years.</div>""", when(2025, 4, 14))
pdf("concert_tickets.pdf", """<h1>E-ticket</h1><div>The Riverside Strings · live in concert</div><div>Saturday 15 November 2026, 8:00 pm · 2 tickets · Row F, seats 11-12</div>""", when(2026, 9, 30))
pdf("doctor_appointment.pdf", """<h1>Appointment confirmation</h1><div>Northgate Clinic</div><div>Sam Carter, annual physical, Tuesday 4 November 2026 at 9:30 am. Please arrive 10 minutes early.</div>""", when(2026, 10, 1))
docx("recipe_lasagna.docx", "Grandpa's lasagna", ["Ingredients: lasagna sheets, ground beef, tomato sauce, ricotta, mozzarella, parmesan.",
     "Bake 45 minutes at 375 F. Rest 15 minutes before serving."], when(2025, 11, 20))
docx("meeting_notes_q3.docx", "Q3 planning meeting notes", ["Attendees: platform team.", "Decisions: move the build cluster to the new region in October.",
     "Action items: Sam to draft the cost report by Friday."], when(2026, 7, 8))
docx("home_wifi_setup.docx", "Home Wi-Fi", ["Network name: CarterHome", "Router admin page: 192.168.1.1", "Guest network: CarterGuest"], when(2026, 2, 3))
for n, ts in (("Zoom.pkg", when(2026, 4, 2)), ("GoogleChrome.dmg", when(2026, 5, 9)), ("VLC-3.1.dmg", when(2025, 12, 1))):
    (OUT / n).write_bytes(os.urandom(2048)); stamp(OUT / n, ts)
k = 1000
for p in sorted((DATA / "family").glob("*.jpg")):
    k += 1; name = f"IMG_{k}.jpg"; im = Image.open(p).convert("RGB"); im.thumbnail((1600, 1600)); im.save(OUT / name, quality=85)
    stamp(OUT / name, when(2026, random.randint(1, 10), random.randint(1, 28)))
for i, p in enumerate(sorted((DATA / "junk/screenshot").glob("chat_*.png")), 1):
    name = f"Screenshot 2026-0{i + 3}-1{i} at 9.4{i}.12 PM.png"; shutil.copy(p, OUT / name); stamp(OUT / name, when(2026, i + 3, 10 + i))
for i, p in enumerate(sorted((DATA / "junk/receipt").glob("*.jpg"))[:5], 1):
    name = f"IMG_{2000 + i}.jpg"; shutil.copy(p, OUT / name); stamp(OUT / name, when(2026, 7, i))
for i, p in enumerate(sorted((DATA / "junk").glob("codex_meme_*.jpg")) + sorted((DATA / "junk").glob("codex_greet_morning*.jpg")), 1):
    name = f"IMG-20260{i + 4}1{i}-WA000{i}.jpg"; shutil.copy(p, OUT / name); stamp(OUT / name, when(2026, i + 4, 10 + i))
files = sorted(f.name for f in OUT.iterdir())
print(len(files), "files in", OUT); json.dump(files, open(ROOT / "rag/downloads_files.json", "w"), indent=1)
