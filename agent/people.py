"""Learn who is in your photos: find every face in some folders, group the same person together, then you name the
groups and teach it. Everything stays in ~/.photosweep/people/ (fingerprints, small face crops, your names and
corrections). Nothing is uploaded, nothing is committed: this folder is outside the repo.
  1. scan:   .venv/bin/python agent/people.py scan <folder> [<folder> ...]
  2. teach:  .venv/bin/python agent/people.py serve     -> http://127.0.0.1:8770
             name a group, click the faces in it that are NOT that person, hide groups you don't care about,
             then "Teach": it shows the faces it is least sure about, you click the wrong ones, it learns from both.
  3. find:   .venv/bin/python agent/people.py find      -> per name: faces you confirmed, photos found, and how often
             its guesses were right in your teach rounds (only numbers are printed, never paths)
Fingerprints: ArcFace (agent/faceid.py), with SFace as a fallback. Grouping: a face joins the group whose average
fingerprint is closest if the similarity is at least JOIN, else starts a group. Naming two groups the same name
merges them (side views and different ages often form their own group). Learning = examples, not retraining: every
face you confirm becomes another example of that person (so a 2-year-old and a 5-year-old Kays both count), and
every face you reject becomes a "not them" example that blocks look-alikes."""
import json, os, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parent))
from faceid import Embedder, finder

HOME = Path(os.environ.get("PHOTOSWEEP_PEOPLE", Path.home() / ".photosweep" / "people")); CROPS = HOME / "crops"
EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}; MIN_FACE = 40
CUT = {"arcface": (0.40, 0.40), "sface": (0.45, 0.50)}   # (JOIN, TH) per model; arcface TH from bench/names.py


def load(p, size=1600):
    if p.suffix.lower() == ".heic":
        from pillow_heif import register_heif_opener; register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im.thumbnail((size, size)); return im


def scan(folders, model="arcface"):
    det, emb = finder(), Embedder(model); CROPS.mkdir(parents=True, exist_ok=True); JOIN = CUT[model][0]
    files = sorted({p for f in folders for p in Path(f).expanduser().rglob("*") if p.suffix.lower() in EXT and "photoslibrary" not in str(p)})
    faces, t0, bad = [], time.time(), 0
    for n, p in enumerate(files):
        try: im = load(p)
        except Exception: bad += 1; continue
        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR); h, w = img.shape[:2]
        det.setInputSize((w, h)); _, found = det.detect(img)
        for f in (found if found is not None else []):
            if min(f[2], f[3]) < MIN_FACE: continue
            c = emb.crop(img, f); i = len(faces); cv2.imwrite(str(CROPS / f"{i}.jpg"), c)
            faces.append({"id": i, "photo": str(p), "box": [round(float(v), 1) for v in f[:4]], "size": [w, h], "vec": emb.vec(c).tolist()})
        if n % 250 == 0: print(f"{n}/{len(files)} photos, {len(faces)} faces, {time.time() - t0:.0f} s", flush=True)
    # group: greedy, biggest faces first so each group starts from a clear face
    order = sorted(range(len(faces)), key=lambda i: -faces[i]["box"][2] * faces[i]["box"][3]); C, S = [], []
    for i in order:
        v = np.array(faces[i]["vec"]); best = max(range(len(C)), key=lambda c: float(S[c] @ v / np.linalg.norm(S[c])), default=None)
        if best is not None and float(S[best] @ v / np.linalg.norm(S[best])) >= JOIN: C[best].append(i); S[best] = S[best] + v
        else: C.append([i]); S.append(v.copy())
    groups = sorted(C, key=len, reverse=True)
    stats = {"model": model, "photos": len(files), "unreadable": bad, "faces": len(faces), "groups": len(groups),
             "groups_with_5_plus_faces": sum(len(g) >= 5 for g in groups), "seconds": round(time.time() - t0, 1)}
    json.dump({"faces": faces, "groups": groups, "stats": stats}, open(HOME / "faces.json", "w"))
    print(json.dumps(stats))


# ---- what you taught it -------------------------------------------------------------------------------------------
def state():
    p = HOME / "names.json"; s = json.load(open(p)) if p.exists() else {}
    if isinstance(s, list): s = {"names": {str(i): n for i, n in enumerate(s) if n}}      # first version: a plain list
    for k in ("names", "off", "yes", "no"): s.setdefault(k, {})
    for k in ("hidden", "rounds"): s.setdefault(k, [])
    return s


def key(n): return " ".join(n.split()).lower()


class Brain:
    """Every face gets the closest name, if it is closer than TH to that person's examples and closer to them than to
    the faces you said are not them."""
    def __init__(self, D, s):
        self.D, self.s = D, s; self.V = np.array([f["vec"] for f in D["faces"]], np.float32)
        self.TH = CUT[D["stats"].get("model", "sface")][1]; G = D["groups"]
        self.hidden = {i for g in s["hidden"] for i in G[int(g)]}
        self.pos, self.neg, self.label = {}, {}, {}
        for g, n in s["names"].items():
            if not n.strip(): continue
            k = key(n); self.label.setdefault(k, n.strip()); off = set(s["off"].get(g, []))
            self.pos.setdefault(k, set()).update(i for i in G[int(g)] if i not in off); self.neg.setdefault(k, set()).update(off)
        for k, ids in s["yes"].items(): self.pos.setdefault(k, set()).update(ids); self.label.setdefault(k, k.title())
        for k, ids in s["no"].items(): self.neg.setdefault(k, set()).update(ids)
        for k in self.pos: self.pos[k] -= self.neg.get(k, set())
        self.names = [k for k in self.pos if self.pos[k]]
        E = np.full((len(self.names), len(self.V)), -1.0, np.float32)
        for j, k in enumerate(self.names):
            p = (self.V @ self.V[sorted(self.pos[k])].T).max(1)
            ng = self.neg.get(k)
            q = (self.V @ self.V[sorted(ng)].T).max(1) if ng else np.full(len(self.V), -1.0)
            e = np.where(p > q, p, -1.0); e[sorted(self.pos[k])] = 1.0
            if ng: e[sorted(ng)] = -1.0
            E[j] = e
        self.score = E.max(0) if len(self.names) else np.full(len(self.V), -1.0)
        self.best = E.argmax(0) if len(self.names) else np.zeros(len(self.V), int); self.E = E

    def who(self, i): return self.names[self.best[i]] if self.score[i] >= self.TH and i not in self.hidden else None

    def people(self):
        F = self.D["faces"]; out = []; photos = {}
        for i in range(len(F)):
            n = self.who(i)
            if n: photos.setdefault(n, set()).add(F[i]["photo"])
        for k in self.names:
            r = [x for x in self.s["rounds"] if x["name"] == k]
            out.append({"key": k, "name": self.label[k], "examples": len(self.pos[k]), "not_them": len(self.neg.get(k, ())),
                        "photos": len(photos.get(k, ())), "rounds": len(r),
                        "teach_right": sum(x["right_above"] for x in r), "teach_wrong": sum(x["wrong_above"] for x in r)})
        allp = set.intersection(*photos.values()) if len(photos) >= 2 and len(photos) == len(self.names) else set()
        return {"people": out, "all_together_photos": len(allp), "threshold": self.TH, "model": self.D["stats"].get("model")}

    def ask(self, k, n=36):
        """Faces it would call k (or almost would), not yet confirmed or rejected: least sure first."""
        j = self.names.index(k); judged = self.pos[k] | self.neg.get(k, set()) | self.hidden
        c = [i for i in np.where(self.E[j] >= self.TH - 0.08)[0] if i not in judged and (self.best[i] == j or self.score[i] < self.TH)]
        c.sort(key=lambda i: abs(self.E[j][i] - self.TH)); c = c[:n]
        return [{"id": int(i), "score": round(float(self.E[j][i]), 3), "above": bool(self.E[j][i] >= self.TH)} for i in c]


PAGE = r"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Who is this?</title><style>
:root{--bg:#070b14;--card:#0f1626;--line:#1f2a40;--fg:#eef2f8;--mut:#8f9bb3;--ok:#c6f47b;--bad:#ff5d6c;--acc:#3b82f6}
body{background:var(--bg);color:var(--fg);font:15px -apple-system,system-ui,sans-serif;margin:0;padding:16px 20px 80px}
h1{margin:0 0 4px;font-size:26px}h2{margin:18px 0 6px}.mut{color:var(--mut)}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);padding:10px 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center;border-bottom:1px solid var(--line)}
button{background:linear-gradient(90deg,#3b82f6,#7c3aed);color:#fff;border:0;border-radius:10px;padding:10px 16px;font-weight:800;font-size:14px;cursor:pointer}
button.ghost{background:#16203a;color:var(--fg);font-weight:600}button.del{background:#2a1520;color:#ffb3bb;font-weight:600}
.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:7px 12px;cursor:pointer}.chip b{color:var(--ok)}
.g{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px;margin:12px 0}
.g.hid{display:none}.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.g input{background:#0b1120;border:1px solid #2a3a5c;color:#fff;border-radius:9px;padding:9px;font-size:15px;width:190px}
.faces{display:flex;flex-wrap:wrap;gap:6px;max-height:310px;overflow:auto}
.f{position:relative;width:76px;height:76px;border-radius:10px;overflow:hidden;cursor:pointer;border:2px solid transparent}
.f img{width:100%;height:100%;object-fit:cover;display:block}.f.no{border-color:var(--bad)}.f.no img{opacity:.28;filter:grayscale(1)}
.f.no:after{content:"✕";position:absolute;inset:0;display:grid;place-items:center;color:var(--bad);font-size:38px;font-weight:900}
.f .s{position:absolute;left:3px;bottom:2px;font-size:10px;background:#000a;border-radius:4px;padding:0 3px}
#teach{display:none;background:#0b1428;border:1px solid #2a3a5c;border-radius:14px;padding:14px;margin:14px 0}
#photos{display:none}.pg{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}.pg img{width:100%;border-radius:10px}
</style>
<h1>Who is this?</h1><div class=mut>Name a group. Click any face that is <b>not</b> that person (it gets a red ✕). Hide groups you don't care about.
Then press a name to <b>teach</b> it more. Everything stays on this Mac.</div>
<div class=bar><button id=save>Save</button><span id=ppl></span><span class=mut id=sum></span></div>
<div id=teach><h2>Is this <span id=tn></span>?</h2><div class=mut>The faces it is least sure about. Click every one that is NOT them, then Confirm:
the rest become examples of them, the crossed ones teach it who they are not.</div>
<div class=faces id=tf style="max-height:none;margin:10px 0"></div><button id=ok>Confirm</button> <button class=ghost id=tx>Close</button> <span class=mut id=tmsg></span></div>
<div id=photos><h2 id=pn></h2><button class=ghost id=px>Close</button><div class=pg id=pg style="margin-top:10px"></div></div>
<div id=list></div><button class=ghost id=more>Show more groups</button><button class=ghost id=showhid>Show hidden groups</button>
<script>
let G=[],S={},shown=80,showHidden=false,cur=null;
const off=g=>new Set(S.off[g]||[]);
function face(id,cls,extra){return `<div class="f ${cls}" data-id=${id}><img loading=lazy src="/crop/${id}">${extra||''}</div>`}
function draw(){
  const hid=new Set(S.hidden);
  list.innerHTML=G.slice(0,shown).map((g,i)=>{const o=off(i);return `<div class="g ${hid.has(i)&&!showHidden?'hid':''}" data-g=${i}><div class=top>
  <input placeholder="name (or leave empty)" value="${(S.names[i]||'').replace(/"/g,'&quot;')}"><span class=mut>${g.length} faces · ${o.size} crossed</span>
  <button class=ghost data-a=flip>Cross all / none</button><button class=del data-a=hide>${hid.has(i)?'Unhide':'Hide group'}</button></div>
  <div class=faces>${g.map(id=>face(id,o.has(id)?'no':'')).join('')}</div></div>`}).join('');
  more.style.display=shown<G.length?'':'none';
}
function collect(){document.querySelectorAll('.g').forEach(e=>{const g=e.dataset.g,v=e.querySelector('input').value.trim();if(v)S.names[g]=v;else delete S.names[g]})}
list.onclick=e=>{const gEl=e.target.closest('.g');if(!gEl)return;const g=+gEl.dataset.g,f=e.target.closest('.f'),a=e.target.dataset.a;
  if(f){f.classList.toggle('no');S.off[g]=[...gEl.querySelectorAll('.f.no')].map(x=>+x.dataset.id);gEl.querySelector('.top .mut').textContent=`${G[g].length} faces · ${S.off[g].length} crossed`}
  if(a=='flip'){const all=S.off[g]&&S.off[g].length==G[g].length;S.off[g]=all?[]:[...G[g]];collect();draw()}
  if(a=='hide'){collect();const h=new Set(S.hidden);h.has(g)?h.delete(g):h.add(g);S.hidden=[...h];draw()}};
async function doSave(){collect();await fetch('/state',{method:'POST',body:JSON.stringify(S)});save.textContent='Saved ✓';setTimeout(()=>save.textContent='Save',1500);people()}
save.onclick=doSave;
more.onclick=()=>{collect();shown+=80;draw()};showhid.onclick=()=>{collect();showHidden=!showHidden;showhid.textContent=showHidden?'Hide hidden groups':'Show hidden groups';draw()};
async function people(){const P=await (await fetch('/people')).json();
  ppl.innerHTML=P.people.map(p=>`<span class=chip data-k="${p.key}">${p.name} <b>${p.photos}</b> photos · <a data-p="${p.key}" style="color:#9ec5ff">see</a></span>`).join(' ');
  sum.textContent=P.people.length>1?`all together: ${P.all_together_photos} photos`:''}
ppl.onclick=e=>{const p=e.target.dataset.p;if(p)return showPhotos(p);const c=e.target.closest('.chip');if(c)teachName(c.dataset.k)};
async function teachName(k){await doSave();cur=k;const R=await (await fetch('/ask?name='+encodeURIComponent(k))).json();teach.style.display='block';tn.textContent=R.name;
  tf.innerHTML=R.faces.length?R.faces.map(f=>face(f.id,'',`<span class=s>${f.score}</span>`)).join(''):'<span class=mut>Nothing left to ask. It knows this person on these photos.</span>';
  tmsg.textContent='';window.scrollTo(0,0)}
tf.onclick=e=>{const f=e.target.closest('.f');if(f)f.classList.toggle('no')};
ok.onclick=async()=>{const yes=[...tf.querySelectorAll('.f:not(.no)')].map(x=>+x.dataset.id),no=[...tf.querySelectorAll('.f.no')].map(x=>+x.dataset.id);
  const r=await (await fetch('/teach',{method:'POST',body:JSON.stringify({name:cur,yes,no})})).json();S=r.state;people();await teachName(cur);tmsg.textContent=`learned: ${yes.length} yes, ${no.length} no`};
tx.onclick=()=>teach.style.display='none';
async function showPhotos(k){const R=await (await fetch('/photos?name='+encodeURIComponent(k))).json();photos.style.display='block';pn.textContent=`${R.name}: ${R.faces.length} photos`;
  pg.innerHTML=R.faces.map(id=>`<img loading=lazy src="/photo/${id}">`).join('');window.scrollTo(0,0)}
px.onclick=()=>photos.style.display='none';
Promise.all([fetch('/groups').then(r=>r.json()),fetch('/state').then(r=>r.json())]).then(([g,s])=>{G=g;S=s;draw();people()});
</script>"""


def serve(port=8770):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs
    import io
    D = json.load(open(HOME / "faces.json")); S = {"s": state()}

    def put(s):
        S["s"] = s; json.dump(s, open(HOME / "names.json", "w"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def send(self, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(200); self.send_header("content-type", ctype); self.send_header("content-length", str(len(b)))
            self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/": return self.send(PAGE, "text/html")
            if u.path == "/groups": return self.send([g for g in D["groups"] if len(g) >= 2])
            if u.path == "/state": return self.send(S["s"])
            if u.path == "/people": return self.send(Brain(D, S["s"]).people())
            if u.path == "/ask":
                b = Brain(D, S["s"]); k = key(q["name"])
                return self.send({"name": b.label.get(k, k), "faces": b.ask(k) if k in b.names else []})
            if u.path == "/photos":
                b = Brain(D, S["s"]); k = key(q["name"]); seen, out = set(), []
                for i in sorted(range(len(D["faces"])), key=lambda i: -b.score[i]):
                    if b.who(i) == k and D["faces"][i]["photo"] not in seen: seen.add(D["faces"][i]["photo"]); out.append(i)
                return self.send({"name": b.label.get(k, k), "faces": out})
            if u.path.startswith("/crop/"): return self.send((CROPS / (u.path[6:] + ".jpg")).read_bytes(), "image/jpeg")
            if u.path.startswith("/photo/"):                     # the whole photo, with the face it matched framed
                f = D["faces"][int(u.path[7:])]; im = load(Path(f["photo"]), 1600); w0 = f.get("size", [im.width])[0]
                from PIL import ImageDraw
                k = im.width / w0; x, y, w, h = [v * k for v in f["box"]]; ImageDraw.Draw(im).rectangle([x, y, x + w, y + h], outline=(198, 244, 123), width=5)
                im.thumbnail((480, 480)); bio = io.BytesIO(); im.save(bio, "JPEG", quality=82); return self.send(bio.getvalue(), "image/jpeg")
            self.send_error(404)
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/state":
                s = state(); s.update({k: body[k] for k in ("names", "off", "hidden") if k in body}); put(s); return self.send({"ok": 1})
            if self.path == "/teach":
                s = S["s"]; k = key(body["name"]); b = Brain(D, s); j = b.names.index(k)
                above = {i for i in body["yes"] + body["no"] if b.E[j][i] >= b.TH}
                s["rounds"].append({"name": k, "yes": len(body["yes"]), "no": len(body["no"]), "t": round(time.time()),
                                    "right_above": sum(i in above for i in body["yes"]), "wrong_above": sum(i in above for i in body["no"])})
                s["yes"].setdefault(k, []).extend(body["yes"]); s["no"].setdefault(k, []).extend(body["no"]); put(s)
                return self.send({"state": s})
            self.send_error(404)

    print(f"http://127.0.0.1:{port}  (Ctrl-C to stop)", flush=True); ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def find():
    """Numbers only: what it knows per name, and how right its own guesses were in the teach rounds."""
    D = json.load(open(HOME / "faces.json")); P = Brain(D, state()).people()
    for p in P["people"]:
        n = p["teach_right"] + p["teach_wrong"]
        p["guesses_right"] = f'{p["teach_right"]}/{n}' if n else "not taught yet"
        del p["key"]
    print(json.dumps({"scan": D["stats"], **P}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "scan":
        a = sys.argv[2:]; m = "sface" if "--sface" in a else "arcface"; scan([x for x in a if not x.startswith("--")], m)
    elif cmd == "serve": serve()
    elif cmd == "find": find()
    else: print(__doc__)
