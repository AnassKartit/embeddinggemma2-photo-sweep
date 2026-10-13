"""Talk to your files: EmbeddingGemma 2 finds the files (rag/ask.py), a small local model reads only those and answers in one
sentence, naming the file. All on this Mac (MLX).

Two steps so only one model is in memory at a time (24 GB Mac):
  1. .venv/bin/python rag/answer.py --contexts        EmbeddingGemma 2 picks the top 3 files for each question -> results/answer_contexts.json
  2. .venv/bin/python rag/answer.py --eval <mlx repo>  that model answers from the saved files -> results/answer_<model>.json
Eval: 11 questions whose answer is a fact inside a document (one: the latest electricity bill, from the dates), right answers written down first; a hit = the expected text is in
the answer. Run step 2 under localfit/scripts/memwatch.sh."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent; FOLDER = ROOT / "demo/Downloads"; CTX = ROOT / "results/answer_contexts.json"
QA = [("how much was my latest electricity bill?", ["131.44"]), ("how much was my internet bill?", ["79.99"]), ("when does my lease end?", ["31 January 2027", "January 31, 2027"]),
      ("how much do I owe the dentist?", ["185"]), ("what seat am I on my flight to New York?", ["14A"]),
      ("when is my doctor appointment?", ["4 November", "November 4"]), ("what's the guest wifi network called?", ["CarterGuest"]),
      ("when is the school winter break?", ["21 December", "December 21"]), ("how much was my tax refund?", ["1,120", "1120"]),
      ("what's my monthly rent?", ["2,150", "2150"]), ("how long do I bake the lasagna?", ["45"])]
SYSTEM = ("Answer the question in one short sentence using only the files below. Name the file you used. Each file shows when it was "
          "last modified: for 'latest' or 'newest', pick the most recent one. If the files do not say, answer: not in your files.")


def context_of(folder, h):
    """One file for the answer model: name, modified date, and its text (documents only)."""
    import ask
    p = folder / h["name"]; when = time.strftime("%d %B %Y", time.localtime(p.stat().st_mtime))
    body = ask.text_of(p)[:1500] if h["kind"] == "document" else f"({h['kind']} file, no text)"
    return f"FILE: {h['name']} (modified {when})\n{body}"


def photo_answer(hits):
    """A photo has no text to read: when the best match is a photo, point to it directly (no language model)."""
    return f"Here it is: {hits[0]['name']}" if hits and hits[0]["kind"] == "image" else None

def contexts():
    import ask
    idx = ask.Index(ask.load_model(), FOLDER); out = []
    for q, want in QA:
        hits, ms = idx.search(q, k=3)
        ctx = [context_of(FOLDER, h) for h in hits]
        out.append({"q": q, "want": want, "files": [h["name"] for h in hits], "search_ms": round(ms, 1), "context": "\n\n".join(ctx)})
    json.dump(out, open(CTX, "w"), indent=1); print("saved", CTX)

def answer_eval(repo):
    from mlx_lm import load, generate
    import mlx.core as mx
    t = time.time(); model, tok = load(repo); load_s = time.time() - t; rows = []
    for c in json.load(open(CTX)):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": c["context"] + f"\n\nQUESTION: {c['q']}"}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)  # Qwen 3.x think by default and can spend the whole budget
        t = time.time(); out = generate(model, tok, prompt=prompt, max_tokens=80, verbose=False).strip(); s = time.time() - t
        ok = any(w.lower() in out.lower() for w in c["want"])
        rows.append({"q": c["q"], "answer": out, "ok": ok, "answer_s": round(s, 2)}); print("OK  " if ok else "MISS", c["q"], "->", out[:120], f"({s:.1f} s)", flush=True)
    res = {"llm": repo, "load_s": round(load_s, 1), "peak_gb": round(mx.get_peak_memory() / 1e9, 2), "correct": sum(r["ok"] for r in rows), "of": len(rows),
           "median_answer_s": sorted(r["answer_s"] for r in rows)[len(rows) // 2], "rows": rows}
    json.dump(res, open(ROOT / f"results/answer_{repo.split('/')[-1]}.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})

if __name__ == "__main__":
    if "--contexts" in sys.argv: contexts()
    elif "--eval" in sys.argv: answer_eval(sys.argv[sys.argv.index("--eval") + 1])
