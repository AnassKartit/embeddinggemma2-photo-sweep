"""Codex image tool (ChatGPT sub, $0) -> PNG. Prompt on stdin, refs via -i. Usage: codex_gen.py jobs.json [workers]
jobs.json: [{"out": "...png", "prompt": "...", "refs": [...]}]. Skips outputs that exist; logs seconds per image."""
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
CODEX = os.environ.get("CODEX_BIN", "codex")          # any Codex CLI signed in with a ChatGPT plan

def gen(job):
    out = Path(job["out"]).resolve()
    if out.exists(): return {"out": str(out), "skipped": True}
    ask = (f"Use your image generation tool to create ONE new image, then save a copy of it as {out} and print its path. "
           f"{job['prompt']}")
    cmd = [CODEX, "exec", "--skip-git-repo-check", "-C", str(out.parent)]
    for r in job.get("refs", []): cmd += ["-i", str(Path(r).expanduser())]
    t = time.time()
    p = subprocess.run(cmd, input=ask, text=True, capture_output=True, timeout=900)
    r = {"out": str(out), "ok": out.exists(), "s": round(time.time() - t, 1)}
    if not out.exists(): r["err"] = (p.stdout + p.stderr)[-400:]
    print(json.dumps(r), flush=True)
    return r

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1])); w = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with ThreadPoolExecutor(w) as ex: list(ex.map(gen, jobs))
