#!/usr/bin/env python3
"""EpiCache conv-QA accuracy harness — driver.

Launches `llama-server` under one of four *arms*, feeds each conversation's
long multi-session context as a system prompt, asks the benchmark's questions,
and scores answer accuracy (EM / F1 / contains via score.py).

ARMS (each is purely a server-launch configuration — same binary, different
flags/env, so accuracy differences isolate the KV policy):

  full         reference. No compression. Full attention over the whole context.
               (`-fa on`)
  p1           MERGED EpiCache P1: block-wise prefill peak-mem bounding. Reuses
               the TriAttention FA-safe proxy scorer. (`--triattention` +
               LLAMA_EPICACHE_PREFILL=1 LLAMA_EPICACHE_BUDGET=M, `-fa on`)
  plain-evict  TriAttention decode/prefill eviction WITHOUT the EpiCache prefill
               budget — i.e. plain recent-window + top-K + sink eviction, the
               "cheap eviction" reference. (`--triattention`, no epicache env)
  eq3          FAITHFUL EpiCache Eq.3 importance scorer (materialized
               softmax(QK^T), max-over-query). NON-FA path. *NOT YET
               IMPLEMENTED* — this is P3 work (see IMPL-PLAN.md). The arm is
               defined here so a later impl worker only flips a flag; running it
               today requires --allow-unimplemented, which executes a full-attn
               placeholder with `-fa off` to exercise the manual-softmax plumbing.

Output: a results JSON (per-question predictions + scores) and a printed summary
table (overall + per-category EM/F1). The same JSON across arms is diffed by
summarize.py to produce the P1-vs-full accuracy-gap table.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as scorer  # noqa: E402

SYSTEM_TMPL = (
    "You are a helpful assistant answering questions about a long conversation "
    "between two people. Read the conversation below carefully, then answer the "
    "question using ONLY information from the conversation. Answer concisely with "
    "just the fact (a few words). If the answer is not in the conversation, say "
    "\"No information available\".\n\n"
    "CONVERSATION:\n{context}\n"
)

# TriAttention proxy-scorer flags shared by the p1 and plain-evict arms.
# (Matches the calibration used by the merged P1 commit's smoke/eval scripts.)
TRIA_FLAGS = ["--tri-budget", "50", "--tri-window", "256",
              "--tri-sink", "128", "--tri-interval", "1"]


def arm_config(arm, tria_path, budget):
    """Return (extra_server_args, extra_env) for the given arm."""
    env = {}
    args = ["-fa", "on"]
    if arm == "full":
        return args, env
    if arm == "p1":
        if not tria_path:
            sys.exit("arm 'p1' requires --tria <stats.tria>")
        args = ["--triattention", tria_path, *TRIA_FLAGS, "-fa", "on"]
        env = {"LLAMA_EPICACHE_PREFILL": "1", "LLAMA_EPICACHE_BUDGET": str(budget)}
        return args, env
    if arm == "plain-evict":
        if not tria_path:
            sys.exit("arm 'plain-evict' requires --tria <stats.tria>")
        args = ["--triattention", tria_path, *TRIA_FLAGS, "-fa", "on"]
        return args, env
    if arm == "eq3":
        # Faithful Eq.3 scorer is P3 — not yet wired. Placeholder exercises the
        # non-FA manual-softmax path so the harness plumbing is validated.
        args = ["-fa", "off"]
        return args, env
    sys.exit(f"unknown arm {arm!r}")


def wait_health(base, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def chat(base, system, user, max_tokens, ctx_guard):
    body = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "cache_prompt": False,  # arm-agnostic: every question re-prefills the context
        "stream": False,
        # Disable reasoning ("thinking") models' chain-of-thought so the answer
        # lands in `content`, not `reasoning_content` (e.g. Qwen3.x). Harmless on
        # non-reasoning models (unknown kwargs are ignored by the template).
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=ctx_guard) as r:
        resp = json.loads(r.read())
    dt = time.time() - t0
    msg = resp["choices"][0]["message"]
    # Fall back to reasoning_content if a reasoning model still emptied content.
    text = (msg.get("content") or "").strip() or (msg.get("reasoning_content") or "").strip()
    usage = resp.get("usage", {})
    return text, dt, usage


def launch_server(binary, model, arm, tria, budget, host, port, ctx, ngl, threads, log_path):
    extra_args, extra_env = arm_config(arm, tria, budget)
    cmd = [
        binary, "-m", model,
        "--host", host, "--port", str(port),
        "-c", str(ctx), "-b", "512", "-ub", "512",
        "-ngl", str(ngl), "-t", str(threads),
        "--no-warmup", *extra_args,
    ]
    env = dict(os.environ)
    env.update(extra_env)
    logf = open(log_path, "w")
    logf.write(f"# CMD: {' '.join(cmd)}\n# ENV: {extra_env}\n")
    logf.flush()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return proc, logf, cmd, extra_env


def run(args):
    if args.arm == "eq3" and not args.allow_unimplemented:
        sys.exit("arm 'eq3' (faithful Eq.3) is NOT YET IMPLEMENTED (P3). "
                 "Re-run with --allow-unimplemented to exercise the placeholder.")

    subset = json.load(open(args.subset))
    base = f"http://{args.host}:{args.port}"
    log_path = args.out.replace(".json", ".server.log")
    proc, logf, cmd, extra_env = launch_server(
        args.bin, args.model, args.arm, args.tria, args.budget,
        args.host, args.port, args.ctx, args.ngl, args.threads, log_path)

    results = {
        "arm": args.arm, "model": os.path.basename(args.model),
        "benchmark": subset.get("benchmark"), "budget": args.budget,
        "server_cmd": cmd, "server_env": extra_env, "items": [],
    }
    try:
        if not wait_health(base, timeout=args.health_timeout):
            proc.terminate()
            sys.exit(f"server did not become healthy; see {log_path}")
        print(f"[{args.arm}] server up. running {sum(len(c['questions']) for c in subset['conversations'])} questions...")
        for c in subset["conversations"]:
            system = SYSTEM_TMPL.format(context=c["context"])
            for q in c["questions"]:
                try:
                    pred, dt, usage = chat(base, system, q["question"],
                                           args.max_tokens, args.req_timeout)
                except Exception as e:
                    pred, dt, usage = f"<ERROR: {e}>", 0.0, {}
                s = scorer.score_one(pred, q["answer"], q.get("category"))
                results["items"].append({
                    "conv_id": c["conv_id"], "qid": q["qid"],
                    "category": q.get("category"), "question": q["question"],
                    "gold": q["answer"], "pred": pred,
                    "em": s["em"], "f1": s["f1"], "contains": s["contains"],
                    "adversarial": s["adversarial"],
                    "latency_s": round(dt, 2),
                    "prompt_tokens": usage.get("prompt_tokens"),
                })
                print(f"  [{q.get('category')}] f1={s['f1']:.2f} em={s['em']:.0f} "
                      f"({dt:.1f}s) {q['question'][:55]!r} -> {pred[:45]!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        logf.close()

    json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    _print_summary(results)
    print(f"\nwrote {args.out}")
    return 0


def _print_summary(results):
    items = results["items"]
    if not items:
        print("no items scored"); return
    def agg(rows):
        n = len(rows)
        return (n,
                sum(r["em"] for r in rows) / n,
                sum(r["f1"] for r in rows) / n,
                sum(r["contains"] for r in rows) / n)
    n, em, f1, ct = agg(items)
    print(f"\n=== arm={results['arm']} model={results['model']} "
          f"budget={results['budget']} ===")
    print(f"{'category':<14}{'n':>4}{'EM':>8}{'F1':>8}{'contains':>10}")
    bycat = {}
    for r in items:
        bycat.setdefault(r["category"], []).append(r)
    for cat in sorted(bycat):
        cn, cem, cf1, cct = agg(bycat[cat])
        print(f"{('cat'+str(cat)):<14}{cn:>4}{cem:>8.3f}{cf1:>8.3f}{cct:>10.3f}")
    print(f"{'OVERALL':<14}{n:>4}{em:>8.3f}{f1:>8.3f}{ct:>10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="llama-server binary path")
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--subset", required=True, help="subset json from prepare_subset.py")
    ap.add_argument("--arm", required=True,
                    choices=["full", "p1", "plain-evict", "eq3"])
    ap.add_argument("--tria", default="", help=".tria stats (p1/plain-evict arms)")
    ap.add_argument("--budget", type=int, default=1024, help="EpiCache prefill budget M")
    ap.add_argument("--out", required=True, help="results json path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--ctx", type=int, default=24576,
                    help="server context size; MUST exceed the longest conversation "
                         "(EpiCache bounds resident KV via --budget, not n_ctx)")
    ap.add_argument("--ngl", type=int, default=0, help="GPU layers (0=CPU)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--health-timeout", type=int, default=180)
    ap.add_argument("--req-timeout", type=int, default=600)
    ap.add_argument("--allow-unimplemented", action="store_true")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
