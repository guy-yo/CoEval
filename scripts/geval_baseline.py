"""G-Eval baseline (single GPT-4o + chain-of-thought) vs the CoEval ensemble.

Faithful-enough reimplementation of G-Eval (Liu et al., 2023): a single strong
judge scores each summary with explicit chain-of-thought evaluation steps, then
emits an integer quality score. We correlate G-Eval scores with the native
BERTScore metric on the EXP-001 summarization responses and contrast with the
CoEval cross-family ensemble and the best single CoEval judge.

Usage:  python scripts/geval_baseline.py [--limit N]
Cost:   ~600 gpt-4o calls (~$3 on OpenRouter); real-time.
"""
from __future__ import annotations
import argparse, json, glob, re, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "Runs" / "EXP001-benchmark-grounded-comparison"
KEY = yaml.safe_load(open(ROOT / "keys.yaml"))["providers"]["openrouter"]

GEVAL_PROMPT = """You are an expert evaluator of text summaries. You will be given a source text and a candidate summary.

Evaluation Steps:
1. Read the source text carefully and identify its main facts.
2. Read the candidate summary.
3. Assess: (a) faithfulness (no hallucinated or incorrect facts), (b) coverage of the most important information, (c) conciseness and fluency.
4. Assign an overall summary-quality score from 1 (very poor) to 5 (excellent).

Source text:
{source}

Candidate summary:
{summary}

Reason briefly through the evaluation steps, then end with exactly one line:
SCORE: <integer 1-5>"""


def _call(source: str, summary: str) -> float | None:
    body = json.dumps({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user",
                      "content": GEVAL_PROMPT.format(source=source[:6000], summary=summary[:2000])}],
        "temperature": 0.0, "max_tokens": 400,
    }).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=90))
            txt = r["choices"][0]["message"]["content"]
            m = re.search(r"SCORE:\s*([1-5])", txt)
            if m:
                return float(m.group(1))
            m2 = re.findall(r"\b([1-5])\b", txt[-40:])
            return float(m2[-1]) if m2 else None
        except Exception:
            time.sleep(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dps = {}
    for f in glob.glob(str(RUN / "phase3_datapoints" / "*summ*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); dps[r["id"]] = r.get("prompt", "")
    bench = {}
    for l in open(RUN / "benchmark_response_scores.jsonl", encoding="utf-8"):
        b = json.loads(l)
        if b["metric"] == "bertscore":
            bench[b["response_id"]] = b["benchmark_native_score"]

    items = []
    for f in glob.glob(str(RUN / "phase4_responses" / "*summ*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in bench:
                items.append((r["id"], dps.get(r["datapoint_id"], ""), r.get("response", "")))
    if args.limit:
        items = items[:args.limit]
    print(f"Scoring {len(items)} summaries with G-Eval (gpt-4o CoT)...", flush=True)

    scores = {}
    def work(it):
        rid, src, summ = it
        return rid, _call(src, summ)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (rid, sc) in enumerate(ex.map(work, items)):
            if sc is not None:
                scores[rid] = sc
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(items)}", flush=True)

    from analyzer.stats import correlation_ci
    common = [rid for rid in scores if rid in bench]
    g = [scores[r] for r in common]; bs = [bench[r] for r in common]
    e = correlation_ci(g, bs, method="spearman", seed=0)
    out = {"baseline": "G-Eval (gpt-4o CoT, single judge)", "n": len(common),
           "spearman_vs_bertscore": round(e.point, 4), "lo": round(e.lo, 4), "hi": round(e.hi, 4),
           "coeval_ensemble_vs_bertscore": 0.241, "coeval_best_single_judge": 0.354,
           "score_dist": {int(k): g.count(k) for k in sorted(set(g))}}
    (RUN / "reports" / "geval_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "Code"))
    main()
