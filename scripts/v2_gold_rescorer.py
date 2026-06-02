"""Format-robust gold scorer for the EXP-010 science_qa MCQ task.

The raw inclusion_match metric is output-format sensitive: a model that answers
with the option LETTER ("C") scores 0 against a TEXT reference ("fossilization")
even when the letter is correct. This rescorer parses the option block from each
datapoint prompt, resolves the correct option (the one whose text matches the
reference), and credits a student response if it names the correct option by
EITHER its letter or its text. Writes benchmark_response_scores.jsonl (overwrite).

Run:  python scripts/v2_gold_rescorer.py Runs/EXP010-scale-ranking-pilot
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


def _parse_options(prompt: str):
    """Return {letter: text} parsed from an '(A) ... (B) ...' option block."""
    opts = {}
    for m in re.finditer(r"\(([A-Z])\)\s*([^\n(]+)", prompt):
        opts[m.group(1).upper()] = m.group(2).strip().rstrip(".").strip()
    return opts


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _correct_letter(opts: dict, ref: str):
    rn = _norm(ref)
    for L, t in opts.items():
        if _norm(t) == rn:
            return L
    # fallback: substring
    for L, t in opts.items():
        if rn and (rn in _norm(t) or _norm(t) in rn):
            return L
    return None


def _is_correct(resp: str, opts: dict, correct_L, ref: str) -> float:
    r = resp.strip()
    rn = _norm(r)
    # 1) explicit letter mention, e.g. "C", "(C)", "answer: C", "C)"
    lm = re.findall(r"\b([A-Z])\b", r[:40])
    if correct_L and lm and lm[0].upper() == correct_L:
        return 1.0
    # 2) text inclusion: the correct option text appears in the response
    if _norm(ref) and _norm(ref) in rn:
        return 1.0
    if correct_L and _norm(opts.get(correct_L, "")) and _norm(opts[correct_L]) in rn:
        return 1.0
    # 3) a WRONG letter clearly stated -> incorrect
    if correct_L and lm and lm[0].upper() in opts and lm[0].upper() != correct_L:
        return 0.0
    # 4) some other option's text appears (and not the correct one) -> incorrect
    for L, t in opts.items():
        if L != correct_L and _norm(t) and _norm(t) in rn:
            return 0.0
    return 0.0


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "Runs/EXP010-scale-ranking-pilot")
    dps = {}
    for f in (run / "phase3_datapoints").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                dps[r["id"]] = {"opts": _parse_options(r.get("prompt", "")),
                                "ref": r.get("reference_response", ""),
                                "task_id": r.get("task_id", "")}
    for dp in dps.values():
        dp["correct_L"] = _correct_letter(dp["opts"], dp["ref"])

    out_lines, n, hit = [], 0, 0
    for f in (run / "phase4_responses").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            dp = dps.get(rec.get("datapoint_id"))
            if not dp:
                continue
            score = _is_correct(rec.get("response", ""), dp["opts"], dp["correct_L"], dp["ref"])
            n += 1
            hit += score
            out_lines.append(json.dumps({
                "response_id": rec["id"],
                "datapoint_id": rec["datapoint_id"],
                "task_id": "science_qa",
                "student_model_id": rec["student_model_id"],
                "metric": "mcq_robust_match",
                "benchmark_native_score": score,
            }))
    (run / "benchmark_response_scores.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"rescored {n} responses, mean gold acc = {hit/n:.3f}")


if __name__ == "__main__":
    main()
