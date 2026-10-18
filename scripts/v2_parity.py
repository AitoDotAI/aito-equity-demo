"""Replay every Aito query shape the demo sends against /api/v1 and /api/v2, diff the answers.

Read-only. Each case is a (v1 request, v2 request) pair — the v2 request is the
ported spelling the demo would send after migrating (e.g. `_similarity` →
`_query` ordered by `$similarity`). By default both hit `master`; pass
`--v2-env v2` to send the v2 side to a branched env whose tables have been
migrated to the v2 storage engine (`_migrate`), which is the real comparison.

Every incompatibility in prior demo migrations came back as HTTP 200 with a
different payload, so this compares payloads, not status codes.

    uv run python scripts/v2_parity.py                 # v1 vs v2 API, same (master) storage
    uv run python scripts/v2_parity.py --v2-env v2     # v1@master vs v2@env/v2 (migrated)
    → docs/v2-parity.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)
URL = os.environ["AITO_API_URL"].rstrip("/")
H = {"x-api-key": os.environ["AITO_API_KEY"]}

NVDA = {"market_position": "dominant", "moat_type": "intangible_assets", "moat_strength": "wide",
        "market_quality": "secular_growth", "leadership_quality": 5, "sector": "Information Technology",
        "valuation_bucket": "expensive", "growth_bucket": "high", "momentum_bucket": "strong"}
DOG = {"market_quality": "disrupted", "leverage_bucket": "high", "valuation_bucket": "cheap", "moat_strength": "narrow"}
SIM = {k: NVDA[k] for k in ["market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality", "sector"]}
REL = ["market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality", "sector",
       "valuation_bucket", "growth_bucket", "leverage_bucket"]
F7 = ["market_position", "moat_type", "moat_strength", "market_quality", "leadership_quality", "sector"]


def pred(where, target, **kw):
    return {"from": "companies", "where": where, "predict": target, "select": ["feature", "$p", "$why"],
            "config": {"ai": "high"}, **kw}


def pred2(where, target, **kw):
    return {"from": "companies", "where": where, "predict": target, "select": ["$value", "$p", "$why"],
            "config": {"ai": "high"}, **kw}


EVAL = {"test": {"eval_fold": "0"},
        "evaluate": {"from": "companies", "where": {f: {"$get": f} for f in F7}, "predict": "outcome_bucket"}}

# (name, where in the demo, v1 endpoint, v1 body, v2 endpoint, v2 body)
CASES = [
    ("lab_predict_longrun", "Company Lab + Company File + backtest/error/value-trap emitters",
     "_predict", pred(NVDA, "outcome_bucket"), "_predict", pred2(NVDA, "outcome_bucket")),
    ("lab_predict_longrun_v1body", "same, v1 body sent verbatim to /api/v2 (does an unported client survive?)",
     "_predict", pred(NVDA, "outcome_bucket"), "_predict", pred(NVDA, "outcome_bucket")),
    ("lab_predict_6m", "Company Lab 6-month card",
     "_predict", pred(NVDA, "fwd_6m_bucket"), "_predict", pred2(NVDA, "fwd_6m_bucket")),
    ("predict_trap_profile", "value-trap view (cheap, indebted, disrupted)",
     "_predict", pred(DOG, "outcome_bucket"), "_predict", pred2(DOG, "outcome_bucket")),
    ("predict_empty_where", "Lab with nothing set (base rate)",
     "_predict", pred({}, "outcome_bucket"), "_predict", pred2({}, "outcome_bucket")),
    ("relate_great", "Thesis view headline relate",
     "_relate", {"from": "companies", "where": {"outcome_bucket": "great"}, "relate": REL, "limit": 20},
     "_relate", {"from": "companies", "where": {"outcome_bucket": "great"}, "relate": REL, "limit": 20}),
    ("similarity_profile", "Lab analogues + Historical Analogues view",
     "_similarity", {"from": "companies", "similarity": SIM, "limit": 10},
     "_query", {"from": "companies", "where": SIM, "orderBy": "$similarity", "limit": 10,
                "select": ["ticker", "vintage_year", "company_name", "outcome_bucket", "$similarity"]}),
    ("evaluate_fold0", "Evaluation view (eval_aito / emit_eval)",
     "_evaluate", EVAL, "_evaluate", EVAL),
    ("relate_great_scoped", "v2 idiom: population = rows whose (nullable) outcome is defined",
     "_relate", {"from": "companies", "where": {"outcome_bucket": "great"}, "relate": REL, "limit": 20},
     "_relate", {"from": "companies", "where": {"$on": [{"outcome_bucket": "great"}, {"outcome_bucket": {"$defined": True}}]},
                 "relate": REL, "limit": 20}),
    ("relate_not_null", "population via $not null (G1 / aito-core#1061 repro)",
     "_relate", {"from": "companies", "where": {"$on": [{"outcome_bucket": "great"}, {"$not": {"outcome_bucket": None}}]}, "relate": REL, "limit": 5},
     "_relate", {"from": "companies", "where": {"$on": [{"outcome_bucket": "great"}, {"$not": {"outcome_bucket": None}}]}, "relate": REL, "limit": 5}),
    ("console_predict", "Live Query sample (predict, no select)",
     "_predict", {"from": "companies", "where": {"market_position": "dominant", "moat_strength": "wide",
                  "market_quality": "secular_growth", "founder_still_ceo": True}, "predict": "outcome_bucket"},
     "_predict", {"from": "companies", "where": {"market_position": "dominant", "moat_strength": "wide",
                  "market_quality": "secular_growth", "founder_still_ceo": True}, "predict": "outcome_bucket"}),
    ("console_relate_features", "Live Query sample (relate: $features)",
     "_relate", {"from": "companies", "where": {"outcome_bucket": "great"}, "relate": "$features", "limit": 12},
     "_relate", {"from": "companies", "where": {"outcome_bucket": "great"}, "relate": "$features", "limit": 12}),
    ("console_match", "Live Query sample (match)",
     "_match", {"from": "companies", "match": {"ticker": "NVDA", "vintage_year": 2014}, "limit": 6},
     "_match", {"from": "companies", "match": {"ticker": "NVDA", "vintage_year": 2014}, "limit": 6}),
    ("earnings_predict", "/api/classify-earnings",
     "_predict", {"from": "earnings_events", "where": {"guidance": "raised", "tone": "positive"}, "predict": "react_1d_bucket",
                  "config": {"ai": "high"}},
     "_predict", {"from": "earnings_events", "where": {"guidance": "raised", "tone": "positive"}, "predict": "react_1d_bucket",
                  "config": {"ai": "high"}}),
]


def call(c: httpx.Client, base: str, ver: str, ep: str, body: dict) -> dict:
    t0 = time.perf_counter()
    r = c.post(f"{base}/api/{ver}/{ep}", json=body)
    ms = round((time.perf_counter() - t0) * 1000)
    try:
        j = r.json()
    except Exception:
        j = {"_text": r.text[:300]}
    return {"status": r.status_code, "ms": ms, "body": j}


def summarize(res: dict) -> dict:
    b = res["body"]
    out = {"status": res["status"], "ms": res["ms"]}
    if res["status"] >= 400:
        out["error"] = json.dumps(b)[:300]
        return out
    if isinstance(b, dict) and "kind" in b:
        out["envelope"] = f"kind={b['kind']}"
        b = b.get("data", {})
    if isinstance(b, dict) and "hits" in b:
        hits = b["hits"]
        out["total"] = b.get("total")
        out["hit_keys"] = sorted(hits[0].keys()) if hits else []
        out["top"] = [{k: h.get(k) for k in ("feature", "$value", "$p", "ticker", "vintage_year", "$similarity", "$score", "lift")
                       if k in h} for h in hits[:5]]
        if hits and "$why" in hits[0]:
            w = hits[0]["$why"]
            out["why_root"] = w.get("type") if isinstance(w, dict) else type(w).__name__
            out["why_types"] = sorted(_why_types(w))
        if hits and "related" in hits[0]:
            out["relate_first"] = {k: hits[0].get(k) for k in ("related", "lift", "n", "info") if k in hits[0]}
    elif isinstance(b, dict):
        out["keys"] = sorted(b.keys())[:20]
        out["metrics"] = {k: b.get(k) for k in ("accuracy", "baseAccuracy", "informationGain", "meanRank", "testSamples", "n")
                          if k in b}
    return out


def _why_types(node, acc=None):
    acc = set() if acc is None else acc
    if isinstance(node, dict):
        if "type" in node:
            acc.add(node["type"])
        for v in node.values():
            _why_types(v, acc)
    elif isinstance(node, list):
        for v in node:
            _why_types(v, acc)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-env", default=None, help="send v2 calls to /env/<name> (a migrated branch)")
    ap.add_argument("--out", default="docs/v2-parity.json")
    args = ap.parse_args()
    v2base = f"{URL}/env/{args.v2_env}" if args.v2_env else URL
    report = {"instance": URL, "v2_target": v2base, "cases": []}
    with httpx.Client(headers=H, timeout=120) as c:
        report["version"] = c.get(f"{URL}/api/v2/_version").json()
        for name, where, ep1, b1, ep2, b2 in CASES:
            r1, r2 = summarize(call(c, URL, "v1", ep1, b1)), summarize(call(c, v2base, "v2", ep2, b2))
            report["cases"].append({"case": name, "used_by": where, "v1": {"endpoint": ep1, **r1}, "v2": {"endpoint": ep2, **r2}})
            print(f"── {name}  [{where}]")
            print(f"   v1 {ep1:12s} {json.dumps(r1)[:700]}")
            print(f"   v2 {ep2:12s} {json.dumps(r2)[:700]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
