"""Emit site/data/error_analysis.json — the engine diagnosing its own errors.

Generates held-out predictions (5-fold, ai:high), loads them into Aito as
`eval_predictions` with a `correct` flag, then uses `_relate` to find which
input features associate with errors — overall and, via `$on`, within each
actual/predicted bucket. The demo's own relate query, pointed at itself.

Slow (5 train-table uploads + ~1300 predicts + relate). One-shot; commit the JSON:

    uv run python -m pipeline.model.emit_error_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd

from pipeline.aito.client import AitoClient
from pipeline.aito.load import df_to_aito_rows, load_schema
from pipeline.model import featureclust as fc

OUT = Path("site/data/error_analysis.json")
CL = fc.CLASSES
INT = {"leadership_quality", "capital_allocation", "strategic_clarity", "execution_track_record"}
AI = {"ai": "high"}
TMP, TABLE = "holdout_tmp", "eval_predictions"


def _wof(row, feats):
    w = {}
    for f in feats:
        v = row[f]
        if pd.isna(v) or v == "":
            continue
        w[f] = int(float(v)) if f in INT else (True if str(v) == "True" else (False if str(v) == "False" else str(v)))
    return w


def main() -> None:
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    url = os.environ["AITO_API_URL"].rstrip("/"); key = os.environ["AITO_API_KEY"]; h = {"x-api-key": key}

    df = pd.read_csv("data/companies.csv", low_memory=False)
    df = df[df.outcome_bucket.isin(CL)].reset_index(drop=True)
    df["eval_fold"] = df["eval_fold"].astype(str)
    feats = [f for f in fc.DEFAULT_FEATURES if f in df.columns]
    coltypes = load_schema()["schema"]["companies"]["columns"]

    # 1. held-out predictions, 5-fold, ai:high
    preds = []
    for fold in ["0", "1", "2", "3", "4"]:
        train, test = df[df.eval_fold != fold], df[df.eval_fold == fold]
        with AitoClient() as c:
            c.delete_table(TMP); c.put_schema({"schema": {TMP: load_schema()["schema"]["companies"]}})
            c.upload_batch(TMP, df_to_aito_rows(train))
        with httpx.Client(timeout=60) as cl:
            for _, r in test.iterrows():
                w = _wof(r, feats)
                if not w:
                    continue
                rr = cl.post(f"{url}/api/v1/_predict",
                             json={"from": TMP, "where": w, "predict": "outcome_bucket", "config": AI}, headers=h)
                if rr.status_code >= 400:
                    continue
                d = {x["feature"]: x["$p"] for x in rr.json()["hits"]}
                pred = max(d, key=d.get)
                row = dict(w); row.update({"true_bucket": r.outcome_bucket, "pred_bucket": pred, "correct": bool(pred == r.outcome_bucket)})
                preds.append(row)
        with AitoClient() as c:
            c.delete_table(TMP)
    acc = sum(p["correct"] for p in preds) / len(preds)
    print(f"→ {len(preds)} held-out predictions · accuracy {acc:.3f}")

    # 2. load eval_predictions
    cols = {f: coltypes[f] for f in feats}
    cols.update({"true_bucket": {"type": "String", "nullable": True},
                 "pred_bucket": {"type": "String", "nullable": True},
                 "correct": {"type": "Boolean", "nullable": True}})
    with AitoClient() as c:
        c.delete_table(TABLE); c.put_schema({"schema": {TABLE: {"type": "table", "columns": cols}}})
        c.upload_batch(TABLE, preds)

    # 3. relate errors → features, overall + $on each bucket
    def relate(where):
        r = httpx.post(f"{url}/api/v1/_relate",
                       json={"from": TABLE, "where": where, "relate": feats, "limit": 12}, headers=h, timeout=90)
        if r.status_code >= 400:
            return []
        out = []
        for hit in r.json().get("hits", []):
            rel = hit.get("related", {})
            field = next((k for k in rel if not k.startswith("$")), None)
            if not field:
                continue
            val = rel[field].get("$has") if isinstance(rel[field], dict) else rel[field]
            out.append({"feature": field, "value": str(val), "lift": round(hit.get("lift") or 0, 2),
                        "n": int(hit.get("fs", {}).get("f", 0))})
        out.sort(key=lambda x: -x["lift"])
        return out

    def scope_stats(pred_key=None, true_key=None):
        rows = [p for p in preds if (true_key is None or p["true_bucket"] == true_key) and (pred_key is None or p["pred_bucket"] == pred_key)]
        n = len(rows); err = sum(1 for p in rows if not p["correct"])
        return n, err

    scopes = []
    print("→ relate: overall")
    n, err = scope_stats()
    scopes.append({"key": "overall", "group": "Overall", "label": "All predictions",
                   "sub": "every held-out prediction", "n": n, "errors": err,
                   "relate": relate({"correct": False})})
    for b in CL:
        print(f"→ relate: actual={b}")
        n, err = scope_stats(true_key=b)
        subs = {"great": "the winners — where did it miss?", "disaster": "the crashes — did it warn?"}
        scopes.append({"key": f"true:{b}", "group": "By actual outcome", "label": f"Actual '{b}'",
                       "sub": subs.get(b, f"companies that turned out {b}"), "n": n, "errors": err,
                       "relate": relate({"$on": {"prop": {"correct": False}, "on": {"true_bucket": b}}})})
    for b in CL:
        print(f"→ relate: predicted={b}")
        n, err = scope_stats(pred_key=b)
        scopes.append({"key": f"pred:{b}", "group": "By predicted call", "label": f"Predicted '{b}'",
                       "sub": f"when it called '{b}', when was it wrong?", "n": n, "errors": err,
                       "relate": relate({"$on": {"prop": {"correct": False}, "on": {"pred_bucket": b}}})})

    payload = {
        "note": ("Aito's own relate query pointed at its errors. Held-out predictions "
                 "(5-fold, ai:high grouping); each is marked correct/incorrect and loaded "
                 "back into Aito. relate(correct=false) surfaces which input features "
                 "associate with the engine being wrong. lift > 1 = more error-prone, "
                 "< 1 = more reliable. $on scopes it within an actual/predicted bucket."),
        "n_heldout": len(preds), "accuracy": round(acc, 3), "classes": CL,
        "features": feats, "scopes": scopes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with AitoClient() as c:
        c.delete_table(TABLE)
    print(f"→ {OUT} ({len(scopes)} scopes) · cleaned up {TABLE}")


if __name__ == "__main__":
    main()
