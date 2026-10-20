"""Live contract test: does the running Aito (and the deployed site) still answer
every query shape the demo sends, in the shape the demo reads?

Every incompatibility seen across v1 → v2 came back as HTTP 200 with a changed
payload (`feature` → `$value`, `{kind, data}` envelopes, relate populations),
so these tests assert *shapes and semantics*, through the demo's own parsers —
not status codes.

Read-only. Skipped unless AITO_API_URL / AITO_API_KEY are set, and deselected
from the default `./do test`; run with:

    ./do contract                       # Aito master + https://demos.aito.ai/equity
    EQUITY_SITE_URL=http://localhost:8401 ./do contract
"""
from __future__ import annotations

import os

import httpx
import pytest

from pipeline.aito.client import AitoConfig, hit_value, unwrap
from pipeline.aito.load import profile_token
from pipeline.aito.queries import _parse_relate_hit, parse_why

pytestmark = pytest.mark.live

OUTCOMES = {"disaster", "poor", "market", "good", "great"}
MID = {"lagged", "inline", "outperformed"}
PROFILE = {"market_position": "dominant", "moat_strength": "wide", "market_quality": "secular_growth",
           "leadership_quality": 5, "sector": "Information Technology", "valuation_vs_own": "below_own_norm"}
# Columns the site and the Lab read; dropping or renaming one is a silent break.
SITE_COLUMNS = ["profile_text", "fwd_6m_bucket", "fwd_12m_bucket", "valuation_bucket", "valuation_vs_own",
                "cheap_kind", "pe_vs_own", "outcome_bucket", "total_return_pct_local", "market_quality"]


@pytest.fixture(scope="module")
def aito():
    if not (os.environ.get("AITO_API_URL") and os.environ.get("AITO_API_KEY")):
        pytest.skip("AITO_API_URL / AITO_API_KEY not set")
    cfg = AitoConfig.from_env()
    with httpx.Client(base_url=cfg.api_root, headers={"x-api-key": cfg.api_key}, timeout=120) as c:
        v = c.get("/_version")
        print(f"\ncore {v.json() if v.status_code == 200 else v.status_code} at {cfg.api_root}")
        yield c


def _post(c, ep, body):
    r = c.post(ep, json=body)
    assert r.status_code == 200, f"{ep}: {r.status_code} {r.text[:300]}"
    return r.json()


def _predict(c, where, target, n_expected, values):
    res = _post(c, "/_predict", {"from": "companies", "where": where, "predict": target,
                                 "select": ["$value", "$p", "$why"], "config": {"ai": "high"}})
    hits = res["hits"]
    assert len(hits) == n_expected, f"{target}: {len(hits)} hits"
    assert {hit_value(h) for h in hits} == values
    assert abs(sum(h["$p"] for h in hits) - 1) < 0.02, "probabilities no longer sum to ~1"
    why = parse_why(hits[0]["$why"])
    assert why and why["base_rate"] is not None, f"$why no longer parses: {str(hits[0]['$why'])[:200]}"
    if where:  # evidence → per-feature lifts; no evidence → base rate only
        assert why["factors"], f"$why has no feature lifts: {str(hits[0]['$why'])[:200]}"
    return hits


# ── storage ──────────────────────────────────────────────────────

def test_companies_is_rep2_with_site_columns(aito):
    meta = aito.get("/schema/companies?meta").json()
    assert meta.get("engine") == "v2", f"companies engine is {meta.get('engine')}"
    missing = [c for c in SITE_COLUMNS if c not in meta.get("columns", {})]
    assert not missing, f"columns the site reads are gone: {missing}"


# ── Company Lab / Company File ───────────────────────────────────

def test_lab_longrun_predict(aito):
    _predict(aito, PROFILE, "outcome_bucket", 5, OUTCOMES)


def test_lab_6m_predict(aito):
    _predict(aito, PROFILE, "fwd_6m_bucket", 3, MID)


def test_lab_base_rate_with_nothing_set(aito):
    _predict(aito, {}, "outcome_bucket", 5, OUTCOMES)


def test_lab_analogues_bm25_profile(aito):
    text = " ".join(profile_token(k, v) for k, v in PROFILE.items())
    res = _post(aito, "/_similarity", {"from": "companies", "similarity": {"profile_text": text}, "limit": 6})
    hits = res["hits"]
    assert len(hits) == 6
    for k in ("ticker", "vintage_year", "company_name", "outcome_bucket", "$score"):
        assert k in hits[0], f"analogue hit lost '{k}'"
    assert hits[0]["$score"] >= hits[-1]["$score"], "analogues no longer ranked by score"


# ── relate (Thesis, Cheap for a Reason?, Error Analysis, Live Query) ──

def test_relate_scoped_population(aito):
    res = _post(aito, "/_relate", {"from": "companies",
                                   "where": {"$on": [{"outcome_bucket": "great"}, {"outcome_bucket": {"$defined": True}}]},
                                   "relate": ["sector", "moat_strength"], "limit": 10})
    hits = res["hits"]
    assert hits and all(_parse_relate_hit(h) for h in hits), "relate hits no longer parse"
    total = _post(aito, "/_query", {"from": "companies", "limit": 0})["total"]
    n = hits[0].get("n") or hits[0]["fs"]["n"]
    assert n < total, f"$on/$defined no longer scopes the population (n={n}, table={total})"


# ── Evaluation page ──────────────────────────────────────────────

def test_evaluate_envelope_and_metrics(aito):
    from pipeline.model.emit_eval import _metric
    res = _post(aito, "/_evaluate", {"test": {"eval_fold": "0"}, "evaluate": {
        "from": "companies", "where": {f: {"$get": f} for f in ["sector", "moat_strength", "market_quality"]},
        "predict": "outcome_bucket"}})
    assert res.get("kind") == "evaluation", "_evaluate envelope changed"
    d = unwrap(res)
    for k in ("accuracy", "baseAccuracy", "accuracyGain", "informationGain", "meanRank"):
        assert _metric(d, k) is not None, f"_evaluate lost '{k}'"
    assert _metric(d, "geomMeanLift") is not None, "geomMeanLift can no longer be derived"
    assert 0 < d["baseAccuracy"] < d["accuracy"] + 0.2


# ── Live Query samples + earnings box ────────────────────────────

def test_live_query_match_sample(aito):
    res = _post(aito, "/_match", {"from": "companies", "where": {"market_position": "dominant", "moat_strength": "wide"},
                                  "match": "ticker", "limit": 3})
    assert len(res["hits"]) == 3 and all(hit_value(h) for h in res["hits"])


def test_earnings_predict(aito):
    res = _post(aito, "/_predict", {"from": "earnings_events", "where": {"guidance": "raised", "tone": "positive"},
                                    "predict": "react_1d_bucket", "config": {"ai": "high"}})
    assert {hit_value(h) for h in res["hits"]} == {"up", "flat", "down"}


# ── known core gaps: strict xfail → this test FAILS the day they're fixed ──

@pytest.mark.xfail(strict=True, reason="aito-core td-20261018190238171893: $not-null bitset on trailing-null columns; "
                                       "fixed in core v2.11.8 (#1725) — when this XPASSes, drop the xfail")
def test_known_gap_not_null_on_trailing_nulls(aito):
    _post(aito, "/_relate", {"from": "companies", "where": {"$on": [{"outcome_bucket": "great"},
                             {"$not": {"outcome_bucket": None}}]}, "relate": ["sector"], "limit": 2})


@pytest.mark.xfail(strict=True, reason="aito-core td-20261018190308734654: categorical _similarity on rep2")
def test_known_gap_categorical_similarity(aito):
    _post(aito, "/_similarity", {"from": "companies", "similarity": {"moat_strength": "wide"}, "limit": 2})


# ── the deployed site's proxy ────────────────────────────────────

@pytest.fixture(scope="module")
def site():
    base = os.environ.get("EQUITY_SITE_URL", "https://demos.aito.ai/equity").rstrip("/")
    with httpx.Client(base_url=base, timeout=60) as c:
        yield c


def test_site_serves_data_with_new_columns(site):
    u = site.get("/data/universe.json")
    assert u.status_code == 200
    missing = [c for c in ("pe_vs_own", "cheap_kind", "pe_hist_1y", "fwd_6m_excess") if c not in u.json()["columns"]]
    assert not missing, f"deployed universe.json lacks {missing}"


def test_site_proxy_predict_and_analogues(site):
    for body in ({"kind": "predict", "body": {"from": "companies", "where": PROFILE, "predict": "fwd_6m_bucket",
                                              "select": ["$value", "$p"]}},
                 {"kind": "similarity", "body": {"from": "companies", "limit": 3, "similarity": {
                     "profile_text": " ".join(profile_token(k, v) for k, v in PROFILE.items())}}}):
        r = site.post("/api/query", json=body)
        j = r.json()
        assert r.status_code == 200 and j.get("ok") and j.get("source") == "aito", f"{body['kind']}: {str(j)[:300]}"
        assert j["result"]["hits"], f"{body['kind']}: proxy returned no hits (Lab would show the base rate)"
