# Wide-feature calibration — investigation & closing record

**Status:** closed investigation. Grouping + calibration shipped in aito-core and
measurably helped; one gap remains, precisely characterised below.
**Audience:** aito-core. This is the record of what we found and the one
recommendation that follows from it.

**One-line arc:**
> redundancy → **grouping** (shipped, helped); thin-data noise → **feature
> pruning** (exists in the engine, but cuts the wrong tail — needs a
> support-based, more-aggressive criterion).

**Reproduce everything** against `https://shared.aito.ai/db/aito-equity-demo`
(table `companies`, target `outcome_bucket`, 1,294 graded rows, 16 candidate
features). Tools: `pipeline/model/eval_aito.py` (masked `_evaluate`),
`pipeline/model/featureclust.py` (the pandas reference model), booktests
`book/test_02` / `test_03`. All held-out numbers below are Aito's **own**
`_evaluate` (it masks each test row), unless marked "pandas".

---

## 1 · The problem

Aito's `_predict` is Naive-Bayes-family: base rate × per-feature lift (visible
in `$why`). NB assumes conditional independence; correlated features get their
lifts multiplied as independent evidence, so probabilities blow out. On 16
correlated qualitative features this is the dominant error.

Redundancy is explicit — top normalised mutual information between features:

```
0.68  leadership_quality ~ execution_track_record
0.62  leadership_quality ~ capital_allocation
0.56  market_position    ~ moat_strength
0.48  leadership_quality ~ strategic_clarity
```

The four 1–5 "quality" grades are one latent measured four ways.

Pandas flat-NB, held-out 5-fold, quantifies the starting point:

| model | log-loss | ECE | overconfidence |
|---|---|---|---|
| flat NB, 16 features | 1.823 | 0.194 | **+19.4 pt** |

(overconfidence = mean top-class confidence − accuracy). A prototype
(`featureclust.py`) confirmed the fix direction in pandas: compression-cluster
correlated features into themes and vote once per theme → ECE 0.194 → 0.071,
overconfidence +19.4 → +6.7 pt, keeping all 16 features.

---

## 2 · What shipped, and what it bought

aito-core added **grouping** (data-driven feature interaction/dedup) and a
**calibration** normalizer to the default inference. Both are visible in `$why`
for a 16-feature prediction:

```
product
├─ baseP  0.062
├─ product{ normalizer, normalizer, calibration }        ← new calibration term
├─ relatedPropositionLift {sector ∧ market_quality}  ×3.0   ← GROUP (one vote)
├─ relatedPropositionLift {momentum ∧ volatility}    ×4.9   ← GROUP
├─ relatedPropositionLift {leadership ∧ execution}   ×1.6   ← GROUP
├─ relatedPropositionLift {profitability_bucket}     ×0.89
└─ …
```

Correlated features are combined into `$and` joint factors — the engine's own
version of the prototype's themes. Measured via masked `_evaluate`
(`informationGain` = base-entropy − cross-entropy in bits; >0 means the
predicted probabilities beat the base rate; <0 means worse than base rate),
**before vs after** the update, vintage holdouts:

| holdout | feats | infoGain before | infoGain after | Δ |
|---|---|---|---|---|
| 2014 | 7 | +0.141 | +0.164 | +0.023 |
| 2014 | 16 | +0.044 | **+0.081** | **+0.037** |
| 2017 | 7 | +0.214 | +0.234 | +0.020 |
| 2017 | 16 | +0.116 | **+0.151** | **+0.035** |
| 2020 | 7 | −0.029 | −0.007 | +0.022 |
| 2020 | 16 | −0.168 | **−0.118** | **+0.050** |

Every cell improved, and **the 16-feature gains are ~2× the 7-feature gains** —
the update disproportionately helped the wide, redundant, thin-cell cases, which
is exactly what grouping+calibration should do. The wide-feature penalty (16f vs
7f) narrowed in every vintage. **This is a real, targeted win.**

But it's partial: 16 features still trail 7 in every vintage, and the 2020
holdout is still below base rate. Grouping reduced the penalty; it did not
remove it.

---

## 3 · Isolating the effect from the year trend (random cross-year CV)

Vintage holdouts confound the calibration effect with a **regime shift**:
outcomes are strongly year-dependent (2014 vintage 65.5% upside / +205% median;
2020 56.6% / +69%), and a model with no future-regime feature can't extrapolate
across years. To isolate the redundancy effect, we added a company-grouped
random fold column (`eval_fold`, hash of ticker mod 5 — each company in one
fold, years mixed within each fold, no same-company leakage) and ran 5-fold
random CV:

| feature set | infoGain | gmLift | accuracyGain | per-fold infoGain |
|---|---|---|---|---|
| 7 features | +0.072 | 1.052 | +0.047 | +0.13, −0.00, +0.15, +0.08, −0.00 |
| 16 features | **−0.030** | 0.981 | +0.037 | +0.05, −0.12, +0.08, −0.09, −0.07 |

**The wide-feature penalty survives random sampling** — 16 features are below 7
in all 5 folds, and adding the extra 9 features flips held-out infoGain from
**+0.072 (beats base rate) to −0.030 (worse than base rate)**. The year trend
sets the *level* and the fold-to-fold spread; it does **not** explain the 7-vs-16
*gap*. The gap is genuine redundancy/thin-data cost.

---

## 4 · Where calibration breaks (the ablation)

Standalone `infoGain` per feature (random fold 0) — signal is extremely
concentrated:

```
+0.155  sector              ← carries almost all of it
+0.051  volatility_bucket
+0.035  momentum_bucket
+0.014  market_quality
+0.006  valuation_bucket
+0.005  moat_strength
+0.002  leadership_quality
 0.000  founder_still_ceo
-0.005  market_position
-0.006  capital_allocation
-0.007  moat_type
-0.009  profitability_bucket
-0.012  strategic_clarity
-0.013  execution_track_record
-0.025  leverage_bucket
-0.027  growth_bucket
```

Everything from `market_position` down is **individually noisier than the base
rate**. Adding features in rank order, cumulative infoGain **peaks at 6
features (+0.182)** then erodes to +0.047 at all 16. The calibration-optimal
subset, validated on 5 folds:

| subset | infoGain | gmLift |
|---|---|---|
| **calibrated-6** = {sector, momentum, volatility, market_quality, valuation, moat_strength} | **+0.139** | 1.102 |
| demo's current 7 | +0.072 | — |
| all 16 | −0.030 | 0.981 |

Note the calibrated set keeps the two LLM qualitative features that earn their
place (`market_quality`, `moat_strength`) and drops the redundant management
cluster and the noisy fundamentals.

---

## 5 · Why interaction features (`$on`) don't rescue it

Hypothesis worth testing: `growth`/`margin`/`leverage` mean different things per
sector, so condition them with `$on` (`{"$on":{"prop":…,"on":…}}` — a dependent
feature). Result (mean infoGain, folds 0–2):

```
sector + growth  FLAT              +0.149
sector + growth  $on sector        +0.084   ← conditioning loses
mkt_quality + growth FLAT          −0.005
mkt_quality + growth $on mkt_qual  −0.022   ← loses even at 5-value context
calibrated-6 + {growth,lev,profit} $on sector : −0.198   ← much worse
```

The interaction is real in theory but **unestimable at n≈1,300**: conditioning
on 11-value sector shatters the data into ~200 (feature × sector × outcome)
cells with <5 rows each. Smoothing pulls them back to the marginal, so the
conditional adds variance without recoverable signal — and it loses even at a
coarser 5-value context. Crucially, **the engine already forms interaction
groups where the joint cells are populated** (§2, the `$and` factors), so it's
already doing adaptive, data-driven conditioning; forcing `$on` overrides that
with a fragile full-conditional. `$on` is the right tool with ~10× the data; on
this dataset it's counterproductive.

Every lever points at one wall: **the binding constraint is sample size, not
feature representation.** More features → worse; conditioned features → worse;
fewer marginal features → best.

---

## 6 · Root cause — the engine prunes, but cuts the wrong tail

The engine already drops features per query. Counting how often each of the 16
appears in `$why` across 5 predictions, against its standalone infoGain:

| feature | in `$why` (of 5) | standalone infoGain | verdict |
|---|---|---|---|
| sector | 5/5 | +0.155 | keep ✓ |
| volatility / momentum / market_quality / valuation / moat_strength | 5/5 | +0.05 … +0.005 | keep ✓ |
| **founder_still_ceo** | **0/5** | +0.000 | dropped |
| market_position / capital_allocation / profitability | 5/5 | −0.005 … −0.009 | kept (mild) |
| **strategic_clarity** | **2/5** | −0.012 | mostly dropped ✓ |
| **execution_track_record** | 5/5 | −0.013 | **kept — harmful** ✗ |
| **leverage_bucket** | 5/5 | −0.025 | **kept — harmful** ✗ |
| **growth_bucket** | 5/5 | −0.027 | **kept — harmful** ✗ |

Two facts:

1. **Pruning exists** — `founder_still_ceo` is dropped entirely (0/5),
   `strategic_clarity` mostly (2/5).
2. **It cuts the wrong tail** — it *keeps the three single most harmful features*
   (growth, leverage, execution) in every prediction, while dropping the
   harmless `founder_still_ceo`. The keep/drop decision barely correlates with
   held-out contribution.

The tell: it dropped the near-constant Boolean (`founder`) but kept the
high-cardinality-but-noisy `growth`/`leverage`. That's consistent with a prune
criterion based on **in-sample apparent lift / feature entropy** — which is
exactly what fails to generalize when the supporting cells are thin. Features
whose lift is a thin-cell artifact survive; boring-but-harmless features get cut.

---

## 7 · Recommendation — support-gated tail-cutting

The engine already prunes features per prediction. Make that prune:

- **(a) more aggressive**, and
- **(b) driven by a data-support / generalization criterion, not in-sample lift.**

Concretely: keep a feature (or group) only if its marginal contribution clears
an evidence-based penalty given how many training rows actually populate its
cells — an MDL/BIC-style term, a penalized-likelihood / held-out gate, or an
evidence-count threshold. Grouping already handles the **redundant** tail (a
feature adds nothing once its group is counted); this handles the **noisy/thin**
tail (a feature whose apparent lift the data can't support).

**The unlock:** proper tail-cutting recovers all-16 from **−0.030 back to the
calibrated ceiling +0.139** — so a user could feed *all* features and the engine
would internally keep only the ~6 the data supports. That is strictly better
than external feature selection, because it's **per-query and adaptive**: a
data-rich common profile can support more features; a rare/thin profile fewer.
It turns "throw all your columns at Aito" from a footgun into the default.

This is the missing half. Grouping (shipped) + support-gated pruning (this)
would keep inference calibrated regardless of how wide the feature space is.

---

## 8 · Interim mitigation in the demo (no engine dependency)

Until pruning is support-gated, the demo standardises predicted probabilities on
the **calibrated-6** subset (`sector, momentum_bucket, volatility_bucket,
market_quality, valuation_bucket, moat_strength`; infoGain +0.139 vs the demo's
old 7 at +0.072 and all-16 at −0.030). Extra features (management scores,
leverage, growth) still drive the *analogues* / exploration surfaces, just not
the headline forecast. The year effect is inherent and disclosed, not modelled.

---

## 9 · Reproduce

```bash
./do pipeline load                   # writes the eval_fold column (§3) into the table
./do pipeline model-eval             # masked held-out, vintage holdouts × feature width (§2)
./do pipeline model-eval --random    # 5-fold RANDOM cross-year CV: 7 vs 16 vs calibrated-6 (§3,§4)

# pandas reference model + booktests (Aito-independent, deterministic)
./do test-book                       # test_02 diagnosis, test_03 clustered-NB fix
uv run python -m pipeline.model.export_spec   # → wide-feature-calibration.spec.json
```

`eval_fold` = `hashlib.md5(ticker) % 5` (company-grouped; deterministic across
runs). `--random` reports the §3/§4 numbers directly. The `$why` prune
cross-reference (§6) and the `$on` conditioning test (§5) are ad-hoc `_predict`
/ `_evaluate` calls documented inline above.

`wide-feature-calibration.spec.json` holds the pandas reference numbers, the
feature schema, the discovered clusters, and out-of-fold reference predictions —
useful for a from-scratch re-implementation, though §2–§7 above (Aito's own
masked `_evaluate`) are the authoritative measurements.
