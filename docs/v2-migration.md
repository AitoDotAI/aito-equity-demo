# Aito v1 → v2 migration (equity demo)

Instance: `shared.aito.ai/db/aito-equity-demo`, core **v2.11.6** (`05ae1327`, built 2026-10-11).
Harness: `scripts/v2_parity.py` → `docs/v2-parity.json` (every query shape the demo sends,
replayed against `/api/v1` and `/api/v2`, payloads diffed — not status codes).

## Status

| Step | State |
|---|---|
| Client code on `/api/v2` (pipeline, emitters, proxy, site) | **done** — this PR |
| New tables created as `collection` (rep2) | **done** in code |
| `v2` env branched from master, both tables `_migrate`d to rep2 | **done** 2026-10-18 (3.3 s, 1,549 + 1,677 rows) |
| `companies` reloaded into `v2` as a fresh `collection` with `profile_text` | **done** (`AITO_ENV=v2 ./do pipeline load`) |
| Parity run v1@master vs v2@env/v2 | **done** → `docs/v2-parity-rep2.json` |
| Held-out quality v1 vs rep2 (backtest, value traps) | see *Held-out comparison* below |
| Promote `v2` → master (live demo switches engine) | **not done** — a separate decision |

Master is untouched (`companies` still `engine: v1`). The ported code serves either:
`AITO_ENV=v2` points pipeline, proxy and Lab at the rep2 copy.

## Held-out comparison (v1 master vs rep2 env)

Same code, same 1,294 held-out rows (ticker-grouped 5-fold backtest) and the same
walk-forward value-trap scoring; only `AITO_ENV` differs.

| | v1 | rep2 |
|---|---|---|
| accuracy / within-one-bucket | 0.35 / 0.743 | 0.35 / 0.747 |
| over-optimistic calls | 0.471 | 0.449 |
| top-20 / 50 / 100 CAGR | 20.6 / 18.3 / 15.9 % | 20.1 / 17.8 / 17.1 % |
| bottom-20 CAGR (market 7.9 %) | −4.1 % | −5.0 % |
| score ↔ realised return (Spearman) | 0.318 | 0.318 |
| P(trap) ↔ long-run trap | 0.295 | 0.304 |
| P(trap) ↔ 12-month return | −0.126 | **−0.182** |
| P(re-rate 12 m) ↔ 12-month return | 0.115 | 0.114 |

Same predicted bucket on 87 % of rows, score correlation 0.95, 16/20 shared top-20 picks.
**No quality regression; the trap axis is slightly sharper on rep2.**

## Steps taken / to finish

1. `POST /api/v2/_envs {"name": "v2", "basedOn": "master"}` — branch; master untouched.
2. `POST /env/v2/api/v2/data/_modify {"operations": [{"migrate": "companies", "engine": "v2"}, {"migrate": "earnings_events", "engine": "v2"}]}` — atomic, no reverse.
3. `AITO_ENV=v2 ./do pipeline load` — recreates `companies` in the env as a `collection`
   with the new `profile_text` column (needed for similarity, finding 4).
4. `uv run python scripts/v2_parity.py --v2-env v2 --out docs/v2-parity-rep2.json`, and the
   held-out emitters with `AITO_ENV=v2 … --refresh`.
5. **To go live:** deploy this branch with `AITO_ENV=v2` set, or `POST /api/v2/_envs/v2/promote`
   and run without `AITO_ENV`. Promote swaps master — do it once, deliberately.

## What changed in the code

- `pipeline/aito/client.py`: single `api_root()` = `AITO_API_URL[/env/AITO_ENV]/api/v2`;
  `hit_value()` (`$value`, falls back to `feature`); `unwrap()` (`kind ?? "rows"`);
  `put_schema` creates each table via `PUT /schema/{t}` as `collection`.
- Emitters (`emit_backtest`, `emit_error_analysis`, `emit_value_traps`, `eval_*`): `/api/v2`,
  `select: ["$value", …]`, `hit_value()`. `emit_error_analysis` sent no `select` and read
  `x["feature"]` — on v2 that is a `KeyError` (default projection is `{$value, $p}`).
- `emit_value_traps` relate: population scoped with `$on [{target: true}, {target: {$defined: true}}]`
  (see finding 2).
- `src/app.py` proxy + `/api/classify-earnings`: `/api/v2`, `hit_value()`.
- Site: Lab reads `$value`; panel copy and the curl snippet say `/api/v2`; two Live Query
  samples that were invalid on **both** versions (`relate: "$features"`, object-valued `match`)
  replaced with valid ones.

## Tracked in aito-core (filed by the CPO, 2026-10-18)

| Finding | Ticket | Priority |
|---|---|---|
| 3 — `$not {field: null}` bitset 400 on trailing-null columns | `td-20261018190238171893` (aito-core-2; with the `_relate` defined-range fix, finding 2) | P1 |
| 4, 5, 8 — categorical `_similarity` on rep2, `_evaluate` baseline mismatch, `table`/`collection` label | `td-20261018190308734654` | P2 |

Keep the `profile_text` BM25 workaround until the `_similarity` decision lands.

## Findings

1. **`_predict` is a near drop-in.** `feature` → `$value`; a verbatim v1 body sent to `/api/v2`
   still returns `feature` (select is honoured). `$why` tree node types unchanged, so
   `parse_why` works as is. `$p` identical on 2 of 7 probes, drifts ≤0.02 on the others (same top value everywhere).
2. **`_relate` silently changes the population when the condition field is nullable.**
   v1 counts only rows where the fields are defined; v2 counts every row and treats a missing
   target as "not the condition". This table has 255 rows with no outcome yet (delisted, 2026
   vintage), so v2 inflates lifts: *sector = IT → great* is ×1.94 on v1, ×2.46 on v2; exact
   lifts are 2.04 (rows with outcome + field known) vs 2.58 (all rows, missing = not great).
   *valuation = cheap → great* goes ×1.10 → ×1.34 — enough to change the story a demo tells.
   v2 has the right tool: `$on [cond, {field: {$defined: true}}]` → ×1.12 on 1,294 rows.
   **v1 ignores the same `$on` silently** (returns the unscoped answer). 200 everywhere.
3. **`bit & operation requires same sized bit sets, found: 1549, 1520`** — a 400 from
   `_relate` with `$on [cond, {$not: {outcome_bucket: null}}]` on `/api/v2`. This is
   G1 / aito-core#1061, which the aito-demo migration could not reproduce. Here it is
   deterministic on all three storages: v1 engine, `_migrate`d rep2, and a fresh `collection`.
   **Trigger:** a nullable column whose *trailing* rows are null. `outcome_bucket`'s last
   value is at row 1,519; rows 1,520–1,548 (the 2026 vintage, no outcome yet) are null. The
   `$not {field: null}` path builds its bitset from the column's stored length (1,520), not
   the table's (1,549). `$defined: true` on the same field works.
4. **Categorical profile similarity has no direct v2 counterpart.** On rep2, `_similarity`
   over String/Int columns is refused: *"rep2-native similarity is text BM25"*. (On a v1-engine
   table `/api/v2/_similarity` still answers, although the docs say it is not on v2.)
   `orderBy: "$similarity"` needs a Text `$match` term; `$match` is refused on non-Text columns.
   **Workaround shipped:** a derived Text column `profile_text` (whitespace analyzer, one token
   per field=value, e.g. `moatstrengthxwide`) and BM25 `_similarity` on it. Against v1's
   `_similarity`, top-10 overlap is 8/10 (NVDA-like profile) and 9/10 (value-trap profile),
   same order at the top. Scores change scale (BM25 ≈ 10–80 vs v1 ≈ 1–7).
5. **`_evaluate` changes its numbers on rep2.** Same storage → identical metrics, only the
   `{kind: "evaluation", data}` envelope. On rep2, same query and same 362 test rows:
   accuracy 0.315 → 0.373, **baseAccuracy 0.257 → 0.307**, meanRank 1.74 → 1.09,
   informationGain 0.187 → 0.177, and ~3× slower (7 s → 23 s). A different *base* on the same
   test rows means the baseline is computed differently, so v1 and rep2 `_evaluate` figures are
   not comparable. rep2 adds `brierScore`, `ece`, `logLoss`, `mrr`.
6. **`type: table` is still the default on v2.** A client that swaps `/api/v1` → `/api/v2`
   and keeps its schema keeps creating v1-engine tables. Nothing tells it. There is also no
   documented whole-database `PUT /api/v2/schema` operation page (base-v2 lists it).
7. **`GET /api/v2/schema` echoes `engine` without `?meta`** — the docs say it is opt-in.
   Useful, but the docs and the server disagree.
8. **A `_migrate`d table still reports `type: table`** next to `engine: v2`; a freshly created
   one says `type: collection`. Two names for the same engine depending on how it got there.
9. **`_migrate` is fast and atomic** — two tables in 3.3 s, single commit, row counts preserved.
   `$why` on rep2 adds a `calibration` node (`rowCap`) inside the normalizer group; existing
   parsers that walk `baseP` / `relatedPropositionLift` are unaffected.
10. **v2 error messages are clearly better** — e.g. "no such field 'x' in the where clause.
   This is a strict (type:table) schema …", "_match: missing 'match' (the field whose values
   to rank)". They made the two pre-existing broken console samples obvious.
