# Aito v1 → v2 migration (equity demo)

Instance: `shared.aito.ai/db/aito-equity-demo`, core **v2.11.6** (`05ae1327`, built 2026-10-11).
Harness: `scripts/v2_parity.py` → `docs/v2-parity.json` (every query shape the demo sends,
replayed against `/api/v1` and `/api/v2`, payloads diffed — not status codes).

## Status

| Step | State |
|---|---|
| Client code on `/api/v2` (pipeline, emitters, proxy, site) | **done** — this PR |
| New tables created as `collection` (rep2) | **done** in code; takes effect on next `./do pipeline load` |
| Storage migration of the live tables (`_migrate` → rep2) | **not done** — needs an approved write to the shared instance (plan below) |
| Parity run v1@master vs v2@migrated env | pending the step above |

The ported code runs against today's storage: the v2 API answers over v1-engine
(`type: table`) tables, and all four proxy kinds (`predict`, `relate`, `match`,
`similarity`) were verified live through the local server.

## Plan for the storage step

1. `POST /api/v2/_envs {"name": "v2", "basedOn": "master"}` — branch; master untouched.
2. `POST /env/v2/api/v2/data/_modify {"operations": [{"migrate": "companies", "engine": "v2"}, {"migrate": "earnings_events", "engine": "v2"}]}` — atomic, no reverse.
3. `AITO_ENV=v2 uv run python scripts/v2_parity.py --v2-env v2` and the masked held-out evals
   (`./do pipeline model-eval`, `backtest-page`) with `AITO_ENV=v2` → compare to v1.
4. If the numbers hold: `POST /api/v2/_envs/v2/promote`, or simply re-run `./do pipeline load`
   (it now creates `collection` tables).

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
   deterministic. The two sizes match this table's history — 1,520 rows at first load, 1,549
   after the 2026 rows were added — so a size taken from an older segment/column state looks
   likely; not verified in core. (v1-engine table, v2 API.)
4. **No v2 idiom for categorical profile similarity.** The demo's analogues use
   `_similarity` with a profile of String/Int columns. `orderBy: "$similarity"` is refused
   ("needs a {field: {$match: text}} term"); `$match` is refused on non-Text columns;
   `orderBy: {"$similarity": …}` is a parse error. `/api/v2/_similarity` **does** still
   answer 200 (on a v1-engine table) although the v2 docs say it is not on v2. Kept on
   `_similarity` until there is a documented answer; re-check after `_migrate`.
5. **`_evaluate`**: identical metrics on the same storage; only the `{kind: "evaluation", data}`
   envelope. Documented, handled by `unwrap()`.
6. **`type: table` is still the default on v2.** A client that swaps `/api/v1` → `/api/v2`
   and keeps its schema keeps creating v1-engine tables. Nothing tells it. There is also no
   documented whole-database `PUT /api/v2/schema` operation page (base-v2 lists it).
7. **`GET /api/v2/schema` echoes `engine` without `?meta`** — the docs say it is opt-in.
   Useful, but the docs and the server disagree.
8. **v2 error messages are clearly better** — e.g. "no such field 'x' in the where clause.
   This is a strict (type:table) schema …", "_match: missing 'match' (the field whose values
   to rank)". They made the two pre-existing broken console samples obvious.
