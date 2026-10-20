# CLAUDE.md — agent instructions for aito-equity-demo

This file is read by Claude Code (and similar agents) when working in this
repo. Keep it tight; put narrative in `docs/` if it grows.

## What this is

A demo of the **unstructured-judgment → LLM grading → predictive database**
pattern, vehicle = long-horizon equity research (Buffett-style: moat,
position, market quality, leadership). Primary audience: a large quantitative trading firm. Secondary:
public artifact (HN/LinkedIn/blog) between the two meetings.

**Deployment target:** https://demos.aito.ai/equity/ — path-mounted on the
`aito-demos-unified` platform alongside the other `*.aito.ai` demos.

**The demo is the methodology.** The numbers in the UI come from real
Aito queries against real data — placeholder JSON in `site/data/` exists
only for the period between repo bootstrap and the first pipeline run.

## Stack

This demo **deviates from the demo-server default** (which is FastAPI +
Next.js with a shared shell). The equity demo has its own editorial
visual identity (single-file HTML, Fraunces / Source Serif 4 type, paper
texture, double-rule dividers) and no app backend beyond a thin Aito query proxy.

| Layer | Lives at | Notes |
|---|---|---|
| Data pipeline | `pipeline/` (Python 3.12) | Stages: universe → filings → extract → outcomes → load → precompute |
| Static site | `site/index.html` + `site/data/*.json` | Single-file HTML with inline CSS/JS; fetches JSON at runtime |
| Server | `src/app.py` (FastAPI) | Serves `site/` + `/health` + `/api/health`, plus a thin **read-only query proxy**: `/api/query` (predict / relate / match / similarity, `from` must be `companies`) for the Company Lab and Live Query, and `/api/classify-earnings` (LLM extract → Aito predict). The key stays server-side. |
| Aito | external, **v2 API on rep2 storage** | Predictive database; pipeline writes to it, then precomputes query results to JSON |

Every *view* is precomputed by the pipeline into `site/data/` (reproducible,
works without Aito). Only the interactive surfaces — Company Lab, Live Query,
the earnings box — query Aito at request time, through the proxy above, so
the API key never reaches the browser.

## Aito v2

The demo runs on the Aito **v2 API** (`/api/v2`) and master's tables are
**rep2** (`collection`) storage — the `v2` env was promoted into master on
2026-10-19. Rollback is a database action, not a config edit.

- One root for every call: `pipeline.aito.client.api_root()` =
  `AITO_API_URL[/env/AITO_ENV]/api/v2`. `AITO_ENV` (optional) points the whole
  demo — pipeline, proxy, Lab — at a branched env: branch, load / `_migrate`
  there, regenerate, compare, then promote.
- Ranked-value hits carry `$value`, not `feature` → use `hit_value()`.
  `_evaluate` / `_estimate` / `_aggregate` answer `{kind, data}` → `unwrap()`.
- `put_schema` creates tables as `collection`; `type: table` is still the v2
  default and would silently keep the legacy engine.
- Analogue search: rep2 `_similarity` is text BM25 only, so analogues rank on
  the derived Text column `profile_text` (one token per field=value,
  `profile_token()` in `pipeline/aito/load.py`, `profileToken` in the site).
- `_relate` on a **nullable** condition (`outcome_bucket` is null for
  delisted / current rows): v2 counts a missing value as "not the condition".
  Scope the population: `{"$on": [cond, {field: {"$defined": true}}]}`.
- `scripts/v2_parity.py` replays every query shape the demo sends and diffs
  payloads; `docs/v2-migration.md` has the findings and the held-out
  v1-vs-rep2 comparison.

## Conventions enforced by the platform — DO NOT drift

| Convention | Lives at |
|---|---|
| Health stub entry | `src/app.py` defines `app: FastAPI` |
| Static mount | `app.mount("/", StaticFiles(directory="site", html=True))` — must be LAST so it doesn't shadow `/health` |
| `/health` | Returns 200 cheaply, no Aito call |
| `/api/health` | Returns 200 + `backend: "static-with-query-proxy"` |
| Lockfile | `uv.lock` committed (`uv sync --frozen` in build) |
| Env contract | `AITO_API_URL`, `AITO_API_KEY` (pipeline), `ANTHROPIC_API_KEY` (extraction) |

If you need to break one of these, update both this demo AND the platform's
expectations in the same PR (the platform's `demos.config.yaml` may need
adjustment — there's no per-demo Next.js build for this one).

## Local dev

```bash
./do install                     # uv sync + playwright browser install (one-time)
./do serve                       # uvicorn stub → http://localhost:8401

# Pipeline (each stage feeds the next; all skeletal in v1):
./do pipeline universe           # → data/universe.csv
./do pipeline filings            # → data/10k_excerpts/
./do pipeline extract            # → data/llm_features.csv  (needs ANTHROPIC_API_KEY)
./do pipeline outcomes           # → data/outcomes.csv
./do pipeline load               # push to Aito           (needs AITO_API_URL + AITO_API_KEY)
./do pipeline precompute         # → site/data/*.json
./do pipeline all                # the lot, in order

./do test                        # pytest tests/ + book/
./do test-book                   # booktest snapshots only

./do screenshot-teaser           # assets/teaser.html → assets/teaser.png (1200×630)
./do screenshot-pages            # full-page screenshots of /
./do inspect-mobile              # iPhone-sized screenshots of /
```

## Where to extend

1. **Pipeline stages** (`pipeline/`) — all stages currently raise `NotImplementedError`.
   The protocols (`UniverseSource`, `FilingFetcher`, `PriceSource`) and the
   data shapes (`UniverseEntry`, `Filing`, `TerminalOutcome`) are defined;
   implementing US-only concrete sources is v1 scope.

2. **Static site** (`site/index.html`) — descended from `aito-equity-demo.html`
   (kept at repo root as the design source-of-truth). All data points are
   bound to JSON files in `site/data/`; visual layout / CSS rarely needs
   touching. The four views (Company File / Thesis / Analogues / Methodology)
   map 1:1 to four Aito query types (predict / relate / match / static).

3. **Aito schema** (`pipeline/aito/schema.json`) — includes multi-market
   columns (market, exchange, currency, reporting_standard, filing_language)
   from day 1, even though v1 data is US-only. **Don't strip these.** They
   exist so a later Finnish / Nordic / EU dataset slots in without a
   migration.

4. **Notebooks** (`notebooks/`) — public reproducibility layer; each notebook
   reproduces one view's underlying data. Add when the pipeline lands.

5. **ADRs** (`ADRs/`) — six method-card decisions to document; write last,
   when methodology has settled.

## Multi-market: US-only data, generic shape

v1 ships US-only data (S&P 500 vintages). The architecture is generic so
Finnish (OMX Helsinki), Nordic, or EU datasets can be added without
schema migration. What's already done:

- Schema columns: `market`, `exchange`, `currency`, `reporting_standard`,
  `filing_language` are in `pipeline/aito/schema.json` from day 1
- Tickers Yahoo-namespaced (`NOKIA.HE`, `VOLV-B.ST`) so keys stay unique
- Sector taxonomy is GICS only (works globally; don't mix in ICB)
- `UniverseSource` / `FilingFetcher` / `PriceSource` are Protocols;
  v1 concrete: `SP500WikipediaSource`, `EDGARFetcher`, `YFinancePriceSource`
- Non-US concrete sources are `NotImplementedError` stubs

What's NOT done (deliberately):
- Non-US universe sources
- Prompt translation (v1 LLM prompts are English; Finnish filings would
  need empirical eval first — Sonnet handles non-English but quality drifts)
- Cross-currency outcome model (`total_return_pct_local` + `total_return_pct_usd`
  columns exist; pipeline only fills `_usd` for v1)

## When you write new code

- **Aito queries** — see `CHEATSHEET.md` for the standard predict / match /
  relate / search patterns. The pipeline's `pipeline/aito/queries.py` is
  where the predict / relate / match calls land.
- **Test snapshots** — non-deterministic Aito outputs (probabilities,
  similarities) are easier to assert against with `booktest` than
  `pytest assert ==`. See `book/test_01_aito_schema_book.py` for the pattern.
- **Caching** — the precomputed-JSON design means there's no per-request
  cache problem; cache happens at pipeline-run time (everything is
  one-shot). For pipeline runs themselves, cache 10-K downloads to
  `data/10k_excerpts/` — EDGAR rate-limits at 10 req/s and re-downloads
  are wasteful.
- **Logging** — `print()` works; supervisord forwards stdout/stderr to
  Azure Log Analytics. Pipeline stages are batch-mode so print-per-row
  is fine.

## Common pitfalls

- **`uv sync --frozen` fails** → `uv.lock` is out of date; run `uv lock`
  and re-commit.
- **Site renders blank / "data load failed" banner** → `site/data/*.json`
  missing or malformed. Run `./do pipeline precompute` to regenerate, or
  copy the placeholder JSONs from a known-good commit.
- **`/api/*` 404 unexpectedly** → you've added a route after the
  `app.mount` line in `src/app.py`. The mount shadows everything below.
  Keep the proxy read-only and narrow; if you reach for a new route, ask
  whether the data should be precomputed instead.
- **Lab shows "Aito gave no answer"** → the proxy got zero hits (wrong env,
  empty table mid-reload, unknown field). The card is showing the base rate.
- **Valuation numbers look absurd** (P/E < 1) → market cap must be
  point-in-time: `valuation_history` pairs the as-traded price with a
  split-consistent share count. Never multiply a today-adjusted price by an
  as-reported share count.
- **Pipeline stage fails with `NotImplementedError`** → that stage is
  scaffolded but not implemented yet; see `aito-equity-demo-TASK.md` for
  the v1 build order.
- **EDGAR returns 403 / 429** → User-Agent header missing or rate limit;
  enforce 10 req/s and provide a real contact email in the UA.
- **Survivorship bias** — bankrupt companies stop filing. The
  `EDGARFetcher` returns `[]` rather than failing; the caller must decide
  whether to drop, grade the last-available 10-K, or flag the row. Don't
  silently drop without noting in the ADR.

## Releasing a change

**This is a public repo — make changes on a feature branch and open a PR;
don't commit or push directly to `main`.** Branch → commit → push branch →
open PR → review → merge. `main` is what the demo deploys from, so keep it
green and let changes land through review.

aito-demo-server tracks each demo's `main` branch (`demos.config.yaml`,
entry `equity`, path-mounted at `/equity`, no frontend build). So:

1. PR your change in this repo, merge to `main`
2. In aito-azure, inside its nix-shell (that is where `az` lives):
   `nix-shell --run "./do build-demos"` → prints a date tag and
   `resolved equity: main → <sha>`
3. `./do deploy-demos-test <tag>` → smoke-test at
   https://aito-demos-unified-test.azurewebsites.net/equity/ (on the test app
   a `Host: demos.aito.ai` header 404s; use the default hostname)
4. `./do deploy-demos <tag>` — production, restarts the **shared** Web App
   (all demos). It asks for a typed `yes`; non-interactively:
   `echo yes | ./do deploy-demos <tag>`

Data changes that need Aito (a new column, a reload) go to a branched env
first (`AITO_ENV=<env>`), then promote.

## Files you can ignore as an agent

- `.venv/`, `__pycache__/` — local installs / caches
- `data/` (except its `.gitkeep`) — pipeline-generated; not committed
- `books/.out/` — booktest output, regenerated by tests
- `*.lock` — managed by `uv`, don't hand-edit

## Pointers

- [aito-equity-demo-TASK.md](./aito-equity-demo-TASK.md) — the full task
  brief: methodology, data pipeline, four views, demo moments, scope limits.
- [aito-equity-demo.html](./aito-equity-demo.html) — the design
  source-of-truth (single-file editorial prototype). `site/index.html` is
  the descended implementation.
- [CHEATSHEET.md](./CHEATSHEET.md) — Aito query cookbook
- [CHECKLIST.md](./CHECKLIST.md) — pre-launch checklist
- `/home/arau/episto/src/aito-demo-server/README.md` — the platform that
  hosts this demo
- `/home/arau/episto/src/aito-azure/do` — deploy mechanics (don't run from
  here; this demo is one of N)
