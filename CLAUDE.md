# CLAUDE.md — agent instructions for aito-equity-demo

This file is read by Claude Code (and similar agents) when working in this
repo. Keep it tight; put narrative in `docs/` if it grows.

## What this is

A demo of the **unstructured-judgment → LLM grading → predictive database**
pattern, vehicle = long-horizon equity research (Buffett-style: moat,
position, market quality, leadership). Primary audience: a large quantitative trading firm. Secondary:
public artifact (HN/LinkedIn/blog) between the two meetings.

**Deployment target:** static site at `equity.aito.ai`, served via the
`aito-demos-unified` platform alongside the other `*.aito.ai` demos.

**The demo is the methodology.** The numbers in the UI come from real
Aito queries against real data — placeholder JSON in `site/data/` exists
only for the period between repo bootstrap and the first pipeline run.

## Stack

This demo **deviates from the demo-server default** (which is FastAPI +
Next.js with a shared shell). The equity demo has its own editorial
visual identity (single-file HTML, Fraunces / Source Serif 4 type, paper
texture, double-rule dividers) and no need for a runtime backend.

| Layer | Lives at | Notes |
|---|---|---|
| Data pipeline | `pipeline/` (Python 3.12) | Stages: universe → filings → extract → outcomes → load → precompute |
| Static site | `site/index.html` + `site/data/*.json` | Single-file HTML with inline CSS/JS; fetches JSON at runtime |
| Health stub | `src/app.py` (FastAPI, ~30 lines) | Serves `site/` + `/health` + `/api/health`. The aito-demos-unified platform expects a uvicorn process per demo; this stub satisfies that contract. **No /api/* routes are added here.** |
| Aito | external | Predictive database; pipeline writes to it, then precomputes query results to JSON |

There is **no request-time Aito call**. Every value the user sees in the
browser was precomputed by `./do pipeline precompute` and written to
`site/data/`. This is intentional:

- API key never reaches the browser
- Demo works offline (after build)
- Reproducible from the four notebooks
- Deploy is just a tarball of `site/`

## Conventions enforced by the platform — DO NOT drift

| Convention | Lives at |
|---|---|
| Health stub entry | `src/app.py` defines `app: FastAPI` |
| Static mount | `app.mount("/", StaticFiles(directory="site", html=True))` — must be LAST so it doesn't shadow `/health` |
| `/health` | Returns 200 cheaply, no Aito call |
| `/api/health` | Returns 200 + `backend: "static-only"` |
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
  But: this demo's design says **don't add /api/* routes at all** — if you
  reach for one, ask whether the data should be precomputed instead.
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

aito-demo-server tracks each demo's `main` branch by default. So:

1. PR your change in this repo
2. Merge to `main`
3. In aito-azure: `./do deploy-demos` — rebuilds the unified image (latest
   `main` is pulled at build time via `git ls-remote`) and updates the
   Web App

Note: the platform may need adjustment for this demo since there's no
Next.js build step. If `demos.config.yaml`'s `build.frontend_build` for
this demo still says `cd frontend && npm ci && npx next build`, it needs
to become a no-op (or `cd .` placeholder). Coordinate before deploying.

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
