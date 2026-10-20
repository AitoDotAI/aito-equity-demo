# aito-equity-demo

**Predictive equity research with LLM-graded qualitative features and the
[aito.ai](https://aito.ai) predictive database.**

Live: [demos.aito.ai/equity](https://demos.aito.ai/equity/)

---

## What this is

A test of Warren Buffett's long-horizon investing principles — moat,
market position, market quality, leadership — at scale. We grade
~250 S&P 500 companies on those four qualitative dimensions using
contemporary 10-K and proxy filings (point-in-time, no lookahead), load
the grades alongside quantitative features into Aito, and ask:

> Across hundreds of point-in-time observations, which features most
> predict landing in the 'great' outcome bucket over the next decade?

The answer is the demo. The numbers come from real Aito queries against
real data — there is no narrative shaping of the result. If the data says
quant beats qual, the chart says so.

## What this isn't

**Not a trading signal.** No microstructure, no transaction cost model,
no capacity analysis. Twelve-year horizon, ~250 observations per vintage.
This is a hypothesis test about long-horizon business quality and an
architecture demonstration — not an alpha-generation system.

## The four views

| № | View | Aito query | What it shows |
|---|---|---|---|
| i | The Company File | `POST /_predict` on `outcome_bucket` | Per-focal grades, evidence quotes, calibrated probabilities across five outcome buckets |
| ii | Does the Thesis Hold? | `POST /_relate` with `outcome_bucket = great` | Feature-importance table; the "Buffett thesis quantified" chart, plus a calibration plot |
| iii | Historical Analogues | `POST /_match` for similarity | Six nearest neighbours to the focal company with their actual realised outcomes |
| iv | On Methodology | (static) | Point-in-time discipline, LLM-grading method, lookahead stress test, why this isn't a trading signal |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Python pipeline                          Static site          │
│   ─────────────────                        ───────────          │
│   universe   → data/universe.csv                                │
│   filings    → data/10k_excerpts/                               │
│   extract    → data/llm_features.csv  ─┐                        │
│   outcomes   → data/outcomes.csv       ├─→ Aito ─→ precompute   │
│                                        │           └→ site/data/│
│                                        │                  ↑     │
│                                        │                  │     │
│                                        │           site/index.html  ← browser
│                                        └→ schema.json                │
│                                                                     │
└─────────────────────────────────────────────────────────────────┘
```

No runtime backend. The browser fetches `site/data/*.json` directly;
every value was precomputed at pipeline-run time. API keys never reach
the browser. A minimal FastAPI stub at `src/app.py` serves the static
site and a `/health` endpoint to fit the `aito-demos-unified` platform
contract — nothing else.

## Stack

- **Python 3.12 + uv** — pipeline (`pipeline/`), tests (`tests/`, `book/`),
  health stub (`src/app.py`)
- **Anthropic SDK** — LLM grading at temperature 0.3, modal of three runs
  per feature
- **httpx + lxml + BeautifulSoup** — EDGAR + Wikipedia constituent parsing
- **yfinance** — prices, dividends, splits, terminal events
- **aito.ai** — predictive database (single endpoint family: `/_predict`,
  `/_relate`, `/_match`, no separate model training)
- **HTML/CSS/JS** — single-file editorial frontend (Fraunces / Source Serif 4,
  cream paper texture, double-rule dividers); no Next.js, no React
- **playwright** — screenshot regression
- **FastAPI / uvicorn** — serves the static site plus a read-only Aito query proxy for the interactive views (key stays server-side)

Multi-market columns (`market`, `exchange`, `currency`, `reporting_standard`,
`filing_language`) are in the Aito schema from day 1 even though v1 data
is US-only — Finnish / Nordic / EU datasets slot in without migration.

## Reproducing

```bash
./do install                     # uv sync + playwright browser install
cp .env.example .env             # add AITO_API_URL, AITO_API_KEY, ANTHROPIC_API_KEY
./do pipeline all                # universe → filings → extract → outcomes → load → precompute
./do serve                       # http://localhost:8401
```

All pipeline stages are currently `NotImplementedError` skeletons —
see [aito-equity-demo-TASK.md](./aito-equity-demo-TASK.md) for the build
order. The `site/data/*.json` files in this repo contain placeholder
values mirroring the reference HTML mockup, so the site renders
end-to-end before the pipeline lands.

## Audience

This is a sales / research demo for **[aito.ai](https://aito.ai)**.
The architecture pattern — unstructured judgment → LLM grading →
predictive database — generalises beyond equity research to credit
underwriting, M&A diligence, supplier risk, and talent assessment.
Equity is the vehicle; the architecture is the product.

## See also

- [aito-equity-demo-TASK.md](./aito-equity-demo-TASK.md) — the full
  build brief: methodology, build order, demo moments, scope limits
- [aito-equity-demo.html](./aito-equity-demo.html) — the visual
  design source-of-truth (single-file prototype)
- [CLAUDE.md](./CLAUDE.md) — agent / contributor instructions
- [CHEATSHEET.md](./CHEATSHEET.md) — Aito query reference
- `notebooks/` — public reproducibility layer (pending)
- `ADRs/` — six method decisions (pending)
