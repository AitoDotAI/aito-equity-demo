# Aito Equity Demo — Claude Code Task

## What This Is

A sales/research demo for Aito.ai showing how the predictive database handles
the *qualitative-meets-quantitative* domain: LLM-extracted features from 10-Ks
feeding an inference engine to test long-horizon investing principles
(Buffett-style: moat, market position, market quality, leadership).

The demo is the **vehicle**, not the product. The architectural pattern —
unstructured judgment → LLM grading → predictive database — generalises to
credit underwriting, M&A diligence, supplier risk, talent assessment.

**Primary audience:** a large quantitative trading firm. Secondary: public artifact (HN, LinkedIn, blog) timed between
the two client meetings as a credibility signal.

**Anti-audience:** day traders, wallstreetbets. The editorial aesthetic is
deliberately self-selecting.

**Deployment target:** Static site at `equity.aito.ai` alongside
`accounting.aito.ai`, `erp.aito.ai`, `ecommerce.aito.ai`. Single shareable URL.

---

## Reference Design

`aito-equity-demo.html` in this folder is the source of truth for visual design,
content, and structure. Build to match it exactly — do not redesign.

The aesthetic is **editorial / Economist-Berkshire-letter**:
- Cream paper background `#f4ede1` with subtle SVG noise grain overlay
- Fraunces serif for display (700/600/500), Source Serif 4 for body
- JetBrains Mono for tickers, code, latencies
- Burgundy `#8b1a1a` accent (qualitative bars, eyebrows)
- Deep navy `#1d3a5f` secondary (quantitative bars)
- Gold `#a8862c` for highlights and active states
- Double-rule section dividers, roman numerals, footnoted statements
- The visual register IS the marketing — don't sand the edges off

### Three-pane layout

**Left sidebar** — dark ink `#1a1612` masthead style. Eyebrow caps, large
serif title with italic `<em>` accent in gold, italic deck subtitle. Nav items
numbered with roman numerals in italic serif. Gold `#a8862c` active-border
+ tinted background. Footer in italic small caps.

**Center content** — cream paper with editorial article structure: eyebrow
caps in burgundy, large Fraunces title with italic word, italic deck subtitle,
small-caps byline rule. White-paper-ish `#fbf6ec` cards on cream. Pullquotes
with burgundy left border.

**Right Aito panel** — deep indigo `#0c0f41`, identical styling to the live
aito-demo ContextPanel and the ecommerce-demo right panel:
- `aito..` wordmark in JetBrains Mono, teal `#12B5AD` on the `..`
- Stats row: 4-stat grid, teal values, purple-tinted gradient header backdrop
- Endpoint badges: `#9B69FF` purple with translucent border
- Section labels: `#9B69FF` uppercase
- Code block: `rgba(255,255,255,0.07)` translucent bg, syntax-coloured
- Bullet markers: teal `◆`
- CTA pinned to bottom: teal "Start free trial →" with glow hover

Both sidebar and Aito panel collapsible via topbar toggle buttons (same as
ecommerce demo). On mobile (≤900px) both panels become overlay drawers.

---

## Stack

Match `aito-ecommerce-demo` stack. If that's plain HTML/JS, stay there for v1.
If it's Next.js + TypeScript, use the same. Do not introduce divergent tooling.

EU hosting, no PII. Dataset is public-market data only — disclose source in
the Aito panel footer.

---

## Data Pipeline

### Universe construction (point-in-time, the methodologically critical part)

**Sources:**
- Wikipedia S&P 500 historical constituent list — parse page history to
  reconstruct membership as of `2014-01-01`, `2017-01-01`, `2020-01-01`
- yfinance for prices, returns, survival, delisting events
- SimFin free tier or SEC EDGAR direct for fundamentals at vintage date
- SEC EDGAR for 10-K and DEF 14A filings filed in the year prior to each vintage

**Three vintages × ~500 companies = ~1,500 raw observations.** After dedup
and dropping rows with insufficient filing data, expect ~750-800 final rows.

**For v1, ship with one vintage (2017) and ~250 companies.** This is enough
to populate every view credibly. The full three-vintage version is the
post-June-9 polish work.

**Outcome variables** (computed forward from vintage date to today):
- `outcome_bucket`: {disaster, poor, market, good, great}
  thresholds: -50%+, -50–0%, 0–100%, 100–300%, 300%+
- `survived_intact`: still independently trading today (boolean)
- `beat_sp500_by_2x`: outperformed SPY total return by 2x+ over the window

For acquired companies, use acquisition price as terminal value. For
bankruptcies/delistings, terminal value = 0 unless residual recoverable.

### Quantitative features (~10-12, at vintage date)

- `sector`, `industry`, `market_cap_bucket` (mega/large/mid)
- `pe_ratio`, `pb_ratio`, `ev_ebitda`
- `roic_5y_avg`, `gross_margin`, `operating_margin`
- `revenue_cagr_5y`, `debt_to_equity`
- `years_public`, `founder_still_ceo` (boolean, manual or LLM-derived)

### Qualitative features (LLM-extracted)

Four features, four focused prompts. Run each at `temperature=0.3`, three
times, take modal answer. Store rationale strings as separate columns —
these become the explainability layer in the UI.

**`market_position`** ∈ {dominant, strong, competitive, lagging}
**`moat_type`** ∈ {network_effects, switching_costs, scale_economies, brand,
regulatory, cost_advantage, none} (multi-label OK; primary moat for the col)
**`moat_strength`** ∈ {wide, narrow, none}
**`market_quality`** ∈ {secular_growth, stable, cyclical, declining, disrupted}
**`leadership_quality`** ∈ {1, 2, 3, 4, 5} (composite of capital_allocation,
strategic_clarity, execution_track_record sub-scores)

**Critical: point-in-time prompting.** Each prompt includes ONLY the 10-K
and proxy from the year before vintage. Every prompt ends with:

> "Base your assessment only on the documents provided. Do not use any
> knowledge of what happened to this company after [VINTAGE_DATE]. If you
> find yourself recalling future events, stop and grade based on documents
> alone."

This isn't a perfect defence against LLM training-data leakage, but it's
documentable, and the 2020 vintage serves as a leakage stress test
(if features predict equally well across vintages, leakage is bounded).

**Cost budget:** ~250 companies × 4 prompts × 3 runs × ~3K tokens ≈ 9M tokens.
Roughly $30-50 with Sonnet. Use Haiku for cheaper grades (market_position,
market_quality) if budget tightens.

### Schema

One row per (company, vintage). Final table `companies`. Roughly:

```
ticker (str, key)
vintage_year (int, key)
company_name (str)
sector (str)
market_cap_bucket (str)
-- quant features --
pe_ratio (float)
roic_5y_avg (float)
... etc
-- LLM features --
market_position (str)
moat_type (str)
moat_strength (str)
market_quality (str)
leadership_quality (int)
-- rationale strings (text, displayed in UI) --
market_position_rationale (str)
moat_rationale (str)
market_quality_rationale (str)
leadership_rationale (str)
-- outcomes --
outcome_bucket (str)
survived_intact (bool)
total_return_pct (float)
window_years (int)
```

---

## The Four Views

Identical structure to the reference HTML. Each view is one Aito query type.

### View 1: The Company File
**Query:** `POST /_predict` on `outcome_bucket`
Focal company selector (chips at top), profile card with all four qualitative
grades and their evidence quotes, then prediction bars with calibrated
probabilities across all five outcome buckets. The "great" bar has the gold
gradient + glow.

Four reference companies to ship with: **NVDA·'14, SHLD·'14, COST·'17, META·'20**.
These four together tell a complete story: wide-moat winner, dominant-but-broken
moat loser, durable compounder, recovery story. Switching chips swaps data.

### View 2: Does the Thesis Hold?
**Query:** `POST /_relate` with `where: {outcome_bucket: "great"}`
The money chart. Feature lift table with burgundy bars for qualitative,
navy bars for quantitative. The headline result: top five lifts are all
qualitative. Below: calibration plot showing predicted vs. realised
frequency by decile.

Both bars and calibration plot can be SVG or pure CSS — match the
reference HTML implementation.

### View 3: Historical Analogues
**Query:** `POST /_match` for similarity search
6 nearest neighbours to NVDA·'14 shown in two rows of three cards.
Each card: ticker, vintage, similarity score, brief profile, actual
outcome. Five positive analogues + one cautionary (EMC, acquired).

### View 4: On Methodology
Six numbered method cards (point-in-time universe, LLM grading discipline,
lookahead stress test, outcome bucketing, why this is not a trading signal,
the transferable lesson). Below: data sources / engine stack two-column.

This view is critical for credibility with a quant audience. The quant's first question will be about lookahead bias. The methodology view
has to be visibly the most polished, not an afterthought.

---

## Aito Panel — Per-View Content

Already specified in the reference HTML. Each view has 2-3 sections in the
right panel:
- View 1: The Problem / The Aito Approach / Why it matters here
- View 2: The Question Asked / What this demonstrates
- View 3: The Query / Architectural Note
- View 4: The Pattern / Transferable Domains

Sections swap on view change. CTA stays pinned.

---

## Build Order

1. **Universe construction script** — single-vintage (2017), 250 companies.
   Output: `data/universe_2017.csv` with quant features and outcomes only.
   Verify outcome distribution looks sensible before going further.

2. **LLM extraction pipeline** — `extraction/extract.py`. Run on 20 companies
   first, manually spot-check rationales for hallucination and lookahead.
   *If spot-check fails, fix prompts before scaling.* Then full 250.

3. **Aito loading** — schema.json, load.py. Push to dev instance. Run sample
   `_predict`, `_relate`, `_match` queries from a notebook. Verify lift values
   come out roughly matching the reference HTML mockup. If not, debug the
   data — the mockup numbers are realistic, not arbitrary.

4. **Frontend wiring** — start from the reference HTML, replace mock data
   with live Aito queries. Keep mock mode working as fallback for offline
   demos. The reference HTML's view-switching, panel-swapping, and chip-
   switching logic already exists in the file.

5. **Notebooks** — `01_thesis.ipynb`, `02_predict.ipynb`, `03_relate.ipynb`,
   `04_similarity.ipynb`. These are the public artifact's reproducibility
   layer. Each notebook produces the corresponding view's data.

6. **ADRs** — write last, when decisions are settled. Six ADRs corresponding
   to the six methodology cards.

---

## Repo Layout

```
aito-equity-demo/
├── README.md                    # Thesis + how to run + demo URL
├── data/
│   ├── universe.csv             # Point-in-time snapshots
│   ├── llm_features.csv         # Qualitative grades + rationales
│   ├── outcomes.csv             # Forward returns, survival
│   └── 10k_excerpts/            # Source material per company (gitignored)
├── extraction/
│   ├── prompts/
│   │   ├── market_position.md
│   │   ├── moat.md
│   │   ├── market_quality.md
│   │   └── leadership.md
│   ├── extract.py
│   └── validation/
│       └── spot_check.ipynb
├── aito/
│   ├── schema.json
│   ├── load.py
│   └── views/
│       ├── predict_outcome.json
│       ├── relate_features.json
│       └── match_similar.json
├── frontend/                    # Static demo site
│   ├── index.html               # Built from aito-equity-demo.html reference
│   ├── styles/
│   ├── scripts/
│   └── api/                     # Thin proxy layer to Aito instance
├── notebooks/
│   ├── 01_thesis.ipynb
│   ├── 02_predict.ipynb
│   ├── 03_relate.ipynb
│   └── 04_similarity.ipynb
├── ADRs/
│   ├── 0001-point-in-time-universe.md
│   ├── 0002-llm-grading-methodology.md
│   ├── 0003-outcome-bucketing.md
│   ├── 0004-lookahead-stress-test.md
│   ├── 0005-not-a-trading-signal.md
│   └── 0006-transferable-architecture.md
└── shell.nix
```

---

## Demo Moments — Do Not Break

These are the five things the demo MUST show, in priority order:

1. **NVDA·'14 prediction** — 54.5% probability on `great`. The visceral "wow"
   moment. Showed dominant + wide moat + secular growth + founder-led to the
   engine; engine bet on the right bucket.

2. **The relate chart** — top five features by lift are all qualitative,
   burgundy bars visibly longer than navy. The "Buffett thesis quantified"
   chart. This is the screenshot that gets reshared.

3. **Calibration plot** — predicted vs. realised by decile, well-aligned.
   The credibility moment. "When the engine says 80%, it's right 79% of the
   time." This is the quant's first sanity check.

4. **EMC cautionary analogue** — the one negative case in the similarity
   panel. Demonstrates that similarity search surfaces nuance (market_quality
   shift), not just confirmation bias.

5. **The four-grade evidence quotes** — each qualitative grade has an LLM
   rationale quote visible in the UI. Makes "explainable by construction"
   real, not a slogan.

Claude Code: if you find yourself "simplifying" any of these (collapsing the
relate chart to a single bar, hiding the rationale quotes, dropping the
calibration plot), STOP. They are the demo.

---

## Scope Limits

**In:** four views matching reference HTML, one vintage (2017), 250 companies,
live Aito queries, mock-mode fallback, four notebooks, six ADRs, README.

**Out:** Three-vintage build (post-June-9), real-time price feeds, sector
breakdowns beyond the relate view, user accounts, persistence, mobile
optimisation beyond the responsive breakpoints already in the reference HTML.

**Out, emphatically:** any framing as a trading signal, any backtest with
transaction costs, any "alpha" language, any leaderboard of "top picks"
for today. Method card № v exists specifically to fence this off.

---

## Known Issues to Watch

- **Wikipedia constituent parsing** — page history is messy. Sanity-check
  the 2017 list against any archived constituent list (e.g. SlickCharts
  historical snapshots) before treating it as ground truth.
- **EDGAR rate limits** — 10 req/s with proper User-Agent header. Cache
  filings to local disk; never re-download.
- **yfinance gaps** — some delisted tickers have incomplete history. Track
  delisting dates separately; don't assume `NaN` price = zero terminal value.
- **LLM training-data leakage** — unavoidable but bounded. Document it in
  ADR 0002. Use the 2020 vintage as the empirical leakage probe later.
- **Survivorship bias in 10-K availability** — bankrupt companies stop
  filing. If 2014 10-K can't be retrieved for a company that died by 2018,
  document the exclusion; don't silently drop.
- **Aito API path** — verify `/_predict`, `/_relate`, `/_match` against the
  current Query2 API before writing client code. The reference HTML uses
  the conceptual paths from the docs.

---

## Quality Bar

The methodology has to be defensible to a quant on first read. A small
clean dataset is better than a large compromised one. If Week 1 reveals
universe construction is slipping, cut to 150 companies rather than skip
point-in-time discipline.

The visual identity is part of the credibility. The editorial aesthetic
self-selects for the audience we want — serious quants and researchers,
not day traders. Do not "modernise" or "soften" the design toward generic
SaaS aesthetics.
