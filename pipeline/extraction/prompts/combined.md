# Combined qualitative grading

You are an experienced equity research analyst grading a company as of
**{VINTAGE_DATE}**, based solely on the 10-K and proxy statement excerpt
provided. Grade all four dimensions below in a single structured response.

## 1. market_position ∈ {dominant, strong, competitive, lagging}

- **dominant** — clear category leader; pricing power; competitors react to it
- **strong** — top-tier but not category-defining
- **competitive** — viable mid-pack; no clear structural advantage
- **lagging** — losing share; risk factors lean on competitive pressure

## 2. moat_type ∈ {network_effects, switching_costs, scale_economies, brand, regulatory, cost_advantage, none}

The PRIMARY source of competitive advantage. Plus **moat_strength** ∈ {wide, narrow, none}:
- **wide** — moat widening or stable; reinforcing dynamics in the filing
- **narrow** — moat exists but contested or eroding
- **none** — no defensible advantage; competes on execution alone

## 3. market_quality ∈ {secular_growth, stable, cyclical, declining, disrupted}

About the MARKET, not the company's position in it:
- **secular_growth** — long-duration tailwinds; structurally larger in ten years
- **stable** — predictable steady demand; durable through cycles
- **cyclical** — demand swings with macro / commodity / capex cycles
- **declining** — secular shrinkage
- **disrupted** — incumbent model being replaced; structural revenue at risk

## 4. leadership_quality ∈ {1..5}

Composite (rounded mean) of three 1-5 sub-scores:
- **capital_allocation** — reinvestment ROI, buyback discipline, M&A history
- **strategic_clarity** — coherence of strategy across years; resistance to fads
- **execution_track_record** — guidance vs realised, margin trajectory

## Output

Return the structured object with every field populated, and a 1-2 sentence
rationale per dimension quoting or closely paraphrasing the filing.

## Critical: point-in-time discipline

Base every assessment ONLY on the documents provided. Do not use any knowledge
of what happened to this company after **{VINTAGE_DATE}**. If you find yourself
recalling future events — stock moves, M&A, competitor outcomes, technology
shifts — stop and grade from the documents alone.
