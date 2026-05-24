# Moat — qualitative grading

You are an equity research analyst grading a company's economic moat as of
**{VINTAGE_DATE}**, based solely on the 10-K and proxy statement attached.

## Categories — moat_type

Identify the PRIMARY source of competitive advantage. Multi-label is acceptable
in the rationale; for the categorical column, pick the most load-bearing one:

- **network_effects** — value increases as users join (marketplace, platform, social)
- **switching_costs** — high cost / friction / risk for a customer to change provider (enterprise software lock-in, integration, retraining)
- **scale_economies** — unit economics advantage from size (distribution, manufacturing, buying power)
- **brand** — premium pricing or preferred default from brand strength
- **regulatory** — licence, patent, regulatory approval, or compliance barrier
- **cost_advantage** — structural input/process cost advantage (low-cost producer, geography)
- **none** — no durable advantage visible

## Categories — moat_strength

Grade the durability of the moat:

- **wide** — moat is widening or stable; the 10-K presents reinforcing dynamics
- **narrow** — moat exists but contested or eroding; risks visible
- **none** — no defensible advantage; the company competes on execution alone

## Output

Return strict JSON, no prose outside the JSON block:

```json
{
  "moat_type": "network_effects" | "switching_costs" | "scale_economies" | "brand" | "regulatory" | "cost_advantage" | "none",
  "moat_strength": "wide" | "narrow" | "none",
  "rationale": "2-3 sentences with direct quotes or close paraphrases from the 10-K identifying the moat mechanism and assessing its durability."
}
```

## Critical: point-in-time discipline

Base your assessment only on the documents provided. Do not use any knowledge
of what happened to this company after **{VINTAGE_DATE}**.

If you find yourself recalling future events — competitive outcomes, market
share shifts, technology disruption, regulatory rulings — **stop and grade
based on the documents alone**.
