#!/usr/bin/env bash
# Local-dev wrapper for aito-equity-demo.
#
# Architecture: Python data pipeline → site/data/*.json → static site/index.html.
# A minimal uvicorn stub (src/app.py) serves the static site and a /health endpoint
# to satisfy the aito-demos-unified platform contract; there is no runtime backend.
#
#   ./do install              uv sync + playwright browser install (one-time)
#   ./do serve                run the static-serving stub (uvicorn on :8401)
#
#   ── Pipeline (run in this order; each stage produces input for the next) ──
#   ./do pipeline universe    reconstruct point-in-time index constituents → data/universe.csv
#   ./do pipeline filings     fetch 10-K + DEF 14A before each vintage → data/10k_excerpts/
#   ./do pipeline extract     LLM grade qualitative features → data/llm_features.csv
#                             (requires ANTHROPIC_API_KEY)
#   ./do pipeline outcomes    forward returns + survival → data/outcomes.csv
#   ./do pipeline load        push merged companies table → Aito instance
#                             (requires AITO_API_URL + AITO_API_KEY)
#   ./do pipeline precompute  emit site/data/*.json from real Aito queries
#   ./do pipeline all         run every stage in order
#
#   ── Tests ────────────────────────────────────────────────────────────────
#   ./do test                 run pytest (tests/ + book/)
#   ./do test-book            run booktest snapshot tests only (book/)
#
#   ── Visuals ──────────────────────────────────────────────────────────────
#   ./do screenshot-teaser    render assets/teaser.html → assets/teaser.png (1200×630)
#   ./do screenshot-pages     full-page desktop screenshots of /
#   ./do inspect-mobile       iPhone-sized screenshots of /
#
#   ./do clean                wipe build artifacts (.pytest_cache, scripts/output)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Load .env so cred-gating in `pipeline all` works at the shell level (the
# Python modules also load it via python-dotenv; this is for the bash checks).
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PORT="${PORT:-8401}"

die() { echo "✗ $*" >&2; exit 1; }
say() { echo "→ $*"; }

cmd_install() {
  command -v uv >/dev/null 2>&1 || die "uv not found (see https://docs.astral.sh/uv/)"
  say "uv sync"
  uv sync
  say "playwright install chromium (for screenshot scripts)"
  uv run playwright install chromium
}

cmd_serve() {
  say "site → http://localhost:${PORT} (serving site/ via uvicorn stub)"
  exec uv run uvicorn src.app:app --host 0.0.0.0 --port "$PORT" --reload
}

cmd_pipeline() {
  local stage="${1:-help}"
  shift || true
  case "$stage" in
    universe)     uv run python -m pipeline.universe.sp500 "$@" ;;
    filings)      uv run python -m pipeline.filings.edgar "$@" ;;
    extract)      uv run python -m pipeline.extraction.extract "$@" ;;
    outcomes)     uv run python -m pipeline.outcomes "$@" ;;
    fundamentals) uv run python -m pipeline.fundamentals.sec_xbrl "$@" ;;
    market)       uv run python -m pipeline.fundamentals.market_factors "$@" ;;
    load)         uv run python -m pipeline.aito.load "$@" ;;
    precompute)   uv run python -m pipeline.aito.queries "$@" ;;
    all)
      # Free, deterministic stages always run.
      say "universe"     && cmd_pipeline universe "$@"
      say "filings"      && cmd_pipeline filings
      say "outcomes"     && cmd_pipeline outcomes
      say "fundamentals" && cmd_pipeline fundamentals
      say "market"       && cmd_pipeline market
      # Cost-gated: extraction calls the LLM. Requires explicit --confirm-cost.
      if [ -n "${OPENAI_MODEL_API_KEY:-}" ]; then
        say "extract (LLM — needs --confirm-cost to actually spend; dry-run otherwise)"
        cmd_pipeline extract --resume
      else
        say "extract — SKIPPED (OPENAI_MODEL_API_KEY not set)"
      fi
      # Cred-gated: load + precompute need a live Aito instance.
      if [ -n "${AITO_API_URL:-}" ] && [ -n "${AITO_API_KEY:-}" ]; then
        say "load"       && cmd_pipeline load
        say "precompute" && cmd_pipeline precompute
      else
        say "load + precompute — SKIPPED (AITO_API_URL / AITO_API_KEY not set)"
        say "  → run './do pipeline precompute --static-only' to refresh meta/universe from local CSVs"
      fi
      ;;
    help|-h|--help|"")
      sed -n '/── Pipeline/,/── Tests/p' "$0" | sed -n '/^#/p'
      ;;
    *) die "unknown pipeline stage: $stage (run './do pipeline help')" ;;
  esac
}

cmd_test()      { exec uv run pytest "$@"; }
cmd_test_book() { exec uv run pytest book/ "$@"; }

cmd_screenshot_teaser() { uv run python -m scripts.screenshot_teaser "$@"; }
cmd_screenshot_pages()  { uv run python -m scripts.screenshot_pages "$@"; }
cmd_inspect_mobile()    { uv run python -m scripts.inspect_mobile "$@"; }

cmd_clean() {
  rm -rf .pytest_cache scripts/output/*
  find . -type d -name __pycache__ -prune -exec rm -rf {} +
  say "cleaned"
}

cmd_help() { sed -n '1,40p' "$0" | sed -n '/^#/p'; }

case "${1:-help}" in
  install)             shift; cmd_install "$@" ;;
  serve)               shift; cmd_serve "$@" ;;
  pipeline)            shift; cmd_pipeline "$@" ;;
  test)                shift; cmd_test "$@" ;;
  test-book)           shift; cmd_test_book "$@" ;;
  screenshot-teaser)   shift; cmd_screenshot_teaser "$@" ;;
  screenshot-pages)    shift; cmd_screenshot_pages "$@" ;;
  inspect-mobile)      shift; cmd_inspect_mobile "$@" ;;
  clean)               shift; cmd_clean "$@" ;;
  help|-h|--help)      cmd_help ;;
  *) die "unknown command: $1 (run './do help')" ;;
esac
