---
name: analyze
description: Use when the user wants to run a TradingAgents analysis on a stock ticker. Accepts an optional ticker symbol as args (e.g. /analyze NVDA). Submits analyses to the HTTP API and monitors progress automatically.
---

# TradingAgents Analyze

Run a full TradingAgents analysis via the HTTP API.

## Usage

```
/analyze [TICKER] [DEPTH]
```

- `TICKER` — stock symbol (e.g. NVDA, SPY, BTC-USD). Prompt user if omitted.
- `DEPTH` — `shallow`, `medium`, or `deep` (default: `deep`)

## Steps

**1. Resolve args**

If ticker not provided as args, ask the user for it with `AskUserQuestion` before proceeding.

**2. Ensure the API is running**

```bash
curl -sf http://localhost:8080/health || (cd /home/sags/Projects/TradingAgents && docker compose up -d tradingagents-api)
```

Wait until `curl -sf http://localhost:8080/health` returns `{"status":"ok"}` before proceeding.

**3. Submit the analysis**

```bash
curl -s -X POST http://localhost:8080/analyze -H "Content-Type: application/json" \
  -d '{"ticker": "{TICKER}", "depth": "{DEPTH}"}'
```

Extract `job_id` from the JSON response.

**4. Poll for progress**

Every 10 seconds:
```bash
curl -s http://localhost:8080/analyze/{JOB_ID}
```

Progress reporting is binary, not incremental: `reports_completed` stays
`0` for the entire run and only jumps to the final count once `status`
becomes `"done"`. Don't fabricate a fractional progress readout from it —
just tell the user the analysis is "still running" on each poll, and
mention the typical wait duration (see Timeouts below).

Stop polling when `status` is `"done"` (success) or `"error"` (failure — report the `error` field to the user).

**5. Save the report**

Once Step 4's polling shows `status == "done"`, the final `GET /analyze/{JOB_ID}` response body already contains everything needed — no further request is required. Its shape:

```json
{
  "status": "done",
  "reports": {
    "market_report": "...",
    "sentiment_report": "...",
    "news_report": "...",
    "fundamentals_report": "...",
    "quant_report": "..."
  },
  "reports_completed": 5,
  "reports_total": 5,
  "final_decision": "...",
  "error": null
}
```

Format this into a Markdown report and write it to disk with the `Write` tool at `reports/{TICKER}_{TIMESTAMP}/complete_report.md` (e.g. `reports/NVDA_20260701-143000/complete_report.md`), mirroring the section layout `cli/main.py`'s `save_report_to_disk` uses:

```markdown
# Complete Analysis Report for {TICKER} ({TIMESTAMP})

## I. Analyst Team Reports

### Market Analyst
{reports.market_report}

### Sentiment Analyst
{reports.sentiment_report}

### News Analyst
{reports.news_report}

### Fundamentals Analyst
{reports.fundamentals_report}

### Quant Forecast Analyst
{reports.quant_report}

## II. Final Decision
{final_decision}
```

Omit any section whose report key is missing or empty.

**6. Report back**

Tell the user:
- Final decision — the `final_decision` field from the JSON fetched in Step 4/5
- Total run time — the wall-clock time you (the calling agent) observed between submitting the job in Step 3 and it reaching `"done"` in Step 4 (note the time before submitting and again once polling stops, then report the difference)
- Save path — the path you wrote the report to in Step 5

## Timeouts

Each agent typically takes 30–90 seconds with gemma4:12b-hermes. Deep mode with all 5 analysts (market, sentiment, news, fundamentals, quant) runs ~7–10 minutes end-to-end. Use a 15-minute poll timeout before declaring failure.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GraphRecursionError` | Increase `TRADINGAGENTS_MAX_RECUR_LIMIT` in `.env` (currently 300) |
| `curl: (7) Failed to connect` | API isn't running — Step 2 should have started it; check `docker compose ps` |
| Job stuck in `"running"` for a very long time | Check `docker compose logs tradingagents-api` for errors; deep mode can legitimately take 10+ minutes |
| `status: "error"` in the poll response | Report the `error` field to the user directly — don't retry silently |
| Ollama not running | `ollama serve` in a separate terminal |
