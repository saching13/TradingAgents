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

Report brief status updates to the user using `reports_completed`/`reports_total` from the response.

Stop polling when `status` is `"done"` (success) or `"error"` (failure — report the `error` field to the user).

**5. Save the report**

When "Save report? [Y]:" appears:
```bash
tmux send-keys -t ta_run "Y" Enter
tmux send-keys -t ta_run "" Enter   # accept default save path
```

When "Display full report?" appears:
```bash
tmux send-keys -t ta_run "N" Enter
```

**6. Report back**

Tell the user:
- Final decision (look for `Final Investment Decision:` or `FINAL TRANSACTION PROPOSAL:` in the captured pane)
- Total run time (bottom status bar `⏱ MM:SS`)
- Save path printed after saving

## Timeouts

Each agent typically takes 30–90 seconds with gemma4:12b-hermes. Deep mode with all 4 analysts runs ~7–10 minutes end-to-end. Use a 15-minute poll timeout before declaring failure.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GraphRecursionError` | Increase `TRADINGAGENTS_MAX_RECUR_LIMIT` in `.env` (currently 300) |
| Prompt not advancing | Capture-pane and check what's on screen; resend the key |
| Tmux session already exists | Kill it first: `tmux kill-session -t ta_run` |
| Ollama not running | `ollama serve` in a separate terminal |
