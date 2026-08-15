# FullAgent

An advanced terminal AI agent — pure Python, real code, world-class TUI.

```
╭─ FullAgent ── model: MiMo v2.5 FREE ── effort: HIGH ── session a1b2c3d4 ─╮
│ ❯ type anything… the agent reads, writes, edits files, runs commands     │
╰─ Enter send · Esc+Enter newline · / commands · Ctrl+T models ────────────╯
```

The prompt is a **double-line box** (not full-screen). One application runs
for the whole session, so the box **never disappears** — while the model
works, the bottom border shows a live animated status:

```
╰ ⠹ thinking…  ·  Ctrl+C cancel ────────────────────────────────────────────╯
```

Tokens stream live above the box, tool calls appear as `⚙ name args` with
`✓`/`✗` results, and each turn ends with a stats line (`2.3s · 512→87 tokens`).

## Run

```bash
pip install prompt_toolkit rich requests
python main.py
```

(ya `python -m fullagent` — dono same hain)

## What it can do

- **Files** — read (with line numbers), write, exact-string edit, list, info,
  create dirs, copy, move, delete
- **Shell** — run any bash command, capture exit code + stdout + stderr
- **Search** — regex search through file contents (ripgrep-style), glob files
- **Web** — fetch URLs, search the web
- **Agent loop** — the model keeps calling tools until the task is genuinely
  done (up to 40 iterations), then summarizes

Risky tools (writes, edits, shell, delete, move, copy) ask for approval
**inside the app** — the bottom border becomes an approval bar:
press `y` (yes), `n` (no), or `a` (always). Toggle globally with `/approve`.

## Keys

| Key | Action |
|---|---|
| `Enter` | send |
| `Esc+Enter` | newline inside the box |
| `/` | slash-command completion menu |
| `Ctrl+T` or `/model` | model selector |
| `Ctrl+E` or `/effort` | effort selector |
| `↑↓` / `PgUp` / `PgDn` / `Tab` / `Home` / `End` | navigate selectors **and** the `/` completion menu |
| `Ctrl+R` | search input history |
| `Ctrl+L` | clear screen |
| `Ctrl+X Ctrl+E` | open input in $EDITOR |
| `Ctrl+C` | cancel a running turn (mid-stream) / clear input |
| `Ctrl+D` | quit |

## Slash commands

`/model` `/effort` `/help` `/history` `/new` `/save` `/approve`
`/reasoning` `/usage` `/clear` `/about` `/exit`

## Effort levels

`low` · `medium` · `high` · `extrahigh` · `ultrahigh` — each raises max
tokens, temperature, and reasoning effort.

## Models & providers

Two OpenAI-compatible providers are built in:

- **OpenCode Zen** (`https://opencode.ai/zen/v1`) — mimo-v2.5-free,
  big-pickle, grok-code-fast-1, claude-sonnet-4-5, claude-opus-4-6,
  gemini-3.1-pro, gpt-5.2
- **TokenRouter** (`https://api.tokenrouter.com/v1`) — qwen/qwen3.8-max-free,
  deepseek-ai/DeepSeek-V3.2, moonshotai/Kimi-K2-Instruct

API keys are embedded; override with `OPENCODE_API_KEY` /
`TOKENROUTER_API_KEY` environment variables. Config (model, effort,
auto-approve) persists in `~/.fullagent/config.json`; sessions save to
`~/.fullagent/sessions/`.

## Layout

```
main.py            launcher — python main.py
fullagent/
  __init__.py      package
  config.py        providers, models, effort levels, paths
  tools.py         14 tools: files, shell, search, web
  client.py        streaming OpenAI-compatible client (SSE, retries, cancel)
  agent.py         agent loop: LLM <-> tools with events
  tui.py           persistent double-line box, overlays, streaming, approval
  __main__.py      entry point
```
