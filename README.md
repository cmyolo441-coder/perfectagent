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
- **Temporal Kernel** — every message, tool call, result, and cost is an
  immutable, content-addressed event in an append-only log. State is a pure
  fold of that log, so you can rewind, fork, replay, and verify the timeline
- **Goal contracts** — a goal is a structured object with done-criteria
  clauses and anti-clauses; distance-to-done is a computed number, not a vibe
- **Memory** — closed tasks compress into structured episode records; failed
  approaches land in a dead-end ledger and are blocked deterministically
- **Judge** — claims are verified against reality with deterministic
  predicates (exit codes, file checks, regex) — never the model's own word
- **Swarm** — fan out parallel read-only scout sub-agents; reports land in
  the log without polluting the main context
- **Team** — up to **8 professional worker sub-agents in parallel**
  (researcher / coder / tester / reviewer / analyst), each with real tools
  and a role brief. Reads fan out freely; writes serialise through one
  global lock, so two workers can never mutate the world at once
- **AutoPilot** — the agent decides **for itself** what each turn needs and
  enables it automatically: parallel team when subtasks are independent,
  goal mode when the request is a verifiable mission, real-time web when
  the question needs live data. Every decision is logged and shown live
- **Real-time web** — `web_search` hits DuckDuckGo with a Bing fallback and
  stamps every result with its retrieval time, so the agent answers with
  current facts, not stale knowledge

Risky tools (writes, edits, shell, delete, move, copy) ask for approval
**inside the app** — the bottom border becomes an approval bar:
press `y` (yes), `n` (no), or `a` (always). Toggle globally with `/approve`.
The **autonomy ladder** (`/autonomy 0-5`) controls how much the agent may do
without asking, from read-only observer to fully autonomous.

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

Event-log commands:

- `/goal set <statement> | <clause1> | <clause2>` — set a goal contract
  (prefix a clause with `!` to make it an anti-clause);
  `/goal done <clause>` · `/goal status` · `/goal clear`
- `/autonomy <0-5>` — observer → advisor → assistant → collaborator (default)
  → pilot → autonomous
- `/state` — live projection of the event log (cost, goal, dead-ends, verdicts)
- `/rewind <seq>` — rewind the timeline (bare `/rewind` lists recent seqs)
- `/fork [name]` — branch the timeline and continue on the fork
- `/verify` — verify the event log's Merkle spine
- `/memory` — recent episodes + dead-end ledger
- `/judge <type> <arg>` — deterministic check (`exit_code`, `file_exists`,
  `file_contains`, `file_matches`, `command_output_contains`), or pass a full
  JSON predicate
- `/scout <q1> | <q2> | …` — parallel read-only scout sub-agents
- `/team <task1> | <task2> | …` — parallel worker team (up to 8); prefix a
  task with `role:` to pin it (`researcher:` `coder:` `tester:` `reviewer:`
  `analyst:`)
- `/auto [on|off|status]` — the AutoPilot self-routing brain (on by default)
- `/prompt [main|master|list]` — choose the system prompt: `main` (compact)
  or `master` (the extended 130k+ specification prompt)
- `/mastermind` — the prompt-coherence ledger (sealed prompts, gate,
  composed context, lineage)

## Effort levels

`low` · `medium` · `high` · `extrahigh` · `ultrahigh` — each raises max
tokens, temperature, and reasoning effort.

## Models & providers

Three OpenAI-compatible providers are built in:

- **OpenCode Zen** (`https://opencode.ai/zen/v1`) — mimo-v2.5-free,
  big-pickle, grok-code-fast-1, claude-sonnet-4-5, claude-opus-4-6,
  gemini-3.1-pro, gpt-5.2
- **TokenRouter** (`https://api.tokenrouter.com/v1`) — qwen/qwen3.8-max-free,
  deepseek-ai/DeepSeek-V3.2, moonshotai/Kimi-K2-Instruct
- **Agnes** (`https://apihub.agnes-ai.com/v1`) — agnes-2.5-flash (fast,
  tool-capable, reasoning-aware)

API keys are embedded; override with `OPENCODE_API_KEY` /
`TOKENROUTER_API_KEY` / `AGNES_API_KEY` environment variables. Config
(model, effort, auto-approve) persists in `~/.fullagent/config.json`;
sessions save to `~/.fullagent/sessions/`.

## Layout

```
main.py            launcher — python main.py
fullagent/
  __init__.py      package
  config.py        providers, models, effort levels, paths
  systemprompt.py  the ONE home of every system prompt (single source)
  mastermind.py    prompt coherence: sealed vault, gate, composer, lineage
  tools.py         16 tools: files, shell, search, real-time web
  client.py        streaming OpenAI-compatible client (SSE, retries, cancel)
  agent.py         agent loop: LLM <-> tools, event-sourced on the kernel
  kernel.py        Temporal Kernel: append-only, content-addressed event log
  memory.py        episodic memory + dead-end ledger (fold-derived)
  goal.py          goal contracts with machine-checkable done-criteria
  judge.py         deterministic verification predicates (no LLM judging)
  swarm.py         parallel read-only scout sub-agents
  team.py          parallel worker team (up to 8, role-based, write lock)
  autopilot.py     self-routing: auto team / goal mode / real-time web
  tui.py           persistent double-line box, overlays, streaming, approval
  __main__.py      entry point
```

The event log lives at `~/.fullagent/eventlog.jsonl` (override the directory
with `FULLAGENT_HOME`). Each module ships a self-test:
`python -m fullagent.kernel` (and `.memory`, `.goal`, `.judge`, `.swarm`,
`.team`, `.autopilot`, `.systemprompt`, `.mastermind`).

## System prompts — one file, one delivery path

Every system prompt the model ever sees lives in **`fullagent/systemprompt.py`**
and nowhere else. `agent.py`, `swarm.py` and `team.py` contain **no inline
prompt strings** — they only import from that file. Two structural guarantees:

- **Single source of truth.** Edit a prompt in `systemprompt.py` and it
  changes everywhere at once — main agent, scouts, and all worker roles.
- **One delivery path.** Every message list is built through
  `systemprompt.with_system()`, which guarantees the correct prompt sits at
  position 0 before any request is sent. A model can never be called without
  its prompt, and can never see a stale or partial one.

Two prompts ship in the registry, switchable live with `/prompt`:

| Name | Size | What it is |
|---|---|---|
| `main` | ~1.6k chars | the compact sovereign-agent prompt (default) |
| `master` | **136,928 chars** | MAIN + the full master specification (`project.txt`) embedded — the entire architecture, invariants, subsystem contracts and Goal-Mode grammar in context |

Add more prompts later by dropping a constant in `systemprompt.py` and
registering it in the `PROMPTS` map (or call `register()` at runtime).

## Output budget — 200k tokens

Every effort level requests **200,000 output tokens** (`config.MAX_TOKENS`).
Backends with a lower hard ceiling (e.g. Agnes caps at 65,536) are clamped
per-provider at send time in `client.py`, so the request is never rejected
for an oversized `max_tokens`.

## Mastermind — coherence, not coercion

`fullagent/mastermind.py` makes following `systemprompt.py` *inevitable* —
not by telling the model "you must obey", but by making the sealed prompt
the only coherent center of every request. Three cooperating mechanisms,
all deterministic Python:

| Mechanism | What it does |
|---|---|
| **PromptVault** | Every prompt is sealed with a sha256 fingerprint and recorded in the event log. The vault is the only source a model ever reads a prompt from; prompts registered at runtime are sealed on demand, and a changed prompt is re-sealed — no stale copy is ever served. |
| **PromptGate** | The single door to the model. Every request (main agent, scout, worker) passes `gate.dispatch()`, which guarantees `messages[0]` carries the sealed prompt byte-for-byte at the front, re-seats it if anything shadowed or corrupted it (an integrity restore — recorded, never punished), and seals a `prompt.dispatch` lineage event. There is no other way to reach the API. |
| **CoherenceComposer** | Live context (constitution, goal, web mode, memory) is never appended as raw text that could compete with the prompt. It is composed beneath the sealed prompt as one coherent document: each section is framed as *input to* the prompt, provenance-tagged, ordered by authority, deduplicated. The prompt stays the only voice giving direction. |

There is no enforcement layer — the system observes and records
(PromptLineage), it never punishes. Every dispatch is sealed into the
event log; inspect the live ledger with `/mastermind`.
