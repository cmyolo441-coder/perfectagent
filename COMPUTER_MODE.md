# FullAgent 3.2 — `/on` Computer Mode

## Yeh kya hai?

`/on` terminal mein ek live **mission-control workspace** kholta hai. Project goal dene par eight independent API-backed specialist sessions research, planning, implementation aur review karte hain. Dashboard actual tool calls, command output, task state aur token usage dikhata hai.

Yeh **naya physical computer, remote desktop, virtual OS, ya eight locally-loaded LLMs nahi** hai. Local process lightweight orchestration karta hai; model inference aapke selected provider par hota hai. NASA-grade certification, “world's most powerful,” all-internet coverage, bug-free output, ya arbitrary projects par 4 GB smoothness ka claim nahi hai.

## Quick start

Python 3.9+ required. Extract the supplied ZIP; keep the `aiagent/py.test` directory intact.

```bash
cd aiagent/py.test
python -m venv .venv
# Linux/macOS/Termux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Configure your own model provider key **locally**, not in chat or in committed source. Existing FullAgent model/provider selection is reused. For example, if you use an OpenCode provider:

```bash
export OPENCODE_API_KEY="YOUR_OWN_KEY"
# PowerShell: $env:OPENCODE_API_KEY="YOUR_OWN_KEY"
python main.py
```

Use `/model` to select a model your account can access, with tool-calling support. The pre-existing model catalog is not a guarantee that every listed model is currently available. Custom OpenAI-compatible endpoints remain supported through FullAgent's existing `models.json` configuration.

Inside the terminal:

```text
/on
```

This uses the current working directory. **Better for a new project:** create an empty project directory first, then use:

```text
/on /absolute/path/to/my-project
```

Now type your actual goal. Example only:

```text
Build a small Python CLI for tracking expenses. Include CSV import, tests,
error handling, packaging, and setup instructions. Keep dependencies minimal.
```

You do not need to manually create eight agents. The dashboard initially shows eight idle slots; real model calls begin only after you submit a goal. The requested Devin-like CLI was an example of a future goal, not a hard-coded project in this release.

## The eight specialists

| Agent | Name | Focus |
|---|---|---|
| a1 | Atlas | Architecture, interfaces, integration and acceptance criteria |
| a2 | Scout | Research, primary sources, alternatives and licenses |
| a3 | Forge | Core implementation |
| a4 | Link | APIs, adapters and interoperability |
| a5 | Probe | Tests, regression coverage and edge cases |
| a6 | Shield | Security, permissions, secrets and dependency risk |
| a7 | Pulse | Performance, memory bounds and observability |
| a8 | Guide | Documentation, packaging and operation |

These are separate contexts/tool loops, not eight different newly trained models. By default all use the selected model. Assign different existing models to specialists when useful:

```text
/computer model a1 MODEL_ID_FROM_YOUR_MODEL_LIST
/computer model a5 MODEL_ID_FROM_YOUR_MODEL_LIST
/computer models
```

Model settings are frozen per mission. Changing `/model` during a run affects the next mission, not workers already in progress.

## Real execution pipeline

1. **Parallel research and inspection.** Eight sessions examine the goal and project from their specialties. Missing network sources are reported, never replaced with fabricated results.
2. **Shared refinement rounds.** All eight see the previous round's peer reports and improve their proposals. Each round has a barrier: agents cannot read a peer's future result. The default is two rounds total (initial research plus one shared refinement).
3. **Validated plan.** Atlas synthesizes the reports into exactly eight tasks, explicit file ownership, a dependency DAG and concrete acceptance checks. Cycles, overlapping ownership, secret paths and malformed plans are rejected. Up to three planning attempts are allowed.
4. **Human plan approval.** The complete proposed scopes and commands are shown before implementation. Denying the plan preserves research and a resumable checkpoint without starting implementation.
5. **Concurrent implementation.** Independent tasks run in parallel, up to the configured cap. Dependent tasks wait for prerequisites. Agents share findings/blockers through the board and peer notes. Writes are short atomic, serialized operations with hash-based stale-write protection; model generation and independent work remain parallel.
6. **Actual verification.** Approved acceptance commands execute on the host and return real exit codes/output. File checks inspect actual files. A failed or denied command is a failed check.
7. **Eight-way peer review.** All specialists inspect the outcome read-only and report actionable issues. A passing test suite does not erase a review blocker.
8. **Bounded repair.** Failed checks and peer issues are fed back to the eight workstreams. Checks and reviews run again. The system stops on success, user cancellation, missing capability, an exhausted budget or the repair limit—not on an infinite “keep thinking” loop.

`completed` means the approved checks passed, all tasks supplied valid done reports and all eight reviews passed. It is not proof of universal correctness. Review reports and meaningful acceptance tests matter; weak checks can miss defects.

The existing legacy Crew remains serial. `/on` has a separate bounded parallel scheduler; older workflows are not silently changed to concurrent execution.

## Live terminal controls

Recommended terminal: at least **80 columns × 24 rows**. The compact dashboard also truncates safely on smaller terminals. It redraws at a bounded rate instead of printing simulated activity.

```text
/computer help
/computer status
/computer pause
/computer resume
/computer cancel
/computer list
/computer resume MISSION_ID
/computer report
/off
```

- **Ctrl+C** cancels a running computer mission, including at its approval prompt.
- **Pause** stops scheduling new model/tool actions. An already-started request or command may finish. It is not an OS process freeze.
- **Cancel** stops new side effects, signals active workers and terminates command process groups. Model HTTP calls drain at their bounded read/deadline boundary; cancelling does not instantly erase an already-billed request.
- **`/off`** requests cancellation if needed and restores normal chat after workers stop.
- Mutating legacy slash commands are blocked while a computer mission runs, so the main chat does not concurrently rewrite the same workspace.

The screen shows real reported API tokens separately from estimated/unknown usage, per-agent activity/tool counts, task states, check results, source counts and recent events. It never displays private model chain-of-thought.

## Approvals and safety boundaries

At a computer-mode approval prompt:

- `y`: approve this action.
- `n` or Enter: deny.
- `a` at a **command** prompt: grant that exact argument list and working directory for the current mission, **including reruns after code changes**. Inspect the command/project before granting this.
- `a` at a **plan** prompt: approve this plan once.

Computer-mode approval is deliberately independent of global `/approve` and legacy autonomy. It never enables unrestricted auto-approval behind your back.

File tools:

- stay under the selected workspace;
- reject absolute/traversal paths, symlinks, hardlinks, common credentials and dependency caches;
- limit file size, scan size and returned text;
- require the latest SHA-256 for existing-file writes;
- restrict writes to each agent's human-approved file scopes;
- save content-addressed pre-write backups and intent/completion events;
- do not provide delete/move tools.

Command tools:

- use an explicit argument list with `shell=False`;
- require a separate grant, run one at a time and serialize against computer-mode file writes;
- enforce a wall timeout, capture bounded stdout/stderr and stop process groups;
- remove API keys and most ambient environment variables; use a separate command home;
- set conservative thread/Node-heap hints for common runtimes.

**Important: these commands are NOT sandboxed at the OS level.** A Python/Node script, build tool, test suite or executable can still read/write outside the workspace, access the network, or spawn resource-heavy processes. Removing environment variables and using `shell=False` does not make arbitrary code safe. Use a disposable container/VM and a clean Git working tree for untrusted projects. Resource hints are not universal memory enforcement. Subprocess file changes are not automatically backed up or rolled back.

Do not deploy/publish or use production credentials without a separate human-reviewed workflow. File allowlists and URL checks are defence in depth, not a substitute for container/network isolation.

## Public research adapters

Available sources:

- DuckDuckGo web search (HTML; may be blocked or change markup).
- GitHub repository search API (public, unauthenticated; rate limits apply).
- GitLab public project search API.
- npm registry package search.
- Wikipedia search API.
- **Optional Google Programmable Search**, with your own configuration:

```bash
export GOOGLE_CSE_API_KEY="YOUR_OWN_KEY"
export GOOGLE_CSE_ID="YOUR_SEARCH_ENGINE_ID"
```

Without these Google settings, the agent sees a clear “Google search was not performed” error. It can still use the other available sources. These adapters do not search private repositories, bypass logins/paywalls, crawl every page, automatically install packages, or guarantee complete coverage.

Results retain source names, URLs and retrieval timestamps. Fetched pages also have a content hash. Text/JSON responses are bounded, cached for ten minutes, and clearly treated as **untrusted source data**. Local/private/metadata addresses and nonstandard ports are blocked for research; redirects are rechecked. This is application-level filtering, not a hardened network sandbox or DNS-rebinding-proof security boundary.

Selected project source and reports are sent to your configured model provider. Research queries go to the selected public sources. Keep confidential data and secrets out of the workspace and queries.

## 4 GB systems and budgets

The orchestration process does not load model weights, Docker daemons, browsers, vector-database servers or eight heavyweight copies of the main agent. It uses bounded contexts, a thread pool and existing dependencies. The default supports eight overlapping model/API requests but only one host command at a time.

This is designed to be lightweight, **not a benchmark or guarantee on every 4 GB machine**. Large builds, local models, games, datasets or scientific workloads can exceed available RAM regardless of agent scheduling. API latency and rate limits also affect responsiveness.

While idle, tune the next run:

```text
/computer set max_parallel 8
/computer set token_budget 400000
/computer set wall_minutes 60
/computer set plan_rounds 2
/computer set repair_rounds 2
/computer set work_steps 24
```

For slower services or especially small machines:

```text
/computer set max_parallel 4
/computer set max_output_tokens 2048
/computer set max_context_chars 32000
```

Eight logical specialists remain, but at most four run at once in that configuration. Smaller budgets can pause before completion; that is intentional rather than silent overspending.

```text
/computer set plan_only true
/computer set network false
```

- `plan_only` performs research/planning but no implementation.
- `network false` disables **public research**, not the model API. This is not an offline LLM mode.
- `wall_minutes` includes paused time and time waiting for approval.
- Token reservations are synchronized across workers. Every HTTP attempt, including retries, gets a reservation. Provider-reported usage replaces it when available; unknown/failed calls conservatively retain estimates. Exact billing cannot be enforced without the provider's tokenizer/invoice. The dashboard is explicit about estimates.

Use `/computer set ...` only while idle, then resume a saved mission to extend limits explicitly. Values are validated; unlimited workers, time and context sizes are not accepted.

## Checkpoints and recovery

State lives under:

```text
$FULLAGENT_HOME/computer/<mission-id>/
# default FullAgent home is usually ~/.fullagent
```

Files include:

- `state.json`: atomically written checkpoint and shared task board.
- `events.jsonl`: actual actions/events; one bounded rotated segment is retained.
- `report.md`: final/provisional outcome, plan, checks, peer issues and source provenance.
- `command-*.log`: bounded real command output.
- `backups/<sha256>`: pre-write contents for file tools; `file.intent` and `file.written` events map hashes to paths.

```text
/computer list
/computer resume <mission-id>
```

Resume requires the original workspace root and another plan approval. Successful planning reports are reused. Interrupted tasks re-inspect files before acting; completed owned files are fingerprinted (bounded scans) and changed scopes are re-evaluated. A lost in-flight API reservation is conservatively counted as unknown rather than silently refunded. An OS-level cooperative file lock prevents two computer-mode processes from working on the same root.

Recovery cannot guarantee exactly-once semantics for an arbitrary external command that was interrupted after making a side effect. Inspect the logs and actual state before approving a rerun. Backups are manual recovery evidence, not a complete transaction across processes or external services.

## Headless usage

This path does not need to import the interactive UI:

```bash
python main.py computer doctor
python main.py computer --help
python main.py computer run --root /path/to/project --goal "YOUR REAL GOAL" --plan-only
python main.py computer list --root /path/to/project
python main.py computer status MISSION_ID
python main.py computer report MISSION_ID
```

For automation, noninteractive approval defaults to **deny**. Explicit grants:

```bash
python main.py computer run \
  --root /path/to/project \
  --goal "Implement the requested feature with tests" \
  --approve-plan \
  --allow-command '["python", "-m", "unittest", "discover", "-s", "tests"]'
```

`--approve-plan` permits the generated owned-file plan; it is a real write authorization. `--allow-command` grants only that exact argv in the workspace root, including subsequent reruns during this mission. No catch-all unrestricted execution flag is provided.

## Credentials changed in this update

Nine long hard-coded provider-key fallback values were removed from `fullagent/config.py`. Existing environment-variable names and model configuration remain usable. Configure your own keys locally. If any embedded keys belonged to you or were ever committed/shared, **revoke/rotate them**; removing them from this ZIP does not revoke them or remove old Git history.

No real key values are included in this guide. No existing private key was used to validate remote model capabilities.

## Testing and verification

```bash
python -m compileall -q fullagent tests
python -m unittest discover -s tests -v
python run_selftests.py
```

The new tests use clearly labelled deterministic model fixtures and a local HTTP test server. They exercise actual scheduling, eight-thread overlap, file writes/backups, command execution, failures, repair, cancellation, checkpoints, HTTP JSON/SSE parsing, public-source response shapes, approvals and dashboard layout. They are **not** a fake-agent production mode or evidence that a paid remote model completed a real large project.

See `VERIFICATION.md` for the exact results and limitations observed while preparing this ZIP. External GitHub/PyPI/model connectivity was unavailable in the preparation environment, so live provider/search end-to-end validation is not claimed.
