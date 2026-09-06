# Verification record — FullAgent Computer Mode 3.2.0

Prepared from the user's uploaded `py.test-main.zip` on 2026-09-06. This is an updated source bundle, not a Git clone with history or a published GitHub release.

## Executed checks

Environment: Linux sandbox, CPython **3.13.14**. The source package still declares Python 3.9+, but other Python versions and operating systems were not executed here.

| Check | Observed result |
|---|---|
| Compile all application and test sources | Passed |
| New regression suite | **77 discovered; 76 passed, 1 skipped; no failures** |
| Existing module self-test suite | **55 passed** before changes and 55 passed after integration |
| Real simultaneous scheduler overlap | Eight test-client calls overlapped; a configured cap of two was also enforced |
| End-to-end deterministic project fixture | Eight real files written; actual subprocess/file checks passed |
| Failure → repair → verification fixture | A real failing check triggered a repair round, a backup was retained, and checks reran successfully |
| Safety/recovery | Approval denial, stale-write races, scope/path guards, cancellation, command timeout, checkpoint resume and workspace locking tested |
| HTTP transport | Real localhost HTTP JSON/SSE requests tested; partial streams, rate limits, unsupported streaming options and response bounds covered |
| Research adapters | Six public-source response shapes, caching, missing Google configuration, explicit errors, provenance and public-address validation tested with fixtures |
| UI logic | Dashboard rendering, width bounds, bridge commands, actual approval-handler isolation, command routing hooks and CLI entrypoints tested |
| Dashboard visual inspection | Ready (120 columns), running (80 columns), completed (120 columns) renderer images inspected; no overlapping/overflowing elements observed |
| Credential-pattern scan | No matches for the scanned common live-key patterns in deliverable source/docs; nine long embedded provider-key defaults removed |

Raw output is included in `verification/computer-tests.txt` and `verification/legacy-selftests.txt`.

## What was skipped / not verified

- The **actual prompt_toolkit UI construction test was skipped** because `rich` and `prompt_toolkit` were unavailable in this sandbox. Installing them failed because external DNS/network access was unavailable. The existing project already depends on these packages; install the normal requirements on your machine before running the interactive UI.
- The pure dashboard renderer, bridge logic and approval handler were tested, but this is **not a claim that the entire interactive terminal application was exercised end-to-end in a real TTY**. The test remains in the suite and runs when dependencies are present; the build workflow installs them.
- Live remote model-provider calls and live public search services were **not** end-to-end tested. No remote-model success, free-tier availability or account access is claimed.
- The included screenshots are clearly labelled **LOCAL TEST FIXTURE / NO LIVE MODEL/API CLAIM**. They show actual scheduler/dashboard state from a deterministic local fixture. Their usage numbers are fixture-supplied test data, not billed cloud tokens.
- No Windows, macOS, Termux, actual 4 GB hardware, scientific/safety-critical workload, wheel or standalone executable build was performed here. Setuptools/wheel were also unavailable; the deliverable is source code, not a compiled binary.
- No GitHub push, release, tag, credential revocation, or remote deployment was performed.

## Lightweight orchestration observation

The separate dashboard/scheduler fixture recorded **44.33 MiB peak process RSS** on this Linux environment using `resource.getrusage(RUSAGE_SELF)`. It included Python, Pillow rendering and test-fixture overhead, not local model weights or a large project build. Eight fixture requests overlapped. Raw measurement and scope are in `verification/runtime-fixture.json`.

This is an observation about one small test process, **not** a 4 GB smoothness guarantee or a production-model benchmark. Arbitrary commands, dependencies, browsers, local LLMs and large builds can require far more RAM.

## Remaining operational boundaries

- Remote model quality, availability, pricing, context limits and tool-calling behaviour vary by provider. Use a model/account you can access.
- “Completed” is scoped to the human-approved acceptance checks and the eight peer reports. It is not universal bug-freedom, completeness of all research, or enterprise/scientific certification.
- File tools have workspace guards, but approved subprocesses run on the host. They are not contained by a VM/container and can modify files beyond owned scopes. Use isolation for untrusted projects.
- Hash checks/locks coordinate computer-mode workers, not unrelated editors/processes. Public URL validation is application-level filtering, not a hardened DNS/network sandbox.
- Token reservations use estimates until actual usage is reported. Missing usage and interrupted calls are conservatively charged as estimates. The provider invoice remains authoritative.
- File-tool backups do not cover arbitrary subprocess or external-service side effects. Exactly-once replay cannot be guaranteed for interrupted commands.
- Common credential patterns were scanned, but pattern scanning is not proof that every possible secret format was found. Rotate any keys previously committed or shared.

## Reproduce locally

After installing the package dependencies:

```bash
python -m compileall -q fullagent tests
python -m unittest discover -s tests -v
python run_selftests.py
python main.py computer doctor
python main.py
```

Then select an accessible tool-capable model, use `/on` on a disposable test project, and approve only commands you have reviewed. See `COMPUTER_MODE.md` for setup, budgets and recovery.
