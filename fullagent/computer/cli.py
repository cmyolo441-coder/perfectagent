"""Headless computer mode; no terminal-UI dependency is imported here."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .. import config
from ..config import Config, PROVIDERS, effort_by_key, model_by_id
from .engine import Computer
from .state import Settings, list_missions, load_checkpoint, mission_dir, safe_text
from .transport import APIClient
from .view import status_text


def parser():
    p = argparse.ArgumentParser(prog="fullagent computer", description="Eight-agent API-backed workspace computer (not an OS sandbox)")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="start or resume a bounded mission")
    run.add_argument("--root", default=".")
    run.add_argument("--goal")
    run.add_argument("--resume", metavar="MISSION_ID")
    run.add_argument("--model")
    run.add_argument("--parallel", type=int)
    run.add_argument("--tokens", type=int)
    run.add_argument("--minutes", type=int)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument("--no-research", action="store_true")
    run.add_argument("--approve-plan", action="store_true", help="explicitly approve generated owned-file writes (host commands still require grants)")
    run.add_argument("--allow-command", action="append", default=[], metavar="ARGV_JSON",
                     help="grant one exact argv in the workspace root for this mission, including reruns; may be repeated")
    for command in ("status", "report"):
        item = sub.add_parser(command)
        item.add_argument("mission_id")
    ls = sub.add_parser("list")
    ls.add_argument("--root")
    sub.add_parser("doctor", help="local checks only; no provider/network calls")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = Config.load()
    store = config.APP_DIR / "computer"
    try:
        if args.command == "doctor":
            m = model_by_id(cfg.model_id)
            provider = PROVIDERS.get(m.provider) if m else None
            print("Python:", sys.version.split()[0])
            print("UI dependencies:", ", ".join(f"{name}={'installed' if importlib.util.find_spec(name) else 'missing'}" for name in ("rich", "prompt_toolkit")))
            print("Selected model:", m.label if m else "unknown")
            print("Provider key:", "configured" if provider and provider.api_key else "not configured")
            print("Settings:", json.dumps(asdict(Settings.from_dict(cfg.extra.get("computer_settings", {})))))
            print("Eight logical sessions, at most eight I/O workers; one command at a time.")
            print("No model/network benchmark was performed. Host commands are not sandboxed.")
            return 0
        if args.command == "list":
            for row in list_missions(store, Path(args.root) if args.root else None):
                print(json.dumps(row, ensure_ascii=False))
            return 0
        if args.command in ("status", "report"):
            data = load_checkpoint(store, args.mission_id)
            if args.command == "status":
                print(status_text(data))
            else:
                path = mission_dir(store, args.mission_id) / "report.md"
                if not path.exists():
                    raise ValueError("No finalized report yet; inspect status/state.json")
                print(path.read_text(encoding="utf-8"))
            return 0
        if bool(args.goal) == bool(args.resume):
            raise ValueError("Supply exactly one of --goal or --resume")
        settings = asdict(Settings.from_dict(cfg.extra.get("computer_settings", {})))
        for key, value in (("max_parallel", args.parallel), ("token_budget", args.tokens), ("wall_minutes", args.minutes)):
            if value is not None:
                settings[key] = value
        if args.plan_only:
            settings["plan_only"] = True
        if args.no_research:
            settings["network"] = False  # model API remains enabled
        settings = Settings.from_dict(settings)
        selected = args.model or cfg.model_id
        overrides = dict(cfg.extra.get("computer_models", {}))
        effort = effort_by_key(cfg.effort)
        def factory(agent_id):
            model = model_by_id(selected if args.model else overrides.get(agent_id, selected))
            if model is None:
                raise ValueError("Unknown model; configure an existing tool-capable model first")
            return APIClient(PROVIDERS[model.provider], model, effort, settings.request_timeout)
        grants = set()
        for raw in args.allow_command:
            value = json.loads(raw)
            from .tools import WorkspaceTools
            WorkspaceTools.validate_argv(value)
            grants.add(json.dumps(value))
        root = Path(args.root).expanduser().resolve()
        def approve(kind, details):
            if kind == "plan" and args.approve_plan:
                return True
            if kind == "command" and details["cwd"] == str(root) and json.dumps(details["argv"]) in grants:
                return "always"
            if not sys.stdin.isatty():
                return False
            print("\nAPPROVAL REQUIRED (host execution is NOT sandboxed):")
            print(safe_text(json.dumps(details, indent=2, ensure_ascii=False), 60000))
            # Bounded, cancellation-aware read: input() on a worker thread
            # could otherwise keep an executor alive after Ctrl+C.
            import queue, threading
            answer = queue.Queue(maxsize=1)
            def read_answer():
                try:
                    answer.put(input("Approve this action? [y/N] "))
                except EOFError:
                    answer.put("")
            threading.Thread(target=read_answer, daemon=True).start()
            while True:
                computer._control()
                try:
                    return answer.get(timeout=0.2).strip().lower() == "y"
                except queue.Empty:
                    continue
        def event(value):
            print(json.dumps(value, ensure_ascii=False), flush=True)
        computer = Computer(root, store, settings, factory, approve, event)
        computer.start(args.goal or "", args.resume)
        try:
            while not computer.join(0.2):
                pass
        except KeyboardInterrupt:
            computer.cancel()
            computer.join()
        data = computer.snapshot()
        print(f"Result: {data['status']}\nReport: {computer.board.path / 'report.md'}")
        if data["status"] == "completed" or (data["status"] == "planned" and settings.plan_only):
            return 0
        if data["status"] == "cancelled":
            return 130
        if data["status"] == "budget_exhausted":
            return 3
        return 2
    except (ValueError, OSError, RuntimeError) as exc:
        print("Computer error: " + safe_text(exc, 1000), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
