"""Deterministic TEST fixtures, never a production/offline fake-agent mode."""
from __future__ import annotations
import json
import re
import sys
import threading
import time
from types import SimpleNamespace

from fullagent.client import StreamResult
from fullagent.computer.state import AGENT_IDS


def plan(dependencies=False):
    tasks = [{"id": i, "title": f"Deliver {i}", "instructions": f"Implement {i}.py and check the assigned requirement",
              "files": [f"{i}.py"], "depends_on": []} for i in AGENT_IDS]
    if dependencies:
        tasks[1]["depends_on"] = ["a1"]
    return {"summary": "A small eight-part test fixture, not a real generated project",
            "tasks": tasks,
            "checks": [{"kind": "command", "description": "Verify all eight real output files",
                        "argv": [sys.executable, "-c", "from pathlib import Path; assert all('FIXED' in Path(f'a{i}.py').read_text() for i in range(1,9))"], "cwd": "."},
                       {"kind": "file_contains", "description": "Architecture file has a value", "path": "a1.py", "text": "value ="}]}


def tool(name, args):
    return StreamResult(tool_calls=[{"id": "call_fixture", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}],
                        finish_reason="tool_calls", usage={"prompt_tokens": 60, "completion_tokens": 20})


def answer(text):
    return StreamResult(content=text, finish_reason="stop", usage={"prompt_tokens": 60, "completion_tokens": 20})


class Driver:
    def __init__(self, *, bug=False, barrier=False, failure=None, deps=False, review_failure=False, malformed_plan=False):
        self.bug, self.failure, self.deps = bug, failure, deps
        self.review_failure = review_failure
        self.malformed_plan = malformed_plan
        self.barrier = threading.Barrier(8) if barrier else None
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = []
        self.plans = 0
        self.reached = threading.Event()
        self.release = None

    def factory(self, agent_id):
        driver = self
        class Client:
            model = SimpleNamespace(id="scripted-test-fixture")
            def chat(self, messages, schemas, max_tokens, control, on_update=None):
                control()
                with driver.lock:
                    driver.active += 1
                    driver.peak = max(driver.peak, driver.active)
                try:
                    seed = messages[1]["content"]
                    phase_match = re.search(r"PHASE: ([^\n]+)", seed)
                    phase = phase_match.group(1) if phase_match else "planning"
                    tools = [m for m in messages if m["role"] == "tool"]
                    with driver.lock:
                        driver.calls.append((agent_id, phase, len(tools)))
                    driver.reached.set()
                    if driver.release is not None:
                        while not driver.release.wait(0.05):
                            control()
                    if driver.failure == (agent_id, phase):
                        raise RuntimeError("Intentional test provider failure")
                    if phase == "planning":
                        driver.plans += 1
                        p = plan(driver.deps)
                        if driver.malformed_plan and driver.plans == 1:
                            p["tasks"][1]["files"] = ["a1.py"]
                        return answer(json.dumps(p))
                    if phase == "research" or phase.startswith("refine-"):
                        if driver.barrier and phase == "research" and not tools:
                            driver.barrier.wait(timeout=5)
                        time.sleep(0.015)
                        return answer(f"Fixture report for {agent_id}: inspect code and assign one owned module. No external research is claimed by this fixture.")
                    if phase.startswith("review-"):
                        if not tools:
                            return tool("read_file", {"path": agent_id+".py"})
                        issue = driver.review_failure
                        return answer(json.dumps({"status": "changes_requested" if issue else "pass",
                                                  "summary": "Fixture review inspected its own file", "issues": ["Intentional review blocker"] if issue else []}))
                    if phase in ("building", "repairing"):
                        if not tools:
                            return tool("read_file", {"path": agent_id+".py"})
                        if len(tools) == 1:
                            read = json.loads(tools[-1]["content"])
                            expected = read.get("sha256", "MISSING")
                            value = "BUG" if driver.bug and agent_id == "a3" and phase == "building" else "FIXED"
                            return tool("write_file", {"path": agent_id+".py", "expected_sha256": expected,
                                                       "content": f"value = '{value}'\n"})
                        result = json.loads(tools[-1]["content"])
                        return answer(json.dumps({"status": "done" if result.get("ok") else "blocked",
                                                  "summary": "Fixture wrote the owned module", "issues": [] if result.get("ok") else [str(result)]}))
                    raise AssertionError("Unexpected phase " + phase)
                finally:
                    with driver.lock:
                        driver.active -= 1
            def close(self):
                pass
        return Client()
