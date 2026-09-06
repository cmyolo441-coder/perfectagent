"""Run every fullagent module's __main__ self-test as a subprocess."""
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
mods = sorted(p.stem for p in (PKG / "fullagent").glob("*.py"))
skip = {"__init__", "__main__",
        "agent", "client", "config", "tui"}  # no self-test block

failed = []
for m in mods:
    if m in skip:
        continue
    r = subprocess.run(
        [sys.executable, "-m", f"fullagent.{m}"],
        capture_output=True, text=True, cwd=str(PKG), timeout=300)
    out = (r.stdout + r.stderr).strip().splitlines()
    tail = out[-1] if out else "(no output)"
    marker = r.stdout + r.stderr
    ok = (r.returncode == 0
          and ("PASS" in marker or "passed" in marker.lower()))
    status = "PASS" if ok else "FAIL"
    print(f"{status:>4}  {m:<14} {tail[:110]}")
    if not ok:
        failed.append(m)

print()
if failed:
    print(f"{len(failed)} FAILED: {', '.join(failed)}")
    sys.exit(1)
print(f"ALL {len(mods) - len(skip)} MODULE SELF-TESTS PASS")
