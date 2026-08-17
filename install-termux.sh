#!/usr/bin/env bash
# FullAgent installer for TERMUX (Android) — installs Python from apt
# and then fullagent from source (Termux cannot run native binaries).
# Also works on any system where `python` + `pip` are available.
set -euo pipefail

REPO="cmyolo441-coder/perfectagent"

echo ">> Step 1/4: install python + git (if missing)"
if command -v pkg >/dev/null 2>&1; then
  pkg install -y python git || true
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update -y >/dev/null 2>&1 || true
  apt-get install -y python3 python3-pip git || true
fi

PY="python3"; command -v python >/dev/null 2>&1 && PY="python"

echo ">> Step 2/4: upgrade pip"
$PY -m pip install --upgrade pip >/dev/null 2>&1 || true

echo ">> Step 3/4: install fullagent"
$PY -m pip install --upgrade "git+https://github.com/${REPO}.git"

echo ">> Step 4/4: make sure the launcher is on PATH"
if ! command -v fullagent >/dev/null 2>&1; then
  grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
  export PATH="$HOME/.local/bin:$PATH"
fi

echo ""
echo ">> Installed. Run: fullagent"
fullagent --version 2>/dev/null || $PY -m fullagent --version 2>/dev/null || true
