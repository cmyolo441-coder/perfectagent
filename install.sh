#!/usr/bin/env bash
set -euo pipefail

REPO="cmyolo441-coder/perfectagent"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
NAME="fullagent"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64|amd64) OS_TAG="linux-x64" ;;
      *) OS_TAG="" ;;
    esac ;;
  Darwin)
    case "$ARCH" in
      arm64)  OS_TAG="darwin-arm64" ;;
      x86_64) OS_TAG="darwin-arm64" ;;  # Rosetta 2 runs arm64 binaries
      *)      OS_TAG="" ;;
    esac ;;
  *) OS_TAG="" ;;
esac

# ---------------------------------------------------------------------------
# native binary available? download from the LATEST release
# ---------------------------------------------------------------------------
if [ -n "$OS_TAG" ]; then
  ASSET="${NAME}-${OS_TAG}"
  echo ">> Looking up the latest release of ${REPO} ..."
  if command -v curl >/dev/null 2>&1; then
    URL="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
         | grep "browser_download_url.*${ASSET}\"" | cut -d '"' -f 4 | head -1 || true)"
  elif command -v wget >/dev/null 2>&1; then
    URL="$(wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
         | grep "browser_download_url.*${ASSET}\"" | cut -d '"' -f 4 | head -1 || true)"
  else
    echo "curl or wget required" >&2; exit 1
  fi

  if [ -n "${URL:-}" ]; then
    echo ">> Downloading ${ASSET} ..."
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "$TMP/$NAME" "$URL"
    else
      wget -qO "$TMP/$NAME" "$URL"
    fi
    chmod +x "$TMP/$NAME"

    if [ -w "$INSTALL_DIR" ] || [ "$(id -u)" = "0" ]; then
      mv "$TMP/$NAME" "$INSTALL_DIR/$NAME"
    else
      echo ">> Installing to $INSTALL_DIR (sudo required)"
      sudo mv "$TMP/$NAME" "$INSTALL_DIR/$NAME"
    fi

    echo ">> Installed: $INSTALL_DIR/$NAME"
    echo ">> Run: fullagent"
    "$INSTALL_DIR/$NAME" --version 2>/dev/null || true
    exit 0
  fi
  echo "!! No ${ASSET} asset in the latest release yet — falling back to pip"
fi

# ---------------------------------------------------------------------------
# fallback: pure-Python install (works on EVERY platform with Python 3.9+,
# e.g. Linux arm64, Windows, or when binaries are not yet published)
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to install from source" >&2; exit 1
fi
python3 -m pip install --upgrade "git+https://github.com/${REPO}.git"
echo ">> Installed via pip. Run: fullagent"
fullagent --version 2>/dev/null || true
