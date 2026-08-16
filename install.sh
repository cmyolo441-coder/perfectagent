#!/usr/bin/env bash
set -euo pipefail

REPO="cmyolo441-coder/perfectagent"
VERSION="v2.2"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
NAME="fullagent"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)  OS_TAG="linux-x64" ;;
  Darwin) OS_TAG="darwin" ;;
  *) echo "Unsupported OS: $OS (please provide your own build or build from source)" >&2; exit 1 ;;
esac

case "$ARCH" in
  x86_64|amd64) : ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

ASSET="${NAME}-${OS_TAG}"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

echo ">> Downloading ${NAME} ${VERSION} for ${OS_TAG}..."
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "$TMP/$NAME" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -qO "$TMP/$NAME" "$URL"
else
  echo "curl or wget required" >&2; exit 1
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
