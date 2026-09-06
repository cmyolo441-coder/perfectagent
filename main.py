#!/usr/bin/env python3
"""FullAgent launcher — just run: python main.py"""

import os
import sys

# Default fallbacks so API keys work even without env setup.
# Prefer real env var if already set (setx / $env:).
os.environ.setdefault(
    "AGNES_API_KEY",
    "sk-fKLLAlhfkYdwCMrznXi1rKlh3ZQXgNtucHrpPatC7MQCHYVi",
)
os.environ.setdefault(
    "OPENCODE_API_KEY",
    "sk-h11yU0O2sQxGL9CC0Y5bHQxtdWQSqXAi1mRUG7TSLpA7EvFAzYBpyAJ7NQ6xhDvm",
)
os.environ.setdefault(
    "XKIRO_API_KEY",
    "sk-xt-866c0efd3fe7bb9eb65e9477102211b017dacd9a10db6747",
)
os.environ.setdefault(
    "KIOSAPI_API_KEY",
    "sk-cFXQ576lsIctpudkYD5lPniF5UgHGLy1nKeXDscCEvK1LMZV",
)
os.environ.setdefault(
    "BAI_API_KEY",
    "sk-hm51wsk2klugg9v95lwvm9prtsmtekd2",
)
os.environ.setdefault(
    "ZENMUX_API_KEY",
    "sk-ai-v1-9424a61af5fea4355a34de00530e189d1972da4d4f8324815be47b8d5a6280eb",
)

from fullagent.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
