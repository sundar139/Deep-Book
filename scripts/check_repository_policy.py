"""Run DeepBook repository policy checks."""

from __future__ import annotations

import sys

from deepbook.repository_policy import main

if __name__ == "__main__":
    sys.exit(main())
