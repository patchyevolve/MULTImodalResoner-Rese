"""`python -m mlforge` — same entry point as the installed console script.

The CLI is the single interface into the Workflow API (13 §1); this
module just makes the package executable for environments where the
scripts directory is not on PATH (venvs, containers, CI).
"""

from __future__ import annotations

import sys

from mlforge.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
