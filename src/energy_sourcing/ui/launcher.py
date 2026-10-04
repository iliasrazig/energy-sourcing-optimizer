"""``es-ui`` console script: starts Streamlit on the bundled app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    from streamlit.web import cli  # noqa: PLC0415 - only needed when launching the UI

    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), *sys.argv[1:]]
    return int(cli.main() or 0)


if __name__ == "__main__":
    sys.exit(main())
