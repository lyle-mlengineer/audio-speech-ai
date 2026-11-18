"""``python -m sheng_chatterbox`` -> the same CLI as the ``sheng-chatterbox`` script."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
