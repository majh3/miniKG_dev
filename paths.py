"""Repository-local paths used by the standalone MiniKG core."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_UNIFY_ALL = Path(__file__).resolve().parent / "src_unify_all"
if str(SRC_UNIFY_ALL) not in sys.path:
    sys.path.append(str(SRC_UNIFY_ALL))

