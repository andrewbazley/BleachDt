"""
Utilities for extracting experimental condition labels from filenames/folders.

NOTE: Condition grouping is DISABLED by default. Each movie is treated as its own
entity without attempting to group by experimental condition. To enable condition
grouping, set ENABLE_CONDITION_GROUPING = True.

This project has historically used naming schemes like:
  - WT_ADP_001.tif
  - WT_ATP_002.tif
  - V461F_ADP_101.tif

Newer experiments may encode conditions differently, e.g.:
  - 0.1mMCa_001.tif

When condition grouping is ENABLED, we derive a *condition key* consistently using 
lightweight heuristics:

- Remove path + extension
- Strip a trailing "_results" suffix (folder naming convention)
- Split on "_" into tokens
- If the final token looks like a replicate id (digits, "rep<digits>", "r<digits>"),
  drop it
- The remaining tokens joined by "_" are the condition key

When condition grouping is DISABLED (default), the full movie name is returned
without attempting to parse or group conditions.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

# DISABLED BY DEFAULT - Set to True to enable automatic condition grouping
ENABLE_CONDITION_GROUPING = False

_REP_TOKEN_RE = re.compile(r"^(?:rep|r)\d+$", re.IGNORECASE)


def _tokenize(name: str) -> Tuple[str, list[str]]:
    """Return (base_name_no_ext, non-empty '_' tokens)."""
    base = os.path.basename(str(name)).strip()
    base = os.path.splitext(base)[0]

    # Folder naming convention used throughout this repo
    if base.endswith("_results"):
        base = base[: -len("_results")]

    tokens = [t for t in base.split("_") if t]
    return base, tokens


def extract_condition_key(name: object, enable_grouping: Optional[bool] = None) -> Optional[str]:
    """
    Derive a condition identifier from a filename or folder name.
    
    When condition grouping is DISABLED (default), returns the full movie name
    without attempting to parse or group by experimental condition.
    
    When condition grouping is ENABLED, strips replicate IDs to group movies
    by experimental condition.

    Parameters
    ----------
    name : object
        Filename or folder name to process
    enable_grouping : bool, optional
        Override the global ENABLE_CONDITION_GROUPING setting.
        If None, uses the global setting.

    Examples (with grouping ENABLED)
    ---------------------------------
    - 'WT_ADP_001' -> 'WT_ADP'
    - 'WT_ADP_001_results' -> 'WT_ADP'
    - '/path/to/0.1mMCa_002.tif' -> '0.1mMCa'
    - 'SingleToken' -> 'SingleToken'
    
    Examples (with grouping DISABLED - default)
    --------------------------------------------
    - 'WT_ADP_001' -> 'WT_ADP_001'
    - '0.1mMCa_002.tif' -> '0.1mMCa_002'
    """
    if name is None:
        return None
    
    # Determine whether to use condition grouping
    use_grouping = enable_grouping if enable_grouping is not None else ENABLE_CONDITION_GROUPING

    base, tokens = _tokenize(str(name))
    if not tokens:
        return base or None

    # Handle cases where "_results" was used as a token, not a suffix
    if tokens and tokens[-1].lower() == "results":
        tokens = tokens[:-1]
    if not tokens:
        return base or None

    # If grouping is disabled, return the full name (without _results suffix)
    if not use_grouping:
        return "_".join(tokens) if tokens else (base or None)

    # Grouping enabled: Drop replicate token if present at the end
    last = tokens[-1]
    if last.isdigit() or _REP_TOKEN_RE.match(last):
        tokens = tokens[:-1]
        if not tokens:
            return base or None

    return "_".join(tokens) if tokens else (base or None)


def extract_replicate_id(name: object) -> Optional[str]:
    """
    Extract a replicate identifier from the end of a name, if present.

    Examples
    --------
    - 'WT_ADP_001' -> '001'
    - '0.1mMCa_rep2' -> '2'
    - 'Foo' -> None
    """
    if name is None:
        return None

    _, tokens = _tokenize(str(name))
    if not tokens:
        return None

    last = tokens[-1]
    if last.isdigit():
        return last

    if _REP_TOKEN_RE.match(last):
        digits = "".join(ch for ch in last if ch.isdigit())
        return digits or None

    return None

