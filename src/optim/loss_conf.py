"""Build the legacy `loss_conf` dict from a YAML `losses.{mesh,gs}` section.

YAML structure (per stage group):

    losses:
      mesh:
        common:    {key: value, …}    # applied to both levels
        level_0:   {key: value, …}    # overrides common at level 0
        level_1:   {key: value, …}    # overrides common at level 1

Output dict shape (matches the legacy AvatarOptimizer interface):

    {
      'level_0': {**common, **level_0},
      'level_1': {**common, **level_1},
    }
"""
from __future__ import annotations

from typing import Any, Dict


def build_loss_conf(section: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Merge `common` with each per-level override.

    Values in `level_K` replace any same-named key from `common` at that
    level (no recursion — full replacement, matching `{**common, **level_K}`).
    """
    common = dict(section.get('common', {}))
    out: Dict[str, Dict[str, Any]] = {}
    for level_key in ('level_0', 'level_1'):
        out[level_key] = {**common, **dict(section.get(level_key, {}))}
    return out
