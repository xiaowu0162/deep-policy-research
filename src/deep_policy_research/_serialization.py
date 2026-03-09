from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def drop_nones(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {key: drop_nones(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_nones(item) for item in value]
    return value
