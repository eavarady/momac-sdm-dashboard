from typing import Optional


def safe_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default
