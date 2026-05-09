from __future__ import annotations

import re
import time
from dataclasses import dataclass


def now_monotonic() -> float:
    return time.monotonic()


def clean_ocr_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text).strip()
    text = re.sub(r"[^\w\s,.;:!?áéíóúÁÉÍÓÚñÑüÜ-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(slots=True)
class CooldownGate:
    cooldown_seconds: float
    _last_emit_at: float = 0.0

    def ready(self) -> bool:
        return (now_monotonic() - self._last_emit_at) >= self.cooldown_seconds

    def mark(self) -> None:
        self._last_emit_at = now_monotonic()
