from __future__ import annotations

import sys

from test_ocr_capture import main


def _ensure_raw_ocr_mode() -> None:
    if "--no-expected-fallback" not in sys.argv:
        sys.argv.append("--no-expected-fallback")


if __name__ == "__main__":
    _ensure_raw_ocr_mode()
    raise SystemExit(main())
