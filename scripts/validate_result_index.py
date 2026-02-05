from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "notes" / "result_index.md"


def main() -> int:
    text = INDEX.read_text(encoding="utf-8")
    paths = re.findall(r"`(docs/[^`]+|experiments/[^`]+)`", text)
    missing = []
    checked = 0
    for rel in paths:
        # skip results paths explicitly
        if rel.startswith("results/"):
            continue
        if "<" in rel or ">" in rel:
            continue
        path = ROOT / rel
        checked += 1
        if not path.exists():
            missing.append(rel)

    print(f"checked: {checked}")
    if missing:
        print("missing:")
        for p in missing:
            print(f"- {p}")
        return 1
    print("missing: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
