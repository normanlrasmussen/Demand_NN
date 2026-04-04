"""
Validate nn_arc/x_columns.txt layout:

- Every 500 columns is one full (store × item) × feature block; the same logical
  store_s_item_j repeats every 500 rows (only the feature suffix after __ changes).
- Within each 500-column block, store_s_item_j sits at index (s - 1) * 50 + (j - 1),
  so e.g. store_1_item_1 is at 0, store_2_item_1 at 50, store_3_item_1 at 100 (0-based),
  and store_1_item_2 is at 1, store_2_item_2 at 51, etc.
"""
from __future__ import annotations

import re
from pathlib import Path

_STORE_ITEM_RE = re.compile(r"^store_(\d+)_item_(\d+)__(.+)$")


def _parse_line(line: str) -> tuple[int, int, str]:
    line = line.strip()
    m = _STORE_ITEM_RE.match(line)
    assert m is not None, f"unexpected column name: {line!r}"
    return int(m.group(1)), int(m.group(2)), m.group(3)


def verify_x_columns(path: Path | str) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    assert n % 500 == 0, f"expected line count multiple of 500, got {n}"

    n_blocks = n // 500

    # Within each 500-column block: index (s-1)*50 + (j-1) -> store_s_item_j
    for b in range(n_blocks):
        base = b * 500
        for s in range(1, 11):
            for j in range(1, 51):
                idx = base + (s - 1) * 50 + (j - 1)
                st, it, _feat = _parse_line(lines[idx])
                assert (st, it) == (
                    s,
                    j,
                ), f"line {idx + 1}: expected store_{s}_item_{j}, got store_{st}_item_{it}"

    # Same slot every 500 lines: prefix store_s_item_j matches across feature blocks
    for k in range(500):
        s0, j0, feat0 = _parse_line(lines[k])
        for m in range(1, n_blocks):
            sm, jm, featm = _parse_line(lines[k + m * 500])
            assert (s0, j0) == (sm, jm), (
                f"column slot {k} (1-based lines {k + 1}, {k + 1 + m * 500}): "
                f"store_{s0}_item_{j0} vs store_{sm}_item_{jm}"
            )
            assert feat0 != featm or m == 0, (
                f"duplicate feature for same slot k={k} block 0 and {m}: {feat0!r}"
            )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    verify_x_columns(root / "x_columns.txt")
    print("x_columns.txt: OK")
