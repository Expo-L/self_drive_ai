"""Parse MovingAI .map files and report geometry/clearance statistics."""

from __future__ import annotations

import statistics
from collections import deque
from pathlib import Path
import sys


def load(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    height = width = None
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("height"):
            height = int(stripped.split()[1])
        elif stripped.startswith("width"):
            width = int(stripped.split()[1])
        elif stripped == "map":
            start = index + 1
            break
    if width is None or height is None or start is None:
        raise ValueError(f"bad map header in {path}")
    rows = lines[start : start + height]
    blocked = set()
    free = set()
    for y, row in enumerate(rows):
        for x, char in enumerate(row[:width]):
            (blocked if char != "." else free).add((x, y))
    return width, height, rows, free, blocked


def clearance(width, height, blocked):
    dist = {cell: 0 for cell in blocked}
    queue = deque(blocked)
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour = (x + dx, y + dy)
            if 0 <= neighbour[0] < width and 0 <= neighbour[1] < height and neighbour not in dist:
                dist[neighbour] = dist[(x, y)] + 1
                queue.append(neighbour)
    return dist


def main() -> None:
    root = Path(__file__).resolve().parent if len(sys.argv) < 2 else Path(sys.argv[1])
    for name in sorted(root.glob("*.map")):
        width, height, rows, free, blocked = load(name)
        dist = clearance(width, height, blocked)
        values = sorted(dist[cell] for cell in free)
        print(
            name.name,
            f"size={width}x{height}",
            f"free={len(free)/(width*height):.2f}",
            f"clearance median={values[len(values)//2]}",
            f"p90={values[int(len(values)*0.9)]}",
            f"max={values[-1]}",
        )
        for y in range(0, height, max(1, height // 10)):
            print("   ", rows[y][:70])
        print()


if __name__ == "__main__":
    main()
