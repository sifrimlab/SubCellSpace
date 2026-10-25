#!/usr/bin/env python
# Lists exactly what the data bundle contains, with sizes and checksums, so the
# upload can be verified against it.
#
#   python scripts/zenodo_manifest.py --bundle /path/to/benchmark_data [--checksums]

import argparse
import hashlib
from pathlib import Path


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ALREADY_DEPOSITED = config.ALREADY_DEPOSITED


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--checksums", action="store_true")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    root = Path(A.bundle)

    # cache/ is written while the notebooks run and is not part of the upload
    files = sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())
    files = [p for p in files
             if p.resolve().is_file() and p.relative_to(root).parts[0] != "cache"]

    rows, by_dir = [], {}
    for path in files:
        real = path.resolve()
        size = real.stat().st_size
        rel = path.relative_to(root)
        rows.append((rel, size, sha256(real) if A.checksums else ""))
        by_dir.setdefault(rel.parts[0], [0, 0])
        by_dir[rel.parts[0]][0] += 1
        by_dir[rel.parts[0]][1] += size

    total = sum(r[1] for r in rows)
    have = sum(s for rel, s, _ in rows if str(rel) in ALREADY_DEPOSITED)
    need = [(rel, s, d) for rel, s, d in rows if str(rel) not in ALREADY_DEPOSITED]
    lines = [f"# Data bundle: {len(rows)} files, {human(total)}",
             f"# already deposited: {len(rows) - len(need)} files, {human(have)}",
             f"# STILL TO UPLOAD:   {len(need)} files, {human(total - have)}", ""]
    lines.append(f"{'directory':<14} {'files':>6} {'size':>12}")
    for name, (count, size) in sorted(by_dir.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"{name:<14} {count:>6} {human(size):>12}")
    lines += ["", f"{'file':<58} {'size':>12}" + ("  sha256" if A.checksums else "")]
    for rel, size, digest in rows:
        lines.append(f"{str(rel):<58} {human(size):>12}" + (f"  {digest}" if digest else ""))

    text = "\n".join(lines)
    print(text)
    if A.out:
        Path(A.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
