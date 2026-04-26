#!/usr/bin/env python3
"""Export every image blob from every commit on each ``origin/*`` branch (GTsingh600).

One file per (remote branch, commit, path). Filename:

  <path with / -> __>__branch-<short_name>__<commit ISO8601>__<full SHA><ext>

``short_name`` is the ref with ``origin/`` stripped (e.g. ``main``, ``cleaned``).

Writes ``manifest.csv``:
  remote_branch, path_in_repo, commit_full, commit_date_iso8601, size_bytes, output_file
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMG = re.compile(r"\.(png|jpe?g|gif|webp|svg|ico|bmp|tiff?)$", re.I)


def git_out(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "exported_images_gtsingh600_origin"


def origin_branches() -> list[tuple[str, str]]:
    """Pairs of (full ref e.g. origin/main, short label e.g. main)."""
    raw = git_out("branch", "-r", "--list", "origin/*").strip().split("\n")
    out: list[tuple[str, str]] = []
    for line in raw:
        ref = line.strip()
        if not ref or ref.endswith("/HEAD") or "->" in ref:
            continue
        # line may be "  origin/main"
        ref = ref.split()[-1]
        if not ref.startswith("origin/"):
            continue
        short = ref[len("origin/") :].replace("/", "_")
        out.append((ref, short))
    return sorted(out, key=lambda x: x[0])


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / "manifest.csv"

    branches = origin_branches()
    if not branches:
        print("No origin/* branches found.", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str, str, str, int, str]] = []
    written = 0
    skipped = 0

    for ref, branch_label in branches:
        commits = [
            c for c in git_out("rev-list", ref).strip().split("\n") if c
        ]
        for commit in commits:
            try:
                iso = git_out("show", "-s", "--format=%cI", commit).strip().split("\n")[0]
            except subprocess.CalledProcessError:
                continue
            iso_fs = iso.replace(":", "-")

            try:
                tree_lines = git_out("ls-tree", "-r", "--name-only", commit).strip()
            except subprocess.CalledProcessError:
                continue
            paths = [ln for ln in tree_lines.split("\n") if ln and IMG.search(ln)]

            for rel in paths:
                try:
                    data = git_bytes("show", f"{commit}:{rel}")
                except subprocess.CalledProcessError:
                    skipped += 1
                    continue

                p = Path(rel)
                stem_flat = str(p.with_suffix("")).replace("/", "__")
                ext = p.suffix.lower() if p.suffix else ""
                bsafe = re.sub(r"[^\w.\-()+]", "_", branch_label)
                fname = f"{stem_flat}__branch-{bsafe}__{iso_fs}__{commit}{ext}"
                if len(fname) > 220:
                    fname = (
                        f"{stem_flat[:80]}__branch-{bsafe}__{iso_fs}__{commit}{ext}"
                    )

                dest = OUT_DIR / fname
                if dest.exists():
                    for i in range(1, 10000):
                        alt = OUT_DIR / f"{stem_flat}__branch-{bsafe}__{iso_fs}__{commit}__n{i}{ext}"
                        if len(alt.name) > 240:
                            alt = OUT_DIR / f"{stem_flat[:60]}__br-{bsafe}__{commit[:12]}__n{i}{ext}"
                        if not alt.exists():
                            dest = alt
                            fname = alt.name
                            break

                dest.write_bytes(data)
                written += 1
                rows.append((ref, rel, commit, iso, len(data), fname))

    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "remote_branch_ref",
                "branch_short",
                "path_in_repo",
                "commit_full",
                "commit_date_iso8601",
                "size_bytes",
                "output_file",
            ]
        )
        for ref, rel, commit, iso, nbytes, fname in rows:
            short = ref.replace("origin/", "", 1)
            w.writerow([ref, short, rel, commit, iso, nbytes, fname])

    print(f"Branches: {len(branches)}", file=sys.stderr)
    print(f"Wrote {written} files to {OUT_DIR}", file=sys.stderr)
    print(f"Skipped {skipped} missing blobs", file=sys.stderr)
    print(f"Manifest: {manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
