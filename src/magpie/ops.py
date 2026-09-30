"""Place sorted images in magpie-output/, log every operation, undo the last apply."""

import filecmp
import json
import shutil
from contextlib import suppress
from pathlib import Path
from typing import Literal

from magpie.sort import Result

OUTPUT_DIR = "magpie-output"
LOG_FILE = ".magpie/ops.jsonl"

Mode = Literal["copy", "move", "symlink"]
Op = tuple[Path, Path]

ACTIONS = {
    "copy": shutil.copy2,
    "move": shutil.move,
    "symlink": lambda src, dst: dst.symlink_to(src),
}


def plan(folder: Path, results: list[Result]) -> list[Op]:
    """Destination of each confident image: never an existing file, already placed ones skipped."""
    ops, taken = [], set()
    for r in results:
        if r.category is None:
            continue
        dst = folder / OUTPUT_DIR / r.category / r.path.name
        n = 1
        while dst.exists() or dst in taken:
            if dst.exists() and filecmp.cmp(r.path, dst, shallow=False):
                break
            dst, n = dst.with_stem(f"{r.path.stem}-{n}"), n + 1
        else:
            taken.add(dst)
            ops.append((r.path, dst))
    return ops


def apply(folder: Path, ops: list[Op], mode: Mode) -> None:
    log = folder / LOG_FILE
    log.parent.mkdir(exist_ok=True)
    pairs = [[str(s.relative_to(folder)), str(d.relative_to(folder))] for s, d in ops]
    with log.open("a") as f:  # logged before acting, so an interrupted apply can still be undone
        f.write(json.dumps({"mode": mode, "ops": pairs}) + "\n")
    for src, dst in ops:
        dst.parent.mkdir(parents=True, exist_ok=True)
        ACTIONS[mode](src, dst)


def undo(folder: Path) -> int:
    """Revert the last apply. Returns how many operations were reverted."""
    log = folder / LOG_FILE
    if not log.exists() or not (lines := log.read_text().splitlines()):
        return 0
    entry = json.loads(lines[-1])
    ops = [(folder / s, folder / d) for s, d in entry["ops"]]
    for src, dst in reversed(ops):
        if entry["mode"] != "move":
            dst.unlink(missing_ok=True)
        elif dst.exists() and not src.exists():
            shutil.move(dst, src)
    for d in sorted({d.parent for _, d in ops} | {folder / OUTPUT_DIR}, reverse=True):
        with suppress(OSError):
            d.rmdir()
    log.write_text("".join(f"{line}\n" for line in lines[:-1]))
    return len(ops)
