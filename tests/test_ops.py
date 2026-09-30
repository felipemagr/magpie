from pathlib import Path

import pytest

from magpie import ops
from magpie.sort import Result


def snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def results(folder):
    return [Result(folder / "red.png", "red", 0.9), Result(folder / "blue.png", None, 0.4)]


@pytest.mark.parametrize("mode", ["copy", "move", "symlink"])
def test_apply_then_undo_restores_folder(folder, mode):
    before = snapshot(folder)
    ops.apply(folder, ops.plan(folder, results(folder)), mode)
    assert (folder / "magpie-output/red/red.png").read_bytes() == before[Path("red.png")]
    assert ops.undo(folder) == 1
    assert {k: v for k, v in snapshot(folder).items() if k.parts[0] != ".magpie"} == before
    assert not (folder / "magpie-output").exists()


def test_plan_never_overwrites_and_skips_placed(folder):
    taken = folder / "magpie-output/red/red.png"
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"someone else's file")
    [(_, dst)] = ops.plan(folder, results(folder))
    assert dst.name == "red-1.png"
    ops.apply(folder, [(folder / "red.png", dst)], "copy")
    assert ops.plan(folder, results(folder)) == []
