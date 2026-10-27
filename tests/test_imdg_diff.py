# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-pymatgen package
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Tests for IMDgroup.pymatgen.cli.imdg_diff."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from IMDgroup.pymatgen.cli.imdg_diff import (
    _copy_structures_to,
    _ignored_params_summary,
    _read_structures,
    diff_incar,
    print_incar_groups,
)
from IMDgroup.pymatgen.io.vasp.inputs import Incar

POSCAR = """Si
1.0
5.431 0.0 0.0
0.0 5.431 0.0
0.0 0.0 5.431
Si
1
Direct
0.0 0.0 0.0
"""


def _make_dir(path: Path, *, with_vasprun: bool) -> Path:
    """Create a VASP directory with a POSCAR and optional vasprun.xml."""
    path.mkdir(parents=True)
    (path / "POSCAR").write_text(POSCAR)
    if with_vasprun:
        (path / "vasprun.xml").write_text("<modeling/>")
    return path


def test_read_structures_all_poscar(tmp_path: Path) -> None:
    """Without output and without flags, all POSCARs are read."""
    dirs = [
        _make_dir(tmp_path / "a", with_vasprun=False),
        _make_dir(tmp_path / "b", with_vasprun=False),
    ]
    structures = _read_structures([str(d) for d in dirs])
    assert [s.properties["source_dir"] for s in structures] == [
        str(d) for d in dirs
    ]


def test_read_structures_mixed_raises(tmp_path: Path) -> None:
    """A mixture of output and input-only dirs is an error."""
    dirs = [
        _make_dir(tmp_path / "a", with_vasprun=False),
        _make_dir(tmp_path / "b", with_vasprun=True),
    ]
    with pytest.raises(ValueError, match="Mixed directories"):
        _read_structures([str(d) for d in dirs])


def test_read_structures_force_poscar_mixed(tmp_path: Path) -> None:
    """--poscar reads POSCARs even when the outputs are mixed."""
    dirs = [
        _make_dir(tmp_path / "a", with_vasprun=False),
        _make_dir(tmp_path / "b", with_vasprun=True),
    ]
    structures = _read_structures(
        [str(d) for d in dirs], force_poscar=True
    )
    assert len(structures) == 2


def test_read_structures_force_vasprun_skips(tmp_path: Path) -> None:
    """--vasprun warns and skips directories without vasprun.xml."""
    dirs = [
        _make_dir(tmp_path / "a", with_vasprun=False),
    ]
    with pytest.warns(UserWarning, match="No vasprun.xml"):
        structures = _read_structures(
            [str(d) for d in dirs], force_vasprun=True
        )
    assert structures == []


## _copy_structures_to


def _struct_with_source(source_dir: str) -> SimpleNamespace:
    """A minimal structure stand-in carrying a source_dir property."""
    return SimpleNamespace(properties={"source_dir": source_dir})


def test_copy_structures_to_relative(monkeypatch, tmp_path: Path) -> None:
    """A relative source dir is copied under its own name."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "run"
    src.mkdir()
    (src / "INCAR").write_text("ENCUT = 500\n")
    target = tmp_path / "out"
    target.mkdir()

    _copy_structures_to([_struct_with_source("run")], str(target))

    assert (target / "run" / "INCAR").is_file()


def test_copy_structures_to_absolute(tmp_path: Path) -> None:
    """An absolute source dir is copied under its basename."""
    src = tmp_path / "run"
    src.mkdir()
    (src / "INCAR").write_text("ENCUT = 500\n")
    target = tmp_path / "out"
    target.mkdir()

    _copy_structures_to([_struct_with_source(str(src))], str(target))

    assert (target / "run" / "INCAR").is_file()


## _ignored_params_summary


def _grouped_incars():
    """Build two groups sharing an ignored NELM tag."""
    incars = [
        Incar({"ENCUT": 500, "NELM": 100, "SYSTEM": "dir_a"}),
        Incar({"ENCUT": 500, "NELM": 100, "SYSTEM": "dir_b"}),
        Incar({"ENCUT": 600, "NELM": 200, "SYSTEM": "dir_c"}),
    ]
    common, groups = Incar.group_incars(incars)
    assert common is not None
    return groups


def test_ignored_params_summary_collects_distinct_values() -> None:
    """Distinct values of grouping-ignored tags are collected in order."""
    assert _ignored_params_summary(_grouped_incars()) == ["NELM:100,200"]


def test_ignored_params_summary_excludes_system() -> None:
    """The SYSTEM tag is excluded even when it differs across groups."""
    summary = " ".join(_ignored_params_summary(_grouped_incars()))
    assert "SYSTEM" not in summary


def test_ignored_params_summary_empty() -> None:
    """Groups without any ignored tag produce an empty summary."""
    groups = [
        [Incar({"ENCUT": 500, "SYSTEM": "a"})],
        [Incar({"ENCUT": 600, "SYSTEM": "b"})],
    ]
    assert _ignored_params_summary(groups) == []


## print_incar_groups


def _render_groups(common, groups, max_names=None) -> str:
    """Render print_incar_groups to a string via a recording console."""
    console = Console(record=True, width=200)
    print_incar_groups(common, groups, console=console, max_names=max_names)
    return console.export_text()


def test_print_incar_groups_shows_common_and_groups() -> None:
    """Common params, per-group diffs, and names are all rendered."""
    groups = _grouped_incars()
    common = Incar({"ENCUT": 500, "NELM": 100})
    text = _render_groups(common, groups)
    assert "ENCUT:500.0" in text
    assert "dir_a dir_b" in text
    assert "dir_c" in text


def test_print_incar_groups_max_names_collapses() -> None:
    """max_names truncates the directory list with an ellipsis."""
    incars = [
        Incar({"ENCUT": 500, "SYSTEM": f"dir_{i}"}) for i in range(3)
    ]
    common, groups = Incar.group_incars(incars)
    assert common is not None
    text = _render_groups(common, groups, max_names=2)
    assert "dir_0 dir_1 ..." in text
    assert "dir_2" not in text


def test_print_incar_groups_omits_ignored_diff() -> None:
    """Ignored tags are summarised once, never repeated as group diffs."""
    groups = _grouped_incars()
    common = Incar({"ENCUT": 500})
    text = _render_groups(common, groups)
    # NELM differs across groups but is grouping-ignored, so it shows up
    # only in the "Ignored parameters" summary line.
    assert "Ignored parameters: NELM:100,200" in text
    group_lines = [
        line for line in text.splitlines() if line.startswith("Group")
    ]
    assert group_lines
    assert all("NELM" not in line for line in group_lines)


## diff_incar


def test_diff_incar_reads_and_groups(tmp_path: Path, capsys) -> None:
    """INCAR files are read, grouped, and reported with exit code 0."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "INCAR").write_text("ENCUT = 500\n")
    (b / "INCAR").write_text("ENCUT = 600\n")

    args = SimpleNamespace(dirs=[str(a), str(b)])
    assert diff_incar(args) == 0
    out = capsys.readouterr().out
    assert "ENCUT:500.0" in out
    assert "ENCUT:600.0" in out


def test_diff_incar_missing_incar_warns(tmp_path: Path, capsys) -> None:
    """A directory without INCAR warns and is skipped, not fatal."""
    good = tmp_path / "a"
    missing = tmp_path / "b"
    good.mkdir()
    missing.mkdir()
    (good / "INCAR").write_text("ENCUT = 500\n")

    args = SimpleNamespace(dirs=[str(good), str(missing)])
    with pytest.warns(UserWarning, match="No INCAR found"):
        assert diff_incar(args) == 0
    assert "ENCUT:500.0" in capsys.readouterr().out
