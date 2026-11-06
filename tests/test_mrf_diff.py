# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of MatRiFFlow package
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


"""Tests for matrifflow.pymatgen.cli.mrf_diff."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import pytest
from rich.console import Console

from matrifflow.pymatgen.cli import mrf_diff
from matrifflow.pymatgen.cli.mrf_diff import (
    _copy_structures_to,
    _ignored_params_summary,
    _read_structures,
    add_args,
    diff,
    diff_incar,
    diff_structures,
    incar_add_args,
    structure_add_args,
    print_incar_groups,
)
from matrifflow.pymatgen.io.vasp.inputs import Incar

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


def test_read_structures_compressed_vasprun_detected(tmp_path: Path) -> None:
    """A gzipped vasprun.xml.gz counts as having VASP output."""
    a = _make_dir(tmp_path / "a", with_vasprun=False)
    b = tmp_path / "b"
    b.mkdir()
    (b / "POSCAR").write_text(POSCAR)
    with gzip.open(b / "vasprun.xml.gz", "wb") as f_out:
        f_out.write(b"<modeling/>")
    with pytest.raises(ValueError, match="Mixed directories"):
        _read_structures([str(a), str(b)])


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


class _StructureStub:
    """Minimal structure stand-in carrying a source_dir property."""

    def __init__(self, source_dir: str) -> None:
        self.properties = {"source_dir": source_dir}


def _struct_with_source(source_dir: str) -> _StructureStub:
    """Build a :class:`_StructureStub` for the given source directory."""
    return _StructureStub(source_dir)


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

    args = argparse.Namespace(dirs=[str(a), str(b)])
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

    args = argparse.Namespace(dirs=[str(good), str(missing)])
    with pytest.warns(UserWarning, match="No INCAR found"):
        assert diff_incar(args) == 0
    assert "ENCUT:500.0" in capsys.readouterr().out


## _read_structures: vasprun paths


def test_read_structures_all_vasprun(converged_vasp_dir, tmp_path: Path) -> None:
    """All-vasprun dirs read final structures with source_dir and energy."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    shutil.copy(converged_vasp_dir / "vasprun.xml", a / "vasprun.xml")
    shutil.copy(converged_vasp_dir / "vasprun.xml", b / "vasprun.xml")

    structures = _read_structures([str(a), str(b)])

    assert [s.properties["source_dir"] for s in structures] == [str(a), str(b)]
    assert all("final_energy" in s.properties for s in structures)
    energies = [s.properties["final_energy"] for s in structures]
    assert all(e == energies[0] for e in energies)


def test_read_structures_unparseable_vasprun_warns(tmp_path: Path) -> None:
    """A present-but-unparseable vasprun.xml is warned about and skipped."""
    d = _make_dir(tmp_path / "a", with_vasprun=True)
    with pytest.warns(UserWarning, match="Failed to read vasprun"):
        structures = _read_structures([str(d)])
    assert structures == []


## _copy_structures_to: duplicate basenames


def test_copy_structures_to_duplicate_basename(tmp_path: Path) -> None:
    """Two source dirs sharing a basename get disambiguated suffixes."""
    a = tmp_path / "src1" / "run"
    b = tmp_path / "src2" / "run"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "marker").write_text("a")
    (b / "marker").write_text("b")
    target = tmp_path / "out"
    target.mkdir()

    _copy_structures_to(
        [_struct_with_source(str(a)), _struct_with_source(str(b))],
        str(target),
    )

    assert (target / "run" / "marker").read_text() == "a"
    assert (target / "run.2" / "marker").read_text() == "b"


## diff_structures


class _FakeStructure:
    """Stand-in structure carrying only the attributes diff_structures uses."""

    def __init__(self, source_dir: str, key: str, energy=None) -> None:
        self.properties = {"source_dir": source_dir}
        self._key = key
        if energy is not None:
            self.properties["final_energy"] = energy


class _FakeMatcher:
    """Stub StructureMatcher grouping by a simple key."""

    def __init__(self, **kwargs) -> None:
        pass

    def fit(self, s1, s2) -> bool:
        return s1._key == s2._key


def _diff_args(dirs, *, energy_tol=3, copy_to=None) -> argparse.Namespace:
    return argparse.Namespace(
        dirs=dirs,
        poscar=False,
        vasprun=False,
        energy_tol=energy_tol,
        copy_to=copy_to,
    )


def test_diff_structures_groups_by_similarity(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Identical structures collapse into one group printed as Energy=N/A."""
    s1 = _FakeStructure(str(tmp_path / "a"), key="si")
    s2 = _FakeStructure(str(tmp_path / "b"), key="si")
    s3 = _FakeStructure(str(tmp_path / "c"), key="other")
    monkeypatch.setattr(
        mrf_diff, "_read_structures", lambda *a, **k: [s1, s2, s3]
    )
    monkeypatch.setattr(mrf_diff, "StructureMatcher", _FakeMatcher)

    diff_structures(_diff_args(["a", "b", "c"]))

    out = capsys.readouterr().out
    assert "Group 1" in out
    assert "Group 2" in out
    assert "Energy=N/A" in out


def test_diff_structures_energy_mismatch_splits_groups(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Same key but different energies are kept in separate groups."""
    s1 = _FakeStructure(str(tmp_path / "a"), key="si", energy=1.0)
    s2 = _FakeStructure(str(tmp_path / "b"), key="si", energy=2.0)
    monkeypatch.setattr(mrf_diff, "_read_structures", lambda *a, **k: [s1, s2])
    monkeypatch.setattr(mrf_diff, "StructureMatcher", _FakeMatcher)

    diff_structures(_diff_args(["a", "b"]))

    out = capsys.readouterr().out
    assert "Group 1" in out
    assert "Group 2" in out
    assert "Energy=1.0eV" in out
    assert "Energy=2.0eV" in out
    assert out.index("Energy=1.0eV") < out.index("Energy=2.0eV")


def test_diff_structures_equal_energy_groups_together(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Same key and same energy collapse into one group (energy matches)."""
    s1 = _FakeStructure(str(tmp_path / "a"), key="si", energy=1.0)
    s2 = _FakeStructure(str(tmp_path / "b"), key="si", energy=1.0)
    monkeypatch.setattr(mrf_diff, "_read_structures", lambda *a, **k: [s1, s2])
    monkeypatch.setattr(mrf_diff, "StructureMatcher", _FakeMatcher)

    diff_structures(_diff_args(["a", "b"]))

    out = capsys.readouterr().out
    assert "Group 1" in out
    assert "Group 2" not in out
    assert "Energy=1.0eV" in out


def test_diff_structures_copy_to(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """--copy-to copies the first (unique) structure of every group."""
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    src1.mkdir()
    src2.mkdir()
    (src1 / "marker").write_text("1")
    (src2 / "marker").write_text("2")
    s1 = _FakeStructure(str(src1), key="si")
    s2 = _FakeStructure(str(src2), key="other")
    monkeypatch.setattr(mrf_diff, "_read_structures", lambda *a, **k: [s1, s2])
    monkeypatch.setattr(mrf_diff, "StructureMatcher", _FakeMatcher)
    out = tmp_path / "out"

    diff_structures(_diff_args(["x"], copy_to=str(out)))

    assert (out / "src1" / "marker").read_text() == "1"
    assert (out / "src2" / "marker").read_text() == "2"


def test_diff_structures_real_poscar(tmp_path: Path, capsys) -> None:
    """End-to-end grouping of identical POSCAR directories."""
    dirs = [
        _make_dir(tmp_path / "a", with_vasprun=False),
        _make_dir(tmp_path / "b", with_vasprun=False),
    ]
    args = argparse.Namespace(
        dirs=[str(d) for d in dirs],
        poscar=False,
        vasprun=False,
        energy_tol=3,
        copy_to=None,
    )
    diff_structures(args)
    out = capsys.readouterr().out
    assert "Group 1" in out
    assert "Energy=N/A" in out


## CLI wiring


def test_structure_add_args() -> None:
    """--poscar/--vasprun/--copy-to/--energy-tol register and default."""
    parser = argparse.ArgumentParser()
    structure_add_args(parser)
    assert parser.get_default("func_diff") is diff_structures

    ns = parser.parse_args(["--poscar", "--energy-tol", "5"])
    assert ns.poscar is True
    assert ns.vasprun is False
    assert ns.energy_tol == 5
    assert ns.copy_to is None


def test_incar_add_args() -> None:
    """incar subcommand dispatches to diff_incar."""
    parser = argparse.ArgumentParser()
    incar_add_args(parser)
    assert parser.get_default("func_diff") is diff_incar


def test_add_args_registers_subcommands() -> None:
    """Both structure and incar subcommands parse and set their handler."""
    parser = argparse.ArgumentParser()
    add_args(parser)

    ns = parser.parse_args(["structure", "a", "b"])
    assert ns.func_diff is diff_structures
    assert ns.dirs == ["a", "b"]

    ns = parser.parse_args(["incar", "a"])
    assert ns.func_diff is diff_incar
    assert ns.dirs == ["a"]


def test_diff_dispatches_and_returns_zero() -> None:
    """diff calls the selected handler and returns 0."""
    seen = []

    def fake_func(args):
        seen.append(args)

    args = argparse.Namespace(func_diff=fake_func)
    assert diff(args) == 0
    assert seen == [args]
