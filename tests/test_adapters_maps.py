# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
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


"""Tests for IMDgroup.gorun.adapters.maps_adapter."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import IMDgroup.gorun.adapters.maps_adapter as maps_mod
import IMDgroup.pymatgen.io.vasp.sets as sets_mod

from IMDgroup.gorun.adapters.maps_adapter import MapsAdapter


def _adapter(**kwargs) -> MapsAdapter:
    defaults = {"kpoints": "4x4x4"}
    defaults.update(kwargs)
    return MapsAdapter(**defaults)


# --- __init__ ---------------------------------------------------------------


def test_init_defaults() -> None:
    """Unspecified options fall back to the documented defaults."""
    adapter = _adapter()
    assert adapter.kpoints == "4x4x4"
    assert adapter.frac_tol == 0.5
    assert adapter.max_strain == 0.1
    assert adapter.skip_relax is False
    assert adapter.sublattice_cutoff is None
    assert adapter.maps_args == []


def test_init_explicit() -> None:
    """Every option is stored as passed."""
    adapter = _adapter(
        frac_tol=0.3,
        max_strain=0.05,
        skip_relax=True,
        sublattice_cutoff=0.02,
        maps_args=["-d", "--verbose"],
    )
    assert adapter.frac_tol == 0.3
    assert adapter.max_strain == 0.05
    assert adapter.skip_relax is True
    assert adapter.sublattice_cutoff == 0.02
    assert adapter.maps_args == ["-d", "--verbose"]


def test_class_attributes() -> None:
    """The adapter advertises its name and log file."""
    assert MapsAdapter.name == "maps"
    assert MapsAdapter.log_file == "maps.out"


# --- validate_environment ---------------------------------------------------


def test_validate_environment_missing_commands_raises(monkeypatch) -> None:
    """Missing maps and pollmach both are reported in the OSError."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    with pytest.raises(OSError, match="maps, pollmach"):
        _adapter().validate_environment()


def test_validate_environment_missing_single_command(monkeypatch) -> None:
    """Only the absent command is named."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None if cmd == "maps" else cmd)
    with pytest.raises(OSError, match="maps"):
        _adapter().validate_environment()


def test_validate_environment_ok(monkeypatch) -> None:
    """With both commands present, VASP validation is delegated to."""
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    calls: list[str] = []

    class _FakeVaspAdapter:
        def validate_environment(self) -> None:
            calls.append("validate")

        def setup_commands(self, server_config: dict) -> str:
            return "module load vasp"

    monkeypatch.setattr(maps_mod, "VaspAdapter", _FakeVaspAdapter)
    _adapter().validate_environment()
    assert calls == ["validate"]


# --- is_valid_input ---------------------------------------------------------


def test_is_valid_input_requires_lat(tmp_path: Path) -> None:
    """Without lat.in the directory is never valid."""
    assert _adapter().is_valid_input(tmp_path) is False


def test_is_valid_input_bad_prototype(monkeypatch, tmp_path: Path) -> None:
    """A lat.in whose prototype VASP input fails to build is invalid."""
    (tmp_path / "lat.in").touch()

    class _BadInputSet:
        def __init__(self, directory=None) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(sets_mod, "IMDDerivedInputSet", _BadInputSet)
    assert _adapter().is_valid_input(tmp_path) is False


def test_is_valid_input_ok(monkeypatch, tmp_path: Path) -> None:
    """A lat.in with a working prototype input set is valid, and the
    scratch directory is removed afterwards."""
    (tmp_path / "lat.in").touch()

    class _OkInputSet:
        def __init__(self, directory=None) -> None:
            self.directory = directory

        def write_input(self, test_dir: Path) -> None:
            test_dir.mkdir()

    monkeypatch.setattr(sets_mod, "IMDDerivedInputSet", _OkInputSet)
    assert _adapter().is_valid_input(tmp_path) is True
    assert not (tmp_path / ".__test").exists()


# --- is_converged -----------------------------------------------------------


def test_is_converged_always_false(tmp_path: Path) -> None:
    """maps runs until killed or exhausted, so it never reports converged."""
    assert _adapter().is_converged(tmp_path) is False


# --- has_previous_output ----------------------------------------------------


def test_has_previous_output_none(tmp_path: Path) -> None:
    """A clean directory has no previous output."""
    assert _adapter().has_previous_output(tmp_path) is False


@pytest.mark.parametrize("marker", ["maps.log", "pollmach.log"])
def test_has_previous_output_marker(tmp_path: Path, marker: str) -> None:
    """Either ATAT log file counts as previous output."""
    (tmp_path / marker).touch()
    assert _adapter().has_previous_output(tmp_path) is True


# --- prepare_inputs ---------------------------------------------------------


def test_prepare_inputs_removes_lock_files(tmp_path: Path) -> None:
    """Leftover lock files from a previous run are removed."""
    (tmp_path / "pollmach_is_running").touch()
    (tmp_path / "maps_is_running").touch()
    _adapter().prepare_inputs(tmp_path)
    assert not (tmp_path / "pollmach_is_running").exists()
    assert not (tmp_path / "maps_is_running").exists()


def test_prepare_inputs_no_lock_files(tmp_path: Path) -> None:
    """No lock files is not an error."""
    _adapter().prepare_inputs(tmp_path)


# --- setup_commands ---------------------------------------------------------


def test_setup_commands_delegates_to_vasp(monkeypatch) -> None:
    """The maps wrapper inherits the VASP module environment."""
    cfg = {"vasp": {"setup": "module load vasp"}}

    class _FakeVaspAdapter:
        def validate_environment(self) -> None:
            pass

        def setup_commands(self, server_config: dict) -> str:
            assert server_config is cfg
            return "module load vasp"

    monkeypatch.setattr(maps_mod, "VaspAdapter", _FakeVaspAdapter)
    assert _adapter().setup_commands(cfg) == "module load vasp"


# --- generate_run_commands --------------------------------------------------


def test_generate_run_commands_defaults(tmp_path: Path) -> None:
    """The wrapper exports GORUN_SKIP_SETUP and chains maps + pollmach."""
    script = _adapter().generate_run_commands(tmp_path, {})
    assert "export GORUN_SKIP_SETUP=1" in script
    assert "maps  &" in script
    assert "sleep 5" in script
    assert (
        "pollmach gorun atat-local --local "
        "--kpoints=4x4x4 --frac-tol=0.5 --max-strain=0.1" in script
    )
    assert "--skip-relax" not in script
    assert "--sublattice-cutoff" not in script


def test_generate_run_commands_maps_args(tmp_path: Path) -> None:
    """Extra maps arguments are appended to the maps command."""
    script = _adapter(maps_args=["-d", "--verbose"]).generate_run_commands(tmp_path)
    assert "maps -d --verbose &" in script


def test_generate_run_commands_skip_relax(tmp_path: Path) -> None:
    """--skip-relax is forwarded to pollmach only when requested."""
    script = _adapter(skip_relax=True).generate_run_commands(tmp_path)
    assert "--skip-relax" in script


def test_generate_run_commands_sublattice_cutoff(tmp_path: Path) -> None:
    """A non-None sublattice cutoff is forwarded to pollmach."""
    script = _adapter(sublattice_cutoff=0.02).generate_run_commands(tmp_path)
    assert "--sublattice-cutoff=0.02" in script


# --- backup_excludes --------------------------------------------------------


def test_backup_excludes() -> None:
    """WAVECAR and ATAT scratch directories are excluded from backups."""
    assert _adapter().backup_excludes() == ["WAVECAR", "__*"]
