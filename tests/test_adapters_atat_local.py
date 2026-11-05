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


"""Tests for matrifflow.gorun.adapters.atat_local."""

from __future__ import annotations

from pathlib import Path

import pytest

import matrifflow.gorun.adapters.atat_local as atat_local

from matrifflow.gorun.adapters.atat_local import AtatLocalAdapter


def _adapter(**kwargs) -> AtatLocalAdapter:
    defaults = {"kpoints": "4x4x4"}
    defaults.update(kwargs)
    return AtatLocalAdapter(**defaults)


# --- __init__ ---------------------------------------------------------------


def test_init_defaults() -> None:
    """Unspecified options fall back to the documented defaults."""
    adapter = _adapter()
    assert adapter.kpoints == "4x4x4"
    assert adapter.frac_tol == 0
    assert adapter.max_strain == 0.1
    assert adapter.skip_relax is False
    assert adapter.sublattice_cutoff is None


def test_init_explicit() -> None:
    """Every option is stored as passed."""
    adapter = _adapter(
        frac_tol=0.3,
        max_strain=0.05,
        skip_relax=True,
        sublattice_cutoff=0.02,
    )
    assert adapter.frac_tol == 0.3
    assert adapter.max_strain == 0.05
    assert adapter.skip_relax is True
    assert adapter.sublattice_cutoff == 0.02


def test_class_attributes() -> None:
    """The adapter advertises its name and log file."""
    assert AtatLocalAdapter.name == "atat-local"
    assert AtatLocalAdapter.log_file == "atat-local.out"


# --- validate_environment ---------------------------------------------------


def test_validate_environment_is_noop() -> None:
    """atat-local adds no environment checks; it never raises."""
    _adapter().validate_environment()


# --- is_valid_input ---------------------------------------------------------


def test_is_valid_input_requires_str_out(tmp_path: Path) -> None:
    """Without str.out the directory is never valid."""
    assert _adapter().is_valid_input(tmp_path) is False


def test_is_valid_input_ok(tmp_path: Path) -> None:
    """A str.out file makes the directory valid."""
    (tmp_path / "str.out").touch()
    assert _adapter().is_valid_input(tmp_path) is True


# --- is_converged -----------------------------------------------------------


def test_is_converged_requires_energy(tmp_path: Path) -> None:
    """Without an energy file the directory is not converged."""
    assert _adapter().is_converged(tmp_path) is False


def test_is_converged_ok(tmp_path: Path) -> None:
    """An energy file marks the directory as converged."""
    (tmp_path / "energy").touch()
    assert _adapter().is_converged(tmp_path) is True


# --- has_previous_output ----------------------------------------------------


def test_has_previous_output_none(tmp_path: Path) -> None:
    """A clean directory has no previous output."""
    assert _adapter().has_previous_output(tmp_path) is False


@pytest.mark.parametrize("marker", ["ATAT", "energy", "error"])
def test_has_previous_output_marker(tmp_path: Path, marker: str) -> None:
    """Any ATAT/energy/error marker counts as previous output."""
    (tmp_path / marker).mkdir()
    assert _adapter().has_previous_output(tmp_path) is True


# --- prepare_inputs ---------------------------------------------------------


def test_prepare_inputs_is_noop(tmp_path: Path) -> None:
    """ATAT input is derived inside the run command, not here."""
    _adapter().prepare_inputs(tmp_path)


# --- setup_commands ---------------------------------------------------------


def test_setup_commands_delegates_to_vasp(monkeypatch) -> None:
    """The heredoc inherits the VASP module environment."""
    cfg = {"vasp": {"setup": "module load vasp"}}

    class _FakeVaspAdapter:
        def setup_commands(self, server_config: dict) -> str:
            assert server_config is cfg
            return "module load vasp"

    monkeypatch.setattr(atat_local, "VaspAdapter", _FakeVaspAdapter)
    assert _adapter().setup_commands(cfg) == "module load vasp"


# --- generate_run_commands --------------------------------------------------


def test_generate_run_commands_defaults(tmp_path: Path) -> None:
    """The heredoc calls run_atat_structure with the adapter's options."""
    script = _adapter().generate_run_commands(tmp_path)
    assert (
        "from matrifflow.gorun.gorun_atat_local import run_atat_structure" in script
    )
    assert "kpoints='4x4x4'" in script
    assert "frac_tol=0" in script
    assert "max_strain=0.1" in script
    assert "skip_relax=False" in script
    assert "sublattice_cutoff=None" in script


def test_generate_run_commands_explicit_options(tmp_path: Path) -> None:
    """Every non-default option is serialized into the heredoc."""
    script = _adapter(
        kpoints="2x2x1",
        frac_tol=0.25,
        max_strain=0.05,
        skip_relax=True,
    ).generate_run_commands(tmp_path)
    assert "kpoints='2x2x1'" in script
    assert "frac_tol=0.25" in script
    assert "max_strain=0.05" in script
    assert "skip_relax=True" in script
    assert "sublattice_cutoff=None" in script


def test_generate_run_commands_sublattice_cutoff(tmp_path: Path) -> None:
    """A non-None sublattice cutoff is serialized as a float literal."""
    script = _adapter(sublattice_cutoff=0.02).generate_run_commands(tmp_path)
    assert "sublattice_cutoff=0.02" in script


# --- backup_excludes --------------------------------------------------------


def test_backup_excludes() -> None:
    """WAVECAR is excluded from backups."""
    assert _adapter().backup_excludes() == ["WAVECAR"]
