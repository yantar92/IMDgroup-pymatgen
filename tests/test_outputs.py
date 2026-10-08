"""Tests for IMDgroup.pymatgen.io.vasp.outputs.

Pure parsing/statistics helpers are tested with synthetic text; the
real ``Outcar``/``Vasprun`` parsers are tested against the generated
VASP fixtures and marked ``integration``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from IMDgroup.pymatgen.io.vasp.diagnostics import VaspWarning
from IMDgroup.pymatgen.io.vasp.outputs import (
    Outcar,
    Vasplog,
    VasplogMixin,
    Vasprun,
    _compute_timing_stats,
    _mean_std,
    read_outcar_timing_stats,
)


# --- _mean_std --------------------------------------------------------------


def test_mean_std_empty() -> None:
    """An empty sample has NaN mean and std."""
    result = _mean_std([])
    assert result.n == 0
    assert np.isnan(result.mean)
    assert np.isnan(result.std)


def test_mean_std_single() -> None:
    """A single value has that mean and NaN (ddof=1) std."""
    result = _mean_std([5.0])
    assert result.n == 1
    assert result.mean == 5.0
    assert np.isnan(result.std)


def test_mean_std_multi() -> None:
    """Multiple values compute the sample mean and std."""
    result = _mean_std([1.0, 2.0, 3.0])
    assert result.n == 3
    assert result.mean == 2.0
    assert result.std == 1.0


# --- _compute_timing_stats --------------------------------------------------


def test_compute_timing_stats_groups_by_ionic_step() -> None:
    """LOOP times accumulate per SCF cycle and per ionic step."""
    lines = [
        "  LOOP:  cpu time    1.0: real time    2.0",
        "  LOOP:  cpu time    3.0: real time    4.0",
        "  Ionic step  2",
        "  LOOP:  cpu time    5.0: real time    6.0",
    ]
    stats = _compute_timing_stats(lines)
    assert stats.scf.n == 3
    assert stats.scf.mean == pytest.approx(4.0)
    assert stats.ionic.n == 2
    assert stats.ionic.mean == pytest.approx(6.0)


def test_compute_timing_stats_nsw0_iteration_fallback() -> None:
    """Without an Ionic step marker, the Iteration index is used."""
    lines = [
        "Iteration  1(  2)",
        "  LOOP:  cpu time    1.0: real time    1.0",
    ]
    stats = _compute_timing_stats(lines)
    assert stats.scf.n == 1
    assert stats.ionic.n == 1
    assert stats.ionic.mean == pytest.approx(1.0)


# --- _dedup_lines -----------------------------------------------------------


def test_dedup_lines() -> None:
    """Lines are stripped and deduplicated with occurrence counts."""
    lines, counts = VasplogMixin._dedup_lines(["  a  ", "a", "b", "a"])
    assert lines == ["a", "b"]
    assert counts == {"a": 3, "b": 1}


# --- Vasplog ----------------------------------------------------------------


def test_vasplog_parses_warnings(tmp_path) -> None:
    """Known warning lines are classified by name."""
    path = tmp_path / "stdout"
    path.write_text(
        "ZBRENT: fatal error in bracketing\n"
        "BRMIX: very serious problems\n"
        "normal line\n"
    )
    vplog = Vasplog(str(path))
    names = vplog.warnings.names()
    assert "zbrent" in names
    assert "brmix" in names
    assert vplog.warnings["zbrent"].count == 1


def test_vasplog_excludes_false_positives(tmp_path) -> None:
    """Lines matching an exclude pattern are not classified."""
    path = tmp_path / "stdout"
    path.write_text("kinetic energy error for atom=1\nsome normal text\n")
    vplog = Vasplog(str(path))
    assert not vplog.warnings.has("unclassified")


def test_vasplog_parses_progress(tmp_path) -> None:
    """Progress lines are classified under their progress type."""
    path = tmp_path / "stdout"
    path.write_text("DAV:  1  -0.47E+03  0.12E-01 -0.11E+00\n")
    vplog = Vasplog(str(path))
    assert vplog.progress.has("00SCF")


def test_vasp_log_files_prefers_non_outcar(tmp_path) -> None:
    """OUTCAR is excluded when another log file is present."""
    (tmp_path / "slurm-1.out").write_text("a")
    (tmp_path / "OUTCAR").write_text("b")
    (tmp_path / "INCAR").write_text("c")
    names = [f.name for f in Vasplog.vasp_log_files(str(tmp_path))]
    assert "slurm-1.out" in names
    assert "OUTCAR" not in names
    assert "INCAR" not in names


def test_vasp_log_files_outcar_only(tmp_path) -> None:
    """OUTCAR is returned when it is the only log file."""
    (tmp_path / "OUTCAR").write_text("b")
    names = [f.name for f in Vasplog.vasp_log_files(str(tmp_path))]
    assert names == ["OUTCAR"]


# --- read_outcar_timing_stats (synthetic) ------------------------------------


def test_read_outcar_timing_stats_synthetic(tmp_path) -> None:
    """Timing statistics stream from an OUTCAR file."""
    path = tmp_path / "OUTCAR"
    path.write_text(
        "  LOOP:  cpu time    1.0: real time    2.0\n"
        "  LOOP:  cpu time    3.0: real time    4.0\n"
    )
    stats = read_outcar_timing_stats(str(path))
    assert stats.scf.n == 2
    assert stats.scf.mean == pytest.approx(3.0)


# --- fixture-gated tests ----------------------------------------------------


@pytest.mark.integration
def test_outcar_final_forces(converged_relax_vasp_dir) -> None:
    """A real OUTCAR yields one force vector per atom."""
    forces = Outcar(str(converged_relax_vasp_dir / "OUTCAR")).final_forces
    assert forces is not None
    assert forces.shape == (1, 3)


@pytest.mark.integration
def test_outcar_timing_stats(converged_relax_vasp_dir) -> None:
    """A real OUTCAR yields non-empty SCF and ionic timing series."""
    stats = Outcar(str(converged_relax_vasp_dir / "OUTCAR")).timing_stats
    assert stats.scf.n > 0
    assert stats.ionic.n > 0


@pytest.mark.integration
def test_read_outcar_timing_stats_real(converged_relax_vasp_dir) -> None:
    """read_outcar_timing_stats parses a real OUTCAR."""
    stats = read_outcar_timing_stats(
        str(converged_relax_vasp_dir / "OUTCAR"))
    assert stats.scf.n > 0


@pytest.mark.integration
def test_vasprun_converged(converged_relax_vasp_dir) -> None:
    """A real, converged vasprun reports electronic+ionic convergence."""
    run = Vasprun(str(converged_relax_vasp_dir / "vasprun.xml"))
    assert run.converged_electronic is True
    assert run.converged_ionic is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(run.final_energy, float)


@pytest.mark.integration
def test_vasprun_energy_accuracy_warning(converged_relax_vasp_dir) -> None:
    """ISIF=3 relaxation records the energy-accuracy warning."""
    run = Vasprun(str(converged_relax_vasp_dir / "vasprun.xml"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = run.warnings.names()
    assert "energy_accuracy" in names
