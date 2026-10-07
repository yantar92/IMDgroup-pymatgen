"""Tests for IMDgroup.pymatgen.io.vasp.vaspdir.

Directory-level behaviour (convergence, energy, forces, caching) is
exercised against a real converged VASP run, so these tests are gated
behind the generated ``vasp_converged`` / ``vasp_converged_relax``
fixtures and marked ``integration``.
"""

from __future__ import annotations

import warnings

import pytest

from IMDgroup.pymatgen.io.vasp.diagnostics import VaspWarning
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

pytestmark = pytest.mark.integration


@pytest.fixture
def isolated_lmdb_cache(tmp_path, monkeypatch):
    """Redirect the LMDB cache to a per-test dir and reset class state."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    IMDGVaspDir._lmdb_env = None
    IMDGVaspDir._lmdb_db = None
    IMDGVaspDir._lmdb_meta_db = None
    with IMDGVaspDir._pending_lock:
        IMDGVaspDir._pending_writes.clear()
    yield
    if IMDGVaspDir._lmdb_env is not None:
        IMDGVaspDir._lmdb_env.close()
        IMDGVaspDir._lmdb_env = None
        IMDGVaspDir._lmdb_db = None
        IMDGVaspDir._lmdb_meta_db = None


def test_contains_by_filename(converged_vasp_dir) -> None:
    """``in`` checks file names, not Path objects."""
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert "INCAR" in d
    assert "vasprun.xml" in d
    assert "nonexistent" not in d


def test_converged_scf(converged_vasp_dir) -> None:
    """A converged SCF run is fully converged."""
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert d.converged is True
    assert d.converged_electronic is True
    assert d.converged_ionic is True


def test_converged_ionic_is_bool(converged_vasp_dir) -> None:
    """converged_ionic returns a bool, not a raw OUTCAR pattern list."""
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert isinstance(d.converged_ionic, bool)


def test_converged_relax(converged_relax_vasp_dir) -> None:
    """A converged relaxation run is fully converged."""
    d = IMDGVaspDir(str(converged_relax_vasp_dir))
    assert d.converged is True


def test_converged_ionic_outcar_fallback(converged_relax_vasp_dir) -> None:
    """converged_ionic is read from OUTCAR when vasprun.xml is absent."""
    (converged_relax_vasp_dir / "vasprun.xml").unlink()
    d = IMDGVaspDir(str(converged_relax_vasp_dir))
    assert d.converged_ionic is True
    assert isinstance(d.converged_ionic, bool)


def test_final_energy(converged_vasp_dir) -> None:
    """final_energy is a float for a converged run."""
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert isinstance(d.final_energy, float)


def test_max_force_below_ediffg(converged_relax_vasp_dir) -> None:
    """Converged means max force below |EDIFFG|, not necessarily zero.

    VASP's force-based convergence criterion (EDIFFG < 0) is that all
    forces fall below |EDIFFG|.  This fixture relaxes a single Na atom
    to exactly zero force, but the general contract is the threshold
    bound, so the test asserts that instead.
    """
    d = IMDGVaspDir(str(converged_relax_vasp_dir))
    ediffg = d["INCAR"]["EDIFFG"]
    assert ediffg < 0
    assert 0.0 <= d.max_force() < abs(ediffg)


def test_max_force_no_outcar(tmp_path) -> None:
    """max_force is None when there is no OUTCAR."""
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d.max_force() is None


def test_energy_accuracy_warning(converged_relax_vasp_dir) -> None:
    """An ISIF=3 relaxation records the energy-accuracy warning."""
    d = IMDGVaspDir(str(converged_relax_vasp_dir))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = d.warnings.names()
    assert "energy_accuracy" in names


def test_lmdb_cache_round_trip(converged_vasp_dir, isolated_lmdb_cache) -> None:
    """Parsed files are cached to LMDB and reused on a second load."""
    first = IMDGVaspDir(str(converged_vasp_dir))
    assert first["INCAR"] is not None
    IMDGVaspDir.flush_cache()

    cached = IMDGVaspDir._lmdb_get(first._cache_key)
    assert cached is not None
    assert cached["version"] == IMDGVaspDir.CACHE_VERSION
    assert "INCAR" in cached["parsed_files"]

    second = IMDGVaspDir(str(converged_vasp_dir))
    assert second["INCAR"] is not None
