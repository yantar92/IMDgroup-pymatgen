"""Tests for IMDgroup.pymatgen.io.vasp.vaspdir.

Directory-level behaviour (convergence, energy, forces, caching) is
exercised against a real converged VASP run, so these tests are gated
behind the generated ``vasp_converged`` / ``vasp_converged_relax``
fixtures and marked ``integration``.
"""

from __future__ import annotations

import os
import shutil
import signal
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar

from IMDgroup.pymatgen.io.vasp.diagnostics import VaspWarning
from IMDgroup.pymatgen.io.vasp.vaspdir import (
    IMDGVaspDir,
    TimeoutException,
    timeout_handler,
)

pytestmark = pytest.mark.integration


def _write_poscar(dirpath: Path, a: float, filename: str = "POSCAR") -> None:
    """Write a single-atom Li VASP structure file with cubic lattice ``a``.

    Helper to build minimal VASP directories without the heavy
    generated fixtures; ``IMDGVaspDir`` only needs a POSCAR (or
    CONTCAR) to resolve structure accessors and ``prev_dirs``.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        lattice=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        species=["Li"],
        coords=[[0.0, 0.0, 0.0]],
    )
    Poscar(structure).write_file(dirpath / filename)


def _grid_frac(n: int) -> list[list[float]]:
    """Fractional coordinates of an ``n x n x n`` grid."""
    return [[i / n, j / n, k / n]
            for i in range(n) for j in range(n) for k in range(n)]


def _write_grid_poscars(
        dirpath: Path,
        n: int = 3,
        a: float = 10.0,
        displace: tuple[int, tuple[float, float, float]] | None = None,
) -> None:
    """Write POSCAR + CONTCAR for an ``n^3`` Li grid in a cubic cell.

    ``displace`` is ``(atom_index, fractional_delta)`` applied to the
    CONTCAR only, to build an initial/final pair with a known shift.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    frac = _grid_frac(n)
    species = ["Li"] * len(frac)
    init = Structure(lattice=lattice, species=species, coords=frac)
    Poscar(init).write_file(dirpath / "POSCAR")
    final_frac = [list(f) for f in frac]
    if displace is not None:
        idx, delta = displace
        final_frac[idx] = [x + d for x, d in zip(final_frac[idx], delta)]
    final = Structure(lattice=lattice, species=species, coords=final_frac)
    Poscar(final).write_file(dirpath / "CONTCAR")


def _write_sd_poscar(dirpath: Path, a: float = 10.0) -> None:
    """Write a 2-atom POSCAR with selective dynamics.

    Atom 0 (Li) is free, atom 1 (Na) is fully fixed.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        lattice=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        species=["Li", "Na"],
        coords=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
        site_properties={
            "selective_dynamics": [[True, True, True], [False, False, False]],
        },
    )
    Poscar(structure).write_file(dirpath / "POSCAR")


class _FakeOutcar:
    """Minimal stand-in for pymatgen's Outcar exposing ``final_forces``."""

    def __init__(self, forces: list[list[float]]) -> None:
        self.final_forces = np.asarray(forces)


def _write_neb_tree(dirpath: Path, nimages: int = 1) -> None:
    """Write an INCAR with ``IMAGES`` and empty ``00``..``NN`` image dirs.

    ``nimages`` is the number of intermediate images, so ``IMAGES=n``
    yields ``n + 2`` directories named ``00`` through ``NN``.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "INCAR").write_text(f"IMAGES = {nimages}\n")
    for n in range(nimages + 2):
        (dirpath / f"{n:02d}").mkdir(parents=True, exist_ok=True)


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


def test_initial_structure_from_poscar(tmp_path) -> None:
    """initial_structure reads POSCAR when no previous run exists."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.initial_structure.lattice.a == pytest.approx(5.0)


def test_initial_structure_none_without_structure(tmp_path) -> None:
    """initial_structure is None when no POSCAR/vasprun is present."""
    d = IMDGVaspDir(str(tmp_path))
    assert d.initial_structure is None


def test_initial_structure_follows_prev_dirs(tmp_path) -> None:
    """initial_structure walks gorun_* back to the earliest structure."""
    _write_poscar(tmp_path, a=5.0)  # current run
    _write_poscar(tmp_path / "gorun_1", a=4.0)  # previous run
    _write_poscar(tmp_path / "gorun_1" / "gorun_0", a=3.0)  # earliest

    d = IMDGVaspDir(str(tmp_path))
    assert d.initial_structure.lattice.a == pytest.approx(3.0)


def test_initial_structure_ignores_prev_dir_without_poscar(tmp_path) -> None:
    """A gorun_* dir without POSCAR is not treated as a previous run."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "gorun_1").mkdir()  # no POSCAR -> excluded

    d = IMDGVaspDir(str(tmp_path))
    assert d.initial_structure.lattice.a == pytest.approx(5.0)


def test_prev_dirs_requires_poscar_and_sorts(tmp_path) -> None:
    """prev_dirs lists only gorun_* dirs with a POSCAR, sorted by path."""
    _write_poscar(tmp_path / "gorun_1", a=4.0)
    (tmp_path / "gorun_2").mkdir()  # no POSCAR -> excluded
    _write_poscar(tmp_path / "gorun_0", a=3.0)

    d = IMDGVaspDir(str(tmp_path))
    prevs = d.prev_dirs()
    assert [Path(p.path).name for p in prevs] == ["gorun_0", "gorun_1"]


def test_prev_dirs_returns_empty_list(tmp_path) -> None:
    """prev_dirs always returns a list, empty when no gorun_* dirs exist."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.prev_dirs() == []


def test_getitem_returns_parsed_file(tmp_path) -> None:
    """__getitem__ parses a known file name into its parser object."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    poscar = d["POSCAR"]
    assert isinstance(poscar, Poscar)
    assert poscar.structure.lattice.a == pytest.approx(5.0)


def test_getitem_missing_file_returns_none(tmp_path) -> None:
    """__getitem__ returns None for a file name that does not exist."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d["does_not_exist"] is None


def test_getitem_unknown_present_file_raises(tmp_path) -> None:
    """__getitem__ raises RuntimeError for a present, unmapped file."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "README.txt").write_text("not a VASP file\n")
    d = IMDGVaspDir(str(tmp_path))
    with pytest.raises(RuntimeError):
        d["README.txt"]


def test_getitem_unparseable_file_returns_none(tmp_path) -> None:
    """__getitem__ returns None (and caches None) for a broken file."""
    (tmp_path / "POSCAR").write_text("garbage\nnot a POSCAR\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d["POSCAR"] is None
    assert d["POSCAR"] is None  # cached, not re-raised


@pytest.mark.skipif(not hasattr(signal, "alarm"), reason="SIGALRM not available")
def test_getitem_unparseable_disarms_alarm(tmp_path) -> None:
    """A failed parse leaves no SIGALRM pending.

    Regression: the failure path used to skip ``signal.alarm(0)``,
    leaving a 2-minute timeout armed that could fire later as a stray
    ``TimeoutException``.
    """
    (tmp_path / "POSCAR").write_text("garbage\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d["POSCAR"] is None
    assert signal.alarm(0) == 0  # no alarm left armed


def test_len_and_iter_exclude_default(tmp_path) -> None:
    """len/iter expose file names and skip the default EXCLUDE_PATTERNS."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    (tmp_path / "imdg.log").write_text("log\n")  # class default exclusion
    d = IMDGVaspDir(str(tmp_path))
    assert set(d) == {"POSCAR", "INCAR"}
    assert len(d) == 2


def test_len_and_iter_extra_exclude(tmp_path) -> None:
    """Constructor exclude_patterns extend the default exclusions."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "custom.log").write_text("log\n")
    d = IMDGVaspDir(str(tmp_path), exclude_patterns=["*.log"])
    assert set(d) == {"POSCAR"}
    assert len(d) == 1


def test_logs_empty(tmp_path) -> None:
    """logs() is empty when no VASP log files are present."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.logs() == []


def test_logs_discovers_slurm(tmp_path) -> None:
    """logs() parses slurm output into a Vasplog."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "slurm.out").write_text("some slurm output\n")
    d = IMDGVaspDir(str(tmp_path))
    logs = d.logs()
    assert [log.file.name for log in logs] == ["slurm.out"]


# --- structure / energy family -------------------------------------


def test_structure_prefers_contcar(tmp_path) -> None:
    """structure returns CONTCAR when present, over POSCAR/vasprun."""
    _write_poscar(tmp_path, a=5.0)
    _write_poscar(tmp_path, a=6.0, filename="CONTCAR")
    d = IMDGVaspDir(str(tmp_path))
    assert d.structure.lattice.a == pytest.approx(6.0)


def test_structure_vasprun_fallback(converged_vasp_dir) -> None:
    """structure falls back to vasprun final_structure without CONTCAR."""
    (converged_vasp_dir / "CONTCAR").unlink()
    d = IMDGVaspDir(str(converged_vasp_dir))
    structure = d.structure
    assert isinstance(structure, Structure)
    assert len(structure) == 1


def test_structure_none_without_contcar_or_vasprun(tmp_path) -> None:
    """structure is None when neither CONTCAR nor vasprun is present."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.structure is None


def test_final_energy_reliable_scf_float(converged_vasp_dir) -> None:
    """final_energy_reliable is a float for a converged SCF run (NSW=0)."""
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert isinstance(d.final_energy_reliable, float)


def test_final_energy_reliable_relax_unreliable(converged_relax_vasp_dir) -> None:
    """final_energy_reliable flags ISIF=3 ionic relaxation as unreliable."""
    d = IMDGVaspDir(str(converged_relax_vasp_dir))
    assert d.final_energy_reliable == "unreliable"


def test_final_energy_reliable_unconverged(converged_vasp_dir) -> None:
    """final_energy_reliable returns 'unconverged' when not converged."""
    (converged_vasp_dir / "UNCONVERGED").write_text("")
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert d.final_energy_reliable == "unconverged"


def test_total_magnetization(converged_vasp_dir) -> None:
    """total_magnetization reads the last OSZICAR step's mag."""
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert isinstance(d.total_magnetization, float)


def test_total_magnetization_none_without_oszicar(tmp_path) -> None:
    """total_magnetization is None without an OSZICAR."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.total_magnetization is None


# --- convergence decomposition -------------------------------------


def test_converged_electronic_outcar_fallback(converged_vasp_dir) -> None:
    """converged_electronic reads OUTCAR when vasprun.xml is absent."""
    (converged_vasp_dir / "vasprun.xml").unlink()
    d = IMDGVaspDir(str(converged_vasp_dir))
    assert d.converged_electronic is True


def test_converged_sequence_false_with_pending_incar(tmp_path) -> None:
    """converged_sequence is False when INCAR.[0-9]+ files remain."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "INCAR.1").write_text("ENCUT = 500\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d.converged_sequence is False


def test_converged_sequence_true(tmp_path) -> None:
    """converged_sequence is True when no INCAR.[0-9]+ files remain."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.converged_sequence is True


def test_converged_manual_false_with_marker(tmp_path) -> None:
    """converged_manual is False when an UNCONVERGED marker is present."""
    _write_poscar(tmp_path, a=5.0)
    (tmp_path / "UNCONVERGED").write_text("")
    d = IMDGVaspDir(str(tmp_path))
    assert d.converged_manual is False


def test_converged_manual_true(tmp_path) -> None:
    """converged_manual is True without an UNCONVERGED marker."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d.converged_manual is True


# --- checks and constrained forces ----------------------------------


def test_check_displacements_true(tmp_path) -> None:
    """check_displacements passes for an unchanged structure."""
    _write_grid_poscars(tmp_path)
    d = IMDGVaspDir(str(tmp_path))
    assert d.check_displacements() is True


def test_check_displacements_false_warns(tmp_path) -> None:
    """check_displacements warns and fails for a large displacement."""
    _write_grid_poscars(tmp_path, displace=(0, (0.5, 0.5, 0.5)))
    d = IMDGVaspDir(str(tmp_path))
    with pytest.warns(VaspWarning):
        assert d.check_displacements() is False


def test_selective_dynamics_none(tmp_path) -> None:
    """_selective_dynamics is None without selective dynamics."""
    _write_poscar(tmp_path, a=5.0)
    d = IMDGVaspDir(str(tmp_path))
    assert d._selective_dynamics() is None


def test_selective_dynamics_from_poscar(tmp_path) -> None:
    """_selective_dynamics reads flags from a POSCAR with constraints."""
    _write_sd_poscar(tmp_path)
    d = IMDGVaspDir(str(tmp_path))
    sd = d._selective_dynamics()
    assert sd is not None
    assert np.array_equal(sd, [[True, True, True], [False, False, False]])


def test_max_force_constrained(tmp_path) -> None:
    """max_force ignores fixed atoms unless include_constrained=True."""
    _write_sd_poscar(tmp_path)
    d = IMDGVaspDir(str(tmp_path))
    d.refresh()
    d._parsed_files["OUTCAR"] = _FakeOutcar([[1.0, 0.0, 0.0],
                                             [5.0, 0.0, 0.0]])
    assert d.max_force() == pytest.approx(1.0)
    assert d.max_force(include_constrained=True) == pytest.approx(5.0)


def test_max_force_all_constrained_returns_none(tmp_path) -> None:
    """max_force is None when every atom is fixed."""
    structure = Structure(
        lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Li", "Na"],
        coords=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
        site_properties={
            "selective_dynamics": [[False, False, False], [False, False, False]],
        },
    )
    Poscar(structure).write_file(tmp_path / "POSCAR")
    d = IMDGVaspDir(str(tmp_path))
    d.refresh()
    d._parsed_files["OUTCAR"] = _FakeOutcar([[5.0, 0.0, 0.0],
                                             [5.0, 0.0, 0.0]])
    assert d.max_force() is None


def test_check_framework_symmetry_true(tmp_path) -> None:
    """check_framework_symmetry passes for an unchanged framework."""
    _write_poscar(tmp_path, a=10.0)
    _write_poscar(tmp_path, a=10.0, filename="CONTCAR")
    d = IMDGVaspDir(str(tmp_path))
    assert d.check_framework_symmetry() is True


def test_check_framework_symmetry_false_warns(tmp_path) -> None:
    """check_framework_symmetry warns and fails when the lattice changes.

    A cubic -> tetragonal distortion of a single Li atom changes the
    space group (Pm-3m -> P4/mmm) with zero site displacement, which
    the check treats as suspicious (rms ~ 0).
    """
    init = Structure(
        lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Li"], coords=[[0.0, 0.0, 0.0]])
    final = Structure(
        lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]],
        species=["Li"], coords=[[0.0, 0.0, 0.0]])
    Poscar(init).write_file(tmp_path / "POSCAR")
    Poscar(final).write_file(tmp_path / "CONTCAR")
    d = IMDGVaspDir(str(tmp_path))
    with pytest.warns(VaspWarning):
        assert d.check_framework_symmetry() is False


# --- NEB ------------------------------------------------------------


def test_nebp_false(tmp_path) -> None:
    """nebp is False when INCAR has no IMAGES tag."""
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d.nebp is False
    assert d.neb_dirs() is None


def test_nebp_true(tmp_path) -> None:
    """nebp is True when INCAR has an IMAGES tag."""
    (tmp_path / "INCAR").write_text("IMAGES = 1\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d.nebp is True


def test_neb_dirs_includes_ends(tmp_path) -> None:
    """neb_dirs lists all image dirs (ends included) by default."""
    _write_neb_tree(tmp_path, nimages=1)
    d = IMDGVaspDir(str(tmp_path))
    dirs = d.neb_dirs()
    assert dirs is not None
    assert [Path(p.path).name for p in dirs] == ["00", "01", "02"]


def test_neb_dirs_excludes_ends(tmp_path) -> None:
    """neb_dirs(include_ends=False) drops the first and last images."""
    _write_neb_tree(tmp_path, nimages=1)
    d = IMDGVaspDir(str(tmp_path))
    dirs = d.neb_dirs(include_ends=False)
    assert dirs is not None
    assert [Path(p.path).name for p in dirs] == ["01"]


def test_mtime_detects_file_change(tmp_path) -> None:
    """mtime() reflects a change to a file that matters in the directory."""
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    d = IMDGVaspDir(str(tmp_path))
    before = d.mtime()
    os.utime(tmp_path / "INCAR", (before + 10.0, before + 10.0))
    assert d.mtime() > before


def test_converged_neb_all_images_converged(
        converged_vasp_dir, tmp_path_factory) -> None:
    """converged is True when all interior NEB images are converged."""
    root = tmp_path_factory.mktemp("neb")
    (root / "INCAR").write_text("IMAGES = 1\n")
    for n in (0, 2):  # ends; the interior image is the converged fixture
        (root / f"{n:02d}").mkdir()
    shutil.copytree(converged_vasp_dir, root / "01")
    d = IMDGVaspDir(str(root))
    assert d.nebp is True
    assert d.converged is True


def test_converged_neb_unconverged_image(
        converged_vasp_dir, tmp_path_factory) -> None:
    """converged is False when any interior NEB image is not converged."""
    root = tmp_path_factory.mktemp("neb")
    (root / "INCAR").write_text("IMAGES = 1\n")
    for n in (0, 2):
        (root / f"{n:02d}").mkdir()
    shutil.copytree(converged_vasp_dir, root / "01")
    (root / "01" / "UNCONVERGED").write_text("")
    d = IMDGVaspDir(str(root))
    assert d.converged is False


# --- Group G: timeout / SIGALRM path -----------------------------------------


def test_timeout_handler_raises() -> None:
    """timeout_handler always raises TimeoutException."""
    with pytest.raises(TimeoutException):
        timeout_handler(None, None)


def test_getitem_timeout_returns_none(tmp_path, monkeypatch) -> None:
    """A TimeoutException from parsing is swallowed into None (and cached)."""
    _write_poscar(tmp_path, a=5.0)

    def _timeout(*args, **kwargs):
        raise TimeoutException()

    monkeypatch.setattr(Poscar, "from_file", classmethod(_timeout))
    d = IMDGVaspDir(str(tmp_path))
    assert d["POSCAR"] is None
    assert d["POSCAR"] is None  # cached, not re-parsed


def test_getitem_arms_and_disarms_alarm(tmp_path, monkeypatch) -> None:
    """Parsing arms SIGALRM with TIMEOUT, then disarms it on success."""
    _write_poscar(tmp_path, a=5.0)
    calls: list[int] = []
    monkeypatch.setattr(signal, "alarm", lambda n: calls.append(n))
    d = IMDGVaspDir(str(tmp_path))
    d["POSCAR"]
    assert calls == [IMDGVaspDir.TIMEOUT, 0]


def test_getitem_timeout_disarms_alarm(tmp_path, monkeypatch) -> None:
    """A TimeoutException is swallowed and the alarm is disarmed."""
    _write_poscar(tmp_path, a=5.0)
    calls: list[int] = []

    def _timeout(*args, **kwargs):
        raise TimeoutException()

    monkeypatch.setattr(Poscar, "from_file", classmethod(_timeout))
    monkeypatch.setattr(signal, "alarm", lambda n: calls.append(n))
    d = IMDGVaspDir(str(tmp_path))
    assert d["POSCAR"] is None
    assert calls == [IMDGVaspDir.TIMEOUT, 0]


# --- Group F: directory / cache infrastructure --------------------------------


def test_read_vaspdirs_recurses(tmp_path) -> None:
    """read_vaspdirs finds VASP dirs recursively, skipping non-VASP dirs."""
    _write_poscar(tmp_path / "a", a=5.0)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "OSZICAR").write_text("")
    (tmp_path / "c").mkdir()  # no VASP files -> excluded
    _write_poscar(tmp_path / "d" / "nested", a=6.0)

    result = IMDGVaspDir.read_vaspdirs(tmp_path)
    names = {Path(p).relative_to(tmp_path).as_posix() for p in result}
    assert names == {"a", "b", "d/nested"}


def test_read_vaspdirs_path_filter(tmp_path) -> None:
    """read_vaspdirs applies path_filter to exclude directories."""
    _write_poscar(tmp_path / "keep", a=5.0)
    _write_poscar(tmp_path / "skip", a=6.0)

    result = IMDGVaspDir.read_vaspdirs(
        tmp_path, path_filter=lambda p: p.name == "keep")
    assert set(result) == {str(tmp_path / "keep")}


def test_read_vaspdirs_multiple_roots(tmp_path) -> None:
    """read_vaspdirs accepts a list of roots."""
    r1 = tmp_path / "r1"
    r2 = tmp_path / "r2"
    _write_poscar(r1, a=5.0)
    _write_poscar(r2, a=6.0)

    result = IMDGVaspDir.read_vaspdirs([r1, r2])
    assert set(result) == {str(r1), str(r2)}


def test_refresh_reparses_after_change(tmp_path) -> None:
    """refresh() re-parses files whose mtime (hash) changed."""
    incar_path = tmp_path / "INCAR"
    incar_path.write_text("ENCUT = 500\n")
    d = IMDGVaspDir(str(tmp_path))
    assert d["INCAR"]["ENCUT"] == 500

    incar_path.write_text("ENCUT = 600\n")
    new_mtime = incar_path.stat().st_mtime + 1.0
    os.utime(incar_path, (new_mtime, new_mtime))
    d.refresh()
    assert d["INCAR"]["ENCUT"] == 600


def test_mtime_considers_prev_dirs(tmp_path) -> None:
    """mtime() reflects files in previous gorun_* directories."""
    _write_poscar(tmp_path, a=5.0)
    _write_poscar(tmp_path / "gorun_1", a=4.0)
    prev_incar = tmp_path / "gorun_1" / "INCAR"
    prev_incar.write_text("ENCUT = 500\n")
    future = time.time() + 10000.0
    os.utime(prev_incar, (future, future))

    d = IMDGVaspDir(str(tmp_path))
    assert d.mtime() == pytest.approx(future)


def test_lmdb_cache_version_mismatch_discarded(
        converged_vasp_dir, isolated_lmdb_cache) -> None:
    """A cache entry from an older schema version is discarded on load."""
    first = IMDGVaspDir(str(converged_vasp_dir))
    assert first["INCAR"] is not None
    IMDGVaspDir.flush_cache()

    cached = IMDGVaspDir._lmdb_get(first._cache_key)
    assert cached is not None
    cached["version"] = -1
    IMDGVaspDir._lmdb_set(first._cache_key, cached)

    second = IMDGVaspDir(str(converged_vasp_dir))
    assert second["INCAR"] is not None  # re-parsed, not misread


def test_flush_cache_restores_on_failure(isolated_lmdb_cache, monkeypatch) -> None:
    """flush_cache restores pending writes when the LMDB write fails."""
    IMDGVaspDir._add_pending_write("key1", {"a": 1})

    def _fail(*args, **kwargs):
        return False

    monkeypatch.setattr(IMDGVaspDir, "_lmdb_set_many", _fail)
    IMDGVaspDir.flush_cache()

    with IMDGVaspDir._pending_lock:
        assert IMDGVaspDir._pending_writes["key1"] == {"a": 1}


def test_lmdb_evict_on_map_full(isolated_lmdb_cache, monkeypatch) -> None:
    """A shrunken map overflows and evicts the oldest entries.

    ``MAP_SIZE`` is shrunk far below its default so a handful of small
    entries fill the map.  Overflow then triggers the eviction path in
    ``_lmdb_set_many``, which reclaims space oldest-first and retries.
    """
    monkeypatch.setattr(IMDGVaspDir, "MAP_SIZE", 65536)

    # Seed two entries with distinct ctimes so "oldest" is unambiguous.
    for i in range(2):
        IMDGVaspDir._lmdb_set(f"old{i}", {"payload": "x" * 64})
        time.sleep(0.02)

    # Write enough entries to overflow the shrunken map.
    for i in range(40):
        IMDGVaspDir._lmdb_set(f"new{i}", {"payload": "x" * 64})

    # The oldest entry was evicted; the newest survived the retry.
    assert IMDGVaspDir._lmdb_get("old0") is None
    assert IMDGVaspDir._lmdb_get("new39") is not None
