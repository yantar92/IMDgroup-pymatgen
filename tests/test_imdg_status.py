"""Tests for IMDgroup.pymatgen.cli.imdg_status."""

from __future__ import annotations

from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar

from IMDgroup.pymatgen.cli.imdg_status import _get_neb_summary
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir


def _write_structure(dirpath: Path, x: float, filename: str = "POSCAR") -> None:
    """Write a single-atom Li structure file at fractional coordinate ``x``."""
    dirpath.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Li"],
        coords=[[x, 0.0, 0.0]],
    )
    Poscar(structure).write_file(dirpath / filename)


def test_get_neb_summary_end_images_without_contcar(tmp_path) -> None:
    """NEB summary falls back to POSCAR for fixed end images.

    End images (00 and NN) are fixed endpoints that never produce a
    CONTCAR/vasprun.xml.  ``_get_neb_summary`` must not raise when
    only their POSCAR is present.
    """
    (tmp_path / "INCAR").write_text("IMAGES = 1\n")
    _write_structure(tmp_path / "00", x=0.0)
    _write_structure(tmp_path / "01", x=0.1)
    _write_structure(tmp_path / "01", x=0.15, filename="CONTCAR")
    _write_structure(tmp_path / "02", x=0.2)

    vaspdir = IMDGVaspDir(str(tmp_path))
    summary = _get_neb_summary(vaspdir)

    assert "IMAGE DISTANCES (initial)" in summary
    assert "IMAGE DISTANCES" in summary
