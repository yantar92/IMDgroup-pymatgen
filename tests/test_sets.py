"""Tests for IMDgroup.pymatgen.io.vasp.sets.

Per the locked testing decision, the input sets are exercised with
``no_potcar=True`` plus the POTCAR-mapping helpers directly, rather
than fabricating POTCAR files (see tests/conftest.py).
"""

from __future__ import annotations

import warnings

import pytest
from pymatgen.util.testing import MatSciTest

from IMDgroup.pymatgen.io.vasp.sets import (
    IMDStandardVaspInputSet,
    IMDStandardVaspInputSet_relax,
    IMDStandardVaspInputSet_scf,
)


@pytest.fixture
def cscl():
    """Two-site CsCl structure from pymatgen's curated set."""
    return MatSciTest().get_structure("CsCl")


def _standard_set(structure, **kwargs) -> IMDStandardVaspInputSet:
    return IMDStandardVaspInputSet(structure=structure, no_potcar=True, **kwargs)


def test_write_input_omits_potcar(cscl, tmp_path) -> None:
    """no_potcar=True writes INCAR/KPOINTS/POSCAR but no POTCAR."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _standard_set(cscl).write_input(tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"INCAR", "KPOINTS", "POSCAR"} <= names
    assert "POTCAR" not in names


def test_standard_incar_defaults(cscl) -> None:
    """The standard set applies the group INCAR defaults."""
    incar = _standard_set(cscl).incar
    assert incar["ENCUT"] == 500.0
    assert incar["ALGO"] == "Normal"
    assert incar["SYSTEM"].startswith("CsCl")


def test_kpoints_grid_density(cscl) -> None:
    """KPOINTS are generated from the group-default grid density."""
    inset = _standard_set(cscl)
    assert inset.kpoints is not None
    assert inset._kpoints_grid_density() == 10000


def test_potcar_symbols_and_mapping(cscl) -> None:
    """potcar_symbols and the TOML mapping resolve element -> potential."""
    inset = _standard_set(cscl)
    assert inset.potcar_symbols == ["Cs_sv", "Cl"]
    assert inset._potcar_toml_mapping() == {"Cs": "Cs_sv", "Cl": "Cl"}


def test_functional_updates_default_is_empty(cscl) -> None:
    """Without a functional, no INCAR updates are produced."""
    assert _standard_set(cscl).incar_updates == {}


def test_functional_updates_pbe(cscl) -> None:
    """PBE contributes its GGA tag plus None removal markers."""
    updates = _standard_set(cscl, functional="pbe").incar_updates
    assert updates["GGA"] == "PE"
    assert updates["IVDW"] is None


def test_functional_is_case_insensitive(cscl) -> None:
    """The functional name is matched case-insensitively."""
    lower = _standard_set(cscl, functional="pbe").incar_updates
    upper = _standard_set(cscl, functional="PBE").incar_updates
    assert lower == upper


def test_functional_vdw_df(cscl) -> None:
    """A vdW functional contributes its full parameter block."""
    updates = _standard_set(cscl, functional="vdw-df").incar_updates
    assert updates["LUSE_VDW"] is True
    assert updates["GGA"] == "RE"


def test_relax_defaults(cscl) -> None:
    """The relax set overrides NSW/EDIFFG for geometry optimization."""
    incar = IMDStandardVaspInputSet_relax(
        structure=cscl, no_potcar=True).incar
    assert incar["NSW"] == 500
    assert incar["EDIFFG"] == -0.01


def test_scf_defaults(cscl) -> None:
    """The SCF set sets NSW=0 and the tetrahedron smearing method."""
    incar = IMDStandardVaspInputSet_scf(structure=cscl, no_potcar=True).incar
    assert incar["NSW"] == 0
    assert incar["ISMEAR"] == -5


def test_write_incar_toml_metadata_records_functional(cscl, tmp_path) -> None:
    """Non-default metadata (the functional) is written to INCAR.toml."""
    inset = _standard_set(cscl, functional="vdw-df")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset._write_incar_toml_metadata(tmp_path)
    text = (tmp_path / "INCAR.toml").read_text()
    assert 'functional = "vdw-df"' in text
