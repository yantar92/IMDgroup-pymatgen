"""Tests for IMDgroup.pymatgen.io.vasp.sets.

Per the locked testing decision, the input sets are exercised with
``no_potcar=True`` plus the POTCAR-mapping helpers directly, rather
than fabricating POTCAR files (see tests/conftest.py).
"""

from __future__ import annotations

import warnings

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints, Poscar
from pymatgen.io.vasp.sets import BadInputSetWarning
from pymatgen.util.testing import MatSciTest

from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.io.vasp.sets import (
    IMDDerivedInputSet,
    IMDNEBVaspInputSet,
    IMDRelaxCellulose,
    IMDStandardVaspInputSet,
    IMDStandardVaspInputSet_relax,
    IMDStandardVaspInputSet_scf,
    write_selective_dynamics_summary_maybe,
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


# ---------------------------------------------------------------------------
# write_selective_dynamics_summary_maybe
# ---------------------------------------------------------------------------


def _structure_with_sd(flags) -> Structure:
    """Two-site cubic structure with per-site ``selective_dynamics``.

    *flags* is a two-element sequence; each element is the site's
    ``selective_dynamics`` value (a 3-list of bools, or ``None``).
    """
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["Li", "Na"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    for site, flag in zip(structure, flags):
        site.properties["selective_dynamics"] = flag
    return structure


def test_selective_dynamics_no_constraints_writes_nothing(tmp_path) -> None:
    """Unconstrained sites produce no CIF and a False return value."""
    fname = tmp_path / "summary.cif"
    assert write_selective_dynamics_summary_maybe(
        _structure_with_sd([[True, True, True], [True, True, True]]),
        fname,
    ) is False
    assert not fname.exists()


def test_selective_dynamics_fully_fixed_writes_cif(tmp_path) -> None:
    """A fully fixed site is visualized as Fe and reported as fixed."""
    fname = tmp_path / "summary.cif"
    assert write_selective_dynamics_summary_maybe(
        _structure_with_sd([[False, False, False], [True, True, True]]),
        fname,
    ) is True
    read_back = Structure.from_file(fname)
    assert [site.specie.symbol for site in read_back] == ["Fe", "Ni"]


def test_selective_dynamics_partially_fixed(tmp_path) -> None:
    """A partially fixed site is visualized as Co."""
    fname = tmp_path / "summary.cif"
    write_selective_dynamics_summary_maybe(
        _structure_with_sd([[True, False, True], [True, True, True]]),
        fname,
    )
    read_back = Structure.from_file(fname)
    assert [site.specie.symbol for site in read_back] == ["Co", "Ni"]


def test_selective_dynamics_unknown_alone_writes_nothing(tmp_path) -> None:
    """Sites with no selective dynamics are unknown, not fixed: with no
    fixed site present, nothing is written."""
    fname = tmp_path / "summary.cif"
    assert write_selective_dynamics_summary_maybe(
        _structure_with_sd([None, [True, True, True]]),
        fname,
    ) is False
    assert not fname.exists()


def test_selective_dynamics_unknown_with_fixed_visualized(tmp_path) -> None:
    """An explicit None site is visualized as X alongside a fixed site."""
    fname = tmp_path / "summary.cif"
    with warnings.catch_warnings():
        # CIF round-trip warns that 'X' is not a valid element symbol.
        warnings.simplefilter("ignore")
        write_selective_dynamics_summary_maybe(
            _structure_with_sd([[False, False, False], None]),
            fname,
        )
        read_back = Structure.from_file(fname)
    assert sorted(site.specie.symbol for site in read_back) == ["Fe", "X"]


def test_selective_dynamics_missing_key_visualized_as_x(tmp_path) -> None:
    """A site lacking the selective_dynamics key is marked X, not free.

    This is the mixed-assignment case that arises when the function is
    called independently on a partially-constrained structure.
    """
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["Li", "Na"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    structure[0].properties["selective_dynamics"] = [False, False, False]
    fname = tmp_path / "summary.cif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 'X' is not a valid element symbol
        assert write_selective_dynamics_summary_maybe(structure, fname) is True
        read_back = Structure.from_file(fname)
    assert sorted(site.specie.symbol for site in read_back) == ["Fe", "X"]


# ---------------------------------------------------------------------------
# kpoints
# ---------------------------------------------------------------------------


def test_kpoints_no_kpoints_returns_none(cscl) -> None:
    """no_kpoints=True suppresses KPOINTS generation entirely."""
    assert _standard_set(cscl, no_kpoints=True).kpoints is None


# ---------------------------------------------------------------------------
# incar warnings and SYSTEM naming
# ---------------------------------------------------------------------------


def test_incar_low_encut_warns(cscl) -> None:
    """An ENCUT below the 500 eV default emits a warning."""
    inset = IMDStandardVaspInputSet(
        structure=cscl, no_potcar=True, user_incar_settings={"ENCUT": 300.0})
    with pytest.warns(BadInputSetWarning, match="ENCUT"):
        inset.incar


def test_incar_isif_relaxation_low_encut_warns(cscl) -> None:
    """Volume/shape relaxation (ISIF) with ENCUT below 550 eV warns."""
    inset = IMDStandardVaspInputSet(
        structure=cscl,
        no_potcar=True,
        user_incar_settings={"ENCUT": 500.0, "ISIF": 3, "NSW": 10},
    )
    with pytest.warns(BadInputSetWarning, match="volume/shape"):
        inset.incar


def test_incar_ncore_and_npar_warns(cscl) -> None:
    """Setting both NCORE and NPAR is contradictory and warns."""
    inset = IMDStandardVaspInputSet(
        structure=cscl,
        no_potcar=True,
        user_incar_settings={"NCORE": 4, "NPAR": 4},
    )
    with pytest.warns(BadInputSetWarning, match="NCORE"):
        inset.incar


def test_incar_system_name_includes_mpid(cscl) -> None:
    """The mpid structure property is folded into the SYSTEM name."""
    structure = cscl.copy()
    structure.properties["mpid"] = "mp-1234"
    incar = _standard_set(structure).incar
    assert incar["SYSTEM"] == "CsCl.mp-1234.cubic.221"


# ---------------------------------------------------------------------------
# potcar
# ---------------------------------------------------------------------------


def test_potcar_no_potcar_returns_none(cscl) -> None:
    """no_potcar=True returns None for the potcar property."""
    assert _standard_set(cscl).potcar is None


# ---------------------------------------------------------------------------
# _kpoints_grid_density
# ---------------------------------------------------------------------------


def test_kpoints_grid_density_user_dict(cscl) -> None:
    """A user-supplied density recipe is reported verbatim."""
    inset = _standard_set(cscl, user_kpoints_settings={"grid_density": 5000})
    assert inset._kpoints_grid_density() == 5000


def test_kpoints_grid_density_user_kpoints_returns_none(cscl) -> None:
    """An explicit Kpoints object has no single density to report."""
    inset = _standard_set(
        cscl, user_kpoints_settings=Kpoints.gamma_automatic((4, 4, 4)))
    assert inset._kpoints_grid_density() is None


def test_kpoints_grid_density_updates_dict(cscl) -> None:
    """A grid density coming from kpoints_updates is reported."""

    class _UpdatesSet(IMDStandardVaspInputSet):
        @property
        def kpoints_updates(self):
            return {"grid_density": 7000}

    assert _UpdatesSet(structure=cscl, no_potcar=True)._kpoints_grid_density()\
        == 7000


def test_kpoints_grid_density_updates_kpoints_returns_none(cscl) -> None:
    """A Kpoints object in kpoints_updates reports None."""

    class _UpdatesSet(IMDStandardVaspInputSet):
        @property
        def kpoints_updates(self):
            return Kpoints.gamma_automatic((4, 4, 4))

    assert _UpdatesSet(
        structure=cscl, no_potcar=True)._kpoints_grid_density() is None


# ---------------------------------------------------------------------------
# write_input extras
# ---------------------------------------------------------------------------


def test_write_input_log_file(cscl, tmp_path) -> None:
    """write_input records every constructor field in a log file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _standard_set(cscl).write_input(tmp_path)
    log = (tmp_path / "IMDVaspInputSet.log").read_text()
    assert "no_potcar: True" in log
    assert "functional:" in log


def test_write_input_no_poscar_removes_poscar(cscl, tmp_path) -> None:
    """no_poscar=True deletes the POSCAR pymatgen wrote."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _standard_set(cscl, no_poscar=True).write_input(tmp_path)
    assert "POSCAR" not in {p.name for p in tmp_path.iterdir()}


def test_write_input_no_incar_removes_incar(cscl, tmp_path) -> None:
    """no_incar=True leaves no INCAR file behind."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _standard_set(cscl, no_incar=True).write_input(tmp_path)
    assert "INCAR" not in {p.name for p in tmp_path.iterdir()}


def test_write_input_selective_dynamics_cif(cscl, tmp_path) -> None:
    """A constrained structure writes a selective_dynamics.cif summary."""
    structure = cscl.copy()
    structure[0].properties["selective_dynamics"] = [False, False, False]
    structure[1].properties["selective_dynamics"] = [True, True, True]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _standard_set(structure).write_input(tmp_path)
    assert (tmp_path / "selective_dynamics.cif").exists()


# ---------------------------------------------------------------------------
# IMDDerivedInputSet
# ---------------------------------------------------------------------------


def _write_poscar(dirpath, a=4.1) -> Structure:
    """Write a two-site CsCl POSCAR and return its structure."""
    dirpath.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        lattice=Lattice.cubic(a),
        species=["Cs", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    Poscar(structure).write_file(dirpath / "POSCAR")
    return structure


def test_derived_reads_structure_from_poscar(tmp_path) -> None:
    """A directory with only a POSCAR yields the parsed structure."""
    _write_poscar(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
    assert inset.structure.reduced_formula == "CsCl"


def test_derived_incar_none_when_forced_and_missing(tmp_path) -> None:
    """force_prev_incar_file discards the INCAR when the file is absent."""
    _write_poscar(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(
            directory=tmp_path,
            no_potcar=True,
            force_prev_incar_file=True,
        )
    assert inset.incar is None


def test_derived_kpoints_none_when_forced_and_missing(tmp_path) -> None:
    """force_prev_kpoints_file discards KPOINTS when the file is absent."""
    _write_poscar(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(
            directory=tmp_path,
            no_potcar=True,
            force_prev_kpoints_file=True,
        )
    assert inset.kpoints is None


def test_derived_kpoints_kspacing_returns_none(tmp_path) -> None:
    """A previous INCAR using KSPACING suppresses KPOINTS generation."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("KSPACING = 0.3\nENCUT = 500\n")
    Kpoints.gamma_automatic((4, 4, 4)).write_file(tmp_path / "KPOINTS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
    assert inset.kpoints is None


def test_derived_kpoints_updates_prefers_prev_kpoints(tmp_path) -> None:
    """kpoints_updates returns the inherited KPOINTS file as a Kpoints."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    Kpoints.gamma_automatic((4, 4, 4)).write_file(tmp_path / "KPOINTS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
    assert isinstance(inset.kpoints_updates, Kpoints)
    assert inset.kpoints_updates.kpts == [(4, 4, 4)]


def test_derived_concar_fallback_reads_structure(tmp_path) -> None:
    """A directory with only a CONTCAR yields the relaxed structure."""
    structure = Structure(
        lattice=Lattice.cubic(4.1),
        species=["Cs", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    Poscar(structure).write_file(tmp_path / "CONTCAR")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
    assert inset.structure.reduced_formula == "CsCl"


def test_derived_transfers_incar_variants(tmp_path) -> None:
    """INCAR.[0-9]* and INCAR.toml are always transferred; INCAR.py only
    when inherit_prev_incarpy is set."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR.1").write_text("ENCUT = 500\n")
    (tmp_path / "INCAR.py").write_text("print(1)\n")
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nCs = "Cs_sv"\n')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inherited = IMDDerivedInputSet(
            directory=tmp_path, no_potcar=True, inherit_prev_incarpy=True)
        default = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)

    assert set(inherited.files_to_transfer) == {
        "INCAR.1", "INCAR.py", "INCAR.toml"}
    assert "INCAR.py" not in default.files_to_transfer
    assert "INCAR.1" in default.files_to_transfer
    assert "INCAR.toml" in default.files_to_transfer


# ---------------------------------------------------------------------------
# IMDNEBVaspInputSet
# ---------------------------------------------------------------------------


def test_neb_incar_forces_ibrion_and_warns(monkeypatch) -> None:
    """IMAGES=0 warns; IBRION != 1 warns and is forced to 1."""
    neb = object.__new__(IMDNEBVaspInputSet)
    monkeypatch.setattr(
        IMDDerivedInputSet,
        "incar",
        property(lambda self: Incar({"IMAGES": 0, "IBRION": 2})),
    )
    with pytest.warns(BadInputSetWarning, match="IMAGES=0"):
        with pytest.warns(UserWarning, match="Forcing IBRION=1"):
            incar = IMDNEBVaspInputSet.incar.fget(neb)
    assert incar["IBRION"] == 1


def test_fix_atoms_maybe_fixes_distant_atoms() -> None:
    """Atoms far from a moving atom are pinned via selective dynamics."""
    lattice = Lattice.cubic(10.0)
    beg = Structure(lattice, ["Li", "Na"], [[0, 0, 0], [0.5, 0, 0]])
    end = Structure(lattice, ["Li", "Na"], [[0.1, 0, 0], [0.5, 0, 0]])
    images = [beg.copy(), end.copy()]

    neb = object.__new__(IMDNEBVaspInputSet)
    neb.fix_cutoff = 2.0
    neb._fix_atoms_maybe(images)

    for image in images:
        assert image[0].properties["selective_dynamics"] == [True, True, True]
        assert image[1].properties["selective_dynamics"] == [False, False, False]


def test_fix_atoms_maybe_no_cutoff_noop() -> None:
    """With fix_cutoff=None the images are left untouched."""
    lattice = Lattice.cubic(10.0)
    images = [
        Structure(lattice, ["Li"], [[0, 0, 0]]),
        Structure(lattice, ["Li"], [[0.1, 0, 0]]),
    ]

    neb = object.__new__(IMDNEBVaspInputSet)
    neb.fix_cutoff = None
    assert neb._fix_atoms_maybe(images) is None
    assert all("selective_dynamics" not in site.properties
               for image in images for site in image)
