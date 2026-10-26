"""Tests for IMDgroup.pymatgen.io.vasp.sets.

Per the locked testing decision, the input sets are exercised with
``no_potcar=True`` plus the POTCAR-mapping helpers directly, rather
than fabricating POTCAR files (see tests/conftest.py).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints, PmgVaspPspDirError, Poscar
from pymatgen.io.vasp.sets import BadInputSetWarning
from pymatgen.util.testing import MatSciTest

from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.io.vasp.sets import (
    IMDDerivedInputSet,
    IMDNEBVaspInputSet,
    IMDStandardVaspInputSet,
    IMDVaspInputSet,
    POTCAR_RECOMMENDED,
    write_selective_dynamics_summary_maybe,
)
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir


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


def test_kpoints_density_low_warns(cscl) -> None:
    """A mesh below ~5000 k-points/atom warns.

    ``grid_density`` is a target k-points-per-atom; the check runs on
    the discretized mesh actually generated (``prod(divisions) * n_atoms``).
    A target of 1000 yields ~686 k-points/atom for CsCl, unambiguously
    below the 5000 floor.
    """
    inset = _standard_set(cscl, user_kpoints_settings={"grid_density": 1000})
    with pytest.warns(BadInputSetWarning, match="lower than 5000"):
        inset.kpoints


def test_kpoints_density_high_warns(cscl) -> None:
    """A mesh above ~15000 k-points/atom warns."""
    inset = _standard_set(cscl, user_kpoints_settings={"grid_density": 50000})
    with pytest.warns(BadInputSetWarning, match="higher than 15000"):
        inset.kpoints


def test_kpoints_density_default_no_warning(cscl) -> None:
    """The default 10000 density sits inside the accepted window."""
    inset = _standard_set(cscl)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BadInputSetWarning)
        assert inset.kpoints is not None


def test_kpoints_density_gamma_only_no_warning(cscl) -> None:
    """A 1x1x1 mesh is exempt from the density check."""
    inset = _standard_set(
        cscl, user_kpoints_settings=Kpoints.gamma_automatic((1, 1, 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", BadInputSetWarning)
        assert inset.kpoints is not None


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


def test_encut_reference_fallback_without_potcar(cscl) -> None:
    """Without a POTCAR, the group 500/550 eV references are used."""
    inset = IMDStandardVaspInputSet(structure=cscl, no_potcar=True)
    assert inset._encut_reference() == (500.0, 550.0)


def test_incar_encut_warning_uses_potcar_threshold(cscl, monkeypatch) -> None:
    """ENCUT is compared against the POTCAR-derived ENMAX reference."""
    fake_potcar = [SimpleNamespace(enmax=262.472)]
    monkeypatch.setattr(IMDVaspInputSet, "potcar", fake_potcar)
    inset = IMDStandardVaspInputSet(
        structure=cscl,
        no_potcar=True,
        user_incar_settings={"ENCUT": 300.0, "ISIF": 3, "NSW": 10},
    )
    # base reference = 262.472, relax reference = 341.214.  ENCUT=300 is
    # above the base but below the relaxation reference for ISIF=3.
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


def test_write_input_psp_dir_not_configured_raises_friendly_error(
        monkeypatch, cscl, tmp_path) -> None:
    """A missing PMG_VASP_PSP_DIR setting keeps the type and adds a hint."""
    def _raise(self, *args, **kwargs):
        raise PmgVaspPspDirError("Set PMG_VASP_PSP_DIR=<directory-path> ...")

    monkeypatch.setattr(
        "pymatgen.io.vasp.sets.VaspInputSet.write_input", _raise)
    inset = IMDStandardVaspInputSet(structure=cscl)
    with pytest.raises(
            PmgVaspPspDirError, match="Pseudopotential directory not configured"):
        inset.write_input(tmp_path)


def test_write_input_psp_dir_missing_raises_friendly_error(
        monkeypatch, cscl, tmp_path) -> None:
    """A configured-but-missing PSP dir keeps FileNotFoundError with a hint."""
    def _raise(self, *args, **kwargs):
        raise FileNotFoundError("PMG_VASP_PSP_DIR='/nonexistent' does not exist.")

    monkeypatch.setattr(
        "pymatgen.io.vasp.sets.VaspInputSet.write_input", _raise)
    inset = IMDStandardVaspInputSet(structure=cscl)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        inset.write_input(tmp_path)


def test_write_input_wrong_potcar_reraises_file_not_found(
        monkeypatch, cscl, tmp_path) -> None:
    """A missing POTCAR inside a valid dir is left as FileNotFoundError."""
    def _raise(self, *args, **kwargs):
        raise FileNotFoundError(
            "You do not have the right POTCAR with functional='PBE' ... "
            "in your PMG_VASP_PSP_DIR='/valid'."
        )

    monkeypatch.setattr(
        "pymatgen.io.vasp.sets.VaspInputSet.write_input", _raise)
    inset = IMDStandardVaspInputSet(structure=cscl)
    with pytest.raises(FileNotFoundError):
        inset.write_input(tmp_path)


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


def _neb_image(shift: float) -> IMDVaspInputSet:
    """A CsCl NEB image with its second site displaced by *shift*."""
    structure = Structure(
        lattice=Lattice.cubic(4.1),
        species=["Cs", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5 + shift, 0.5, 0.5]],
    )
    image = IMDVaspInputSet(no_kpoints=True, no_potcar=True, no_incar=True)
    image.structure = structure
    return image


def test_write_input_neb_writes_image_subdirs_and_trajectory(tmp_path) -> None:
    """The NEB branch writes one subdir per image plus a trajectory CIF."""
    structure = Structure(
        lattice=Lattice.cubic(4.1),
        species=["Cs", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    inset = IMDStandardVaspInputSet(
        structure=structure, no_potcar=True,
        user_incar_settings={"IMAGES": 1})
    inset.images = [_neb_image(shift) for shift in (0.0, 0.05, 0.10)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset.write_input(tmp_path)

    for name in ("00", "01", "02"):
        subdir = tmp_path / name
        assert subdir.is_dir()
        assert (subdir / "POSCAR").exists()
    assert (tmp_path / "NEB_trajectory.cif").exists()


def test_write_input_neb_image_without_structure_raises(tmp_path) -> None:
    """An image lacking a structure aborts the trajectory write."""
    structure = Structure(
        lattice=Lattice.cubic(4.1),
        species=["Cs", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    image = IMDVaspInputSet(no_kpoints=True, no_potcar=True, no_incar=True)
    inset = IMDStandardVaspInputSet(structure=structure, no_potcar=True)
    inset.images = [image]

    with pytest.raises(ValueError, match="Cannot build NEB trajectory"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inset.write_input(tmp_path)


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


def _write_neb_parent(dirpath, *, with_kpoints: bool = True):
    """Create a NEB parent dir (INCAR with IMAGES) and return its path.

    The ``IMAGES`` tag makes ``IMDGVaspDir(parent).nebp`` report True,
    which is what the parent-dir INCAR/KPOINTS fallback keys off.
    """
    parent = dirpath / "neb_parent"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "INCAR").write_text("IMAGES = 1\nENCUT = 500\n")
    if with_kpoints:
        Kpoints.gamma_automatic((4, 4, 4)).write_file(parent / "KPOINTS")
    return parent


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


def test_derived_accepts_vaspdir_object(tmp_path) -> None:
    """An IMDGVaspDir object is accepted and coerced to its path string."""
    _write_poscar(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(
            directory=IMDGVaspDir(tmp_path), no_potcar=True)
    assert inset.directory == str(tmp_path)
    assert inset.structure.reduced_formula == "CsCl"


def test_derived_prev_incar_from_file(tmp_path) -> None:
    """A previous INCAR file sets prev_incar, preserving params vasprun.xml
    omits (e.g. NCORE is not stored in vasprun.xml)."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\nNCORE = 4\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
    assert inset.prev_incar is not None
    assert inset.prev_incar["ENCUT"] == 500.0
    assert inset.prev_incar["NCORE"] == 4


def test_derived_prev_kpoints_from_file(tmp_path) -> None:
    """A KPOINTS file in the directory becomes prev_kpoints."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    Kpoints.gamma_automatic((4, 4, 4)).write_file(tmp_path / "KPOINTS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
    assert isinstance(inset.prev_kpoints, Kpoints)
    assert inset.prev_kpoints.kpts == [(4, 4, 4)]


def test_derived_prev_incar_parent_neb_fallback(tmp_path) -> None:
    """A NEB parent's INCAR is inherited when the image dir has none."""
    parent = _write_neb_parent(tmp_path, with_kpoints=False)
    _write_poscar(parent / "00")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=parent / "00", no_potcar=True)
    assert inset.prev_incar is not None
    assert inset.prev_incar["IMAGES"] == 1
    assert inset.prev_incar["ENCUT"] == 500.0


def test_derived_prev_kpoints_parent_neb_fallback(tmp_path) -> None:
    """A NEB parent's KPOINTS is inherited when the image dir has none."""
    parent = _write_neb_parent(tmp_path)
    _write_poscar(parent / "00")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=parent / "00", no_potcar=True)
    assert isinstance(inset.prev_kpoints, Kpoints)
    assert inset.prev_kpoints.kpts == [(4, 4, 4)]


def test_derived_incar_reflects_file(tmp_path) -> None:
    """derived.incar returns the base INCAR built on the inherited file."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\nNCORE = 4\n")
    Kpoints.gamma_automatic((4, 4, 4)).write_file(tmp_path / "KPOINTS")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)
        incar = inset.incar
    assert incar["ENCUT"] == 500.0
    assert incar["NCORE"] == 4


def test_derived_no_input_raises(tmp_path) -> None:
    """An empty directory with no POSCAR/CONTCAR raises ValueError."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No VASP input found"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IMDDerivedInputSet(directory=tmp_path, no_potcar=True)


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


def test_neb_post_init_from_vaspruns(converged_relax_vasp_dir) -> None:
    """Two consistent converged runs set target_structure without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", BadInputSetWarning)
        neb = IMDNEBVaspInputSet(
            directory=converged_relax_vasp_dir,
            target_directory=converged_relax_vasp_dir,
            no_potcar=True,
        )
    assert neb.target_structure.reduced_formula == "Na"
    # Consistent INCARs do not trigger image regeneration.
    assert neb.images is None


def test_neb_post_init_target_missing_vasprun_falls_back_to_poscar(
        converged_relax_vasp_dir, tmp_path) -> None:
    """A target dir without vasprun.xml falls back to its POSCAR."""
    _write_poscar(tmp_path / "target")
    with pytest.warns(UserWarning, match="Falling back to reading POSCAR"):
        neb = IMDNEBVaspInputSet(
            directory=converged_relax_vasp_dir,
            target_directory=tmp_path / "target",
            no_potcar=True,
        )
    assert neb.target_structure.reduced_formula == "CsCl"


def test_neb_post_init_incar_inconsistent_warns(
        converged_relax_vasp_dir, tmp_path, monkeypatch) -> None:
    """Differing INCARs warn and trigger image regeneration.

    A target dir is built from the same converged run with its SYSTEM tag
    altered, so the two vasprun.xml files differ only in that parameter.
    """
    target = tmp_path / "target"
    target.mkdir()
    xml = (converged_relax_vasp_dir / "vasprun.xml").read_text()
    xml = xml.replace(
        '<i type="string" name="SYSTEM">Na</i>',
        '<i type="string" name="SYSTEM">Na_target</i>',
    )
    (target / "vasprun.xml").write_text(xml)

    calls: list[int] = []
    monkeypatch.setattr(
        IMDNEBVaspInputSet,
        "update_images",
        lambda self, *args, **kwargs: calls.append(1),
    )
    with pytest.warns(BadInputSetWarning, match="inconsistent"):
        neb = IMDNEBVaspInputSet(
            directory=converged_relax_vasp_dir,
            target_directory=target,
            no_potcar=True,
        )
    assert calls == [1]
    assert neb.target_structure.reduced_formula == "Na"


def test_neb_write_input_writes_neb_inputs_txt(tmp_path, monkeypatch) -> None:
    """write_input records the begin/end source dirs in NEB-inputs.txt."""
    neb = object.__new__(IMDNEBVaspInputSet)
    neb.images = [_neb_image(0.0), _neb_image(0.1), _neb_image(0.2)]
    neb.directory = "/path/to/begin"
    neb.target_directory = "/path/to/end"
    monkeypatch.setattr(
        IMDVaspInputSet, "write_input", lambda self, *args, **kwargs: None)

    neb.write_input(tmp_path)

    text = (tmp_path / "NEB-inputs.txt").read_text()
    assert "NEB path computed between:\n" in text
    assert "00: /path/to/begin\n" in text
    assert "02: /path/to/end\n" in text


def _neb_endpoints():
    """A begin/end pair sharing a lattice, one atom displaced."""
    lattice = Lattice.cubic(10.0)
    beg = Structure(lattice, ["Li", "Na"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    end = Structure(lattice, ["Li", "Na"], [[0.2, 0, 0], [0.5, 0.5, 0.5]])
    return beg, end


def test_neb_update_images_idpp(tmp_path, monkeypatch) -> None:
    """IDPP interpolation builds len(IMAGES)+2 image input sets."""
    monkeypatch.chdir(tmp_path)  # idpp_interpolate writes idpp.log to CWD
    beg, end = _neb_endpoints()

    neb = object.__new__(IMDNEBVaspInputSet)
    neb.structure = beg
    neb.target_structure = end
    neb.method = "IDPP"
    neb.fix_cutoff = None
    neb.frac_tol = 0.5
    monkeypatch.setattr(
        IMDDerivedInputSet,
        "incar",
        property(lambda self: Incar({"IMAGES": 3, "IBRION": 1})),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neb.update_images()

    assert len(neb.images) == 5  # IMAGES(3) + 2 endpoints
    assert all(isinstance(img, IMDVaspInputSet) for img in neb.images)
    assert all(img.structure is not None for img in neb.images)


def test_neb_update_images_linear(monkeypatch) -> None:
    """Non-IDPP interpolation applies frac_tol validation and builds images."""
    beg, end = _neb_endpoints()

    neb = object.__new__(IMDNEBVaspInputSet)
    neb.structure = beg
    neb.target_structure = end
    neb.method = "linear"
    neb.fix_cutoff = None
    neb.frac_tol = 0.5
    monkeypatch.setattr(
        IMDDerivedInputSet,
        "incar",
        property(lambda self: Incar({"IMAGES": 3, "IBRION": 1})),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neb.update_images()

    assert len(neb.images) == 5
    assert all(img.structure is not None for img in neb.images)


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


def test_fix_atoms_maybe_preserves_existing_sd() -> None:
    """A moving atom that already has selective dynamics keeps them."""
    lattice = Lattice.cubic(10.0)
    beg = Structure(lattice, ["Li", "Na"], [[0, 0, 0], [0.5, 0, 0]])
    end = Structure(lattice, ["Li", "Na"], [[0.1, 0, 0], [0.5, 0, 0]])
    beg[0].properties["selective_dynamics"] = [True, True, True]
    end[0].properties["selective_dynamics"] = [True, True, True]

    neb = object.__new__(IMDNEBVaspInputSet)
    neb.fix_cutoff = 2.0
    neb._fix_atoms_maybe([beg, end])

    assert beg[0].properties["selective_dynamics"] == [True, True, True]
    assert beg[1].properties["selective_dynamics"] == [False, False, False]


def test_neb_update_images_explicit_endpoints(monkeypatch, tmp_path) -> None:
    """Explicit beg/end are used instead of self.structure/target_structure."""
    monkeypatch.chdir(tmp_path)  # idpp_interpolate writes idpp.log to CWD
    beg, end = _neb_endpoints()

    neb = object.__new__(IMDNEBVaspInputSet)
    neb.method = "IDPP"
    neb.fix_cutoff = None
    neb.frac_tol = 0.5
    monkeypatch.setattr(
        IMDDerivedInputSet,
        "incar",
        property(lambda self: Incar({"IMAGES": 3, "IBRION": 1})),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        neb.update_images(beg, end)

    assert neb.target_structure == end
    assert len(neb.images) == 5


def test_write_input_neb_no_incar_skips_image_dirs(tmp_path) -> None:
    """Without IMAGES in the INCAR, image subdirs are skipped with a warning."""
    structure = Structure(
        lattice=Lattice.cubic(4.1),
        species=["Cs", "Cl"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    inset = IMDStandardVaspInputSet(
        structure=structure, no_potcar=True, no_incar=True)
    inset.images = [_neb_image(0.0), _neb_image(0.1)]

    with pytest.warns(BadInputSetWarning, match="IMAGES"):
        inset.write_input(tmp_path)

    assert (tmp_path / "NEB_trajectory.cif").exists()
    assert not (tmp_path / "00").exists()


def test_incar_preserves_user_system_name(cscl) -> None:
    """A user-supplied SYSTEM tag is not overwritten by the auto name."""
    inset = IMDStandardVaspInputSet(
        structure=cscl, no_potcar=True,
        user_incar_settings={"SYSTEM": "my_system"},
    )
    assert inset.incar["SYSTEM"] == "my_system"


def test_kpoints_grid_density_fallthrough_none(cscl) -> None:
    """An input set with no KPOINTS recipe reports None density."""
    inset = IMDVaspInputSet(structure=cscl, no_potcar=True)
    assert inset._kpoints_grid_density() is None


def test_potcar_generation_when_not_suppressed(monkeypatch, cscl) -> None:
    """no_potcar=False delegates POTCAR generation to the parent class."""
    fake = object()
    monkeypatch.setattr(
        "pymatgen.io.vasp.sets.VaspInputSet.potcar",
        property(lambda self: fake),
    )
    inset = IMDStandardVaspInputSet(structure=cscl)
    assert inset.potcar is fake


def test_derived_neb_parent_populates_images(tmp_path) -> None:
    """A NEB parent dir (IMAGES in INCAR) yields one derived set per image."""
    parent = tmp_path / "neb"
    parent.mkdir()
    (parent / "INCAR").write_text("IMAGES = 1\nENCUT = 500\n")
    for name in ("00", "01", "02"):
        _write_poscar(parent / name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=parent, no_potcar=True)

    assert len(inset.images) == 3
    assert all(isinstance(img, IMDDerivedInputSet) for img in inset.images)
    assert all(img.structure.reduced_formula == "CsCl"
               for img in inset.images)


def test_derived_potcar_inheritance(monkeypatch, tmp_path) -> None:
    """A POTCAR* file overrides the resolved POTCAR mapping.

    The fake POTCAR uses arbitrary symbols (not the ASE-recommended
    ones) so the assertion depends only on the inheritance mechanism,
    not on the recommended defaults staying constant.
    """
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    (tmp_path / "POTCAR").write_text("placeholder")

    class _FakePotcar:
        symbols = ["Cs_fake", "Cl_fake"]

    monkeypatch.setattr(
        "pymatgen.io.vasp.inputs.Potcar.from_file",
        lambda path: _FakePotcar(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)

    assert inset._config_dict["POTCAR"] == {"Cs": "Cs_fake", "Cl": "Cl_fake"}


def test_derived_potcar_from_incar_toml_fallback(tmp_path) -> None:
    """Without a POTCAR file, INCAR.toml [POTCAR] supplies the mapping."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nCs = "Cs_custom"\n')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)

    assert inset._config_dict["POTCAR"] == {"Cs": "Cs_custom"}


def test_derived_potcar_file_precedence_over_incar_toml(
        monkeypatch, tmp_path) -> None:
    """A POTCAR file wins over INCAR.toml [POTCAR] when both are present."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nCs = "Cs_from_toml"\n')
    (tmp_path / "POTCAR").write_text("placeholder")

    class _FakePotcar:
        symbols = ["Cs_from_file", "Cl_from_file"]

    monkeypatch.setattr(
        "pymatgen.io.vasp.inputs.Potcar.from_file",
        lambda path: _FakePotcar(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)

    assert inset._config_dict["POTCAR"] == {
        "Cs": "Cs_from_file", "Cl": "Cl_from_file"}


def test_derived_potcar_spec_not_parsed_as_potcar(tmp_path) -> None:
    """POTCAR.spec (a symbol list) is not mistaken for POTCAR content."""
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    (tmp_path / "POTCAR.spec").write_text("Cs_sv\nCl\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)

    assert inset.potcar_symbols == ["Cs_sv", "Cl"]


def test_derived_kpoints_updates_falls_back_when_no_prev_kpoints(
        monkeypatch, tmp_path) -> None:
    """Without an inherited KPOINTS file, kpoints_updates delegates to super().
    """
    _write_poscar(tmp_path)
    (tmp_path / "INCAR").write_text("ENCUT = 500\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset = IMDDerivedInputSet(directory=tmp_path, no_potcar=True)

    sentinel = {"grid_density": 4242}
    monkeypatch.setattr(
        "pymatgen.io.vasp.sets.VaspInputSet.kpoints_updates",
        property(lambda self: sentinel),
    )
    assert inset.kpoints_updates is sentinel


def test_incar_isif_relaxation_adequate_encut_no_warning(cscl) -> None:
    """ISIF relaxation with ENCUT >= 550 does not emit the ENCUT warning."""
    inset = IMDStandardVaspInputSet(
        structure=cscl, no_potcar=True,
        user_incar_settings={"ENCUT": 550.0, "ISIF": 3, "NSW": 10},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", BadInputSetWarning)
        incar = inset.incar
    assert incar["ISIF"] == 3


def test_potcar_symbols_initialises_missing_potcar_config(cscl) -> None:
    """A set without a POTCAR config still resolves recommended symbols."""
    inset = IMDVaspInputSet(structure=cscl, no_potcar=True)
    assert inset.potcar_symbols == [
        POTCAR_RECOMMENDED.get("Cs", "Cs"),
        POTCAR_RECOMMENDED.get("Cl", "Cl"),
    ]


def test_write_input_records_metadata_when_not_suppressed(
        monkeypatch, cscl, tmp_path) -> None:
    """write_input with no_potcar=False records INCAR.toml metadata."""
    monkeypatch.setattr(
        "pymatgen.io.vasp.sets.VaspInputSet.write_input",
        lambda self, *a, **k: None,
    )
    inset = IMDStandardVaspInputSet(structure=cscl, functional="vdw-df")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset.write_input(tmp_path)
    assert (tmp_path / "IMDVaspInputSet.log").exists()
    assert "functional" in (tmp_path / "INCAR.toml").read_text()


def test_write_incar_toml_metadata_all_defaults(cscl, tmp_path) -> None:
    """All-default metadata produces no INCAR.toml (nothing non-default)."""
    inset = IMDVaspInputSet(structure=cscl, no_potcar=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inset._write_incar_toml_metadata(tmp_path)
    assert not (tmp_path / "INCAR.toml").exists()
