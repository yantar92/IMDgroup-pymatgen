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


"""Tests for matrifflow.pymatgen.cli.mrf_derive."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import os

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Kpoints

import matrifflow.pymatgen.cli.mrf_derive as mrf_derive


def _make_structure() -> Structure:
    """A small Na/C structure with a known cubic lattice (a=4.0)."""
    return Structure(
        Lattice.cubic(4.0),
        ["Na", "Na", "C", "C"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
    )


## _str_to_bool


@pytest.mark.parametrize("value", ["true", "1", "yes", "True", "YES"])
def test_str_to_bool_true(value: str) -> None:
    """Recognised truthy strings convert to True."""
    assert mrf_derive._str_to_bool(value) is True


@pytest.mark.parametrize("value", ["false", "0", "no", "False", "NO"])
def test_str_to_bool_false(value: str) -> None:
    """Recognised falsey strings convert to False."""
    assert mrf_derive._str_to_bool(value) is False


def test_str_to_bool_invalid() -> None:
    """An unrecognised string raises an argparse type error."""
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid boolean"):
        mrf_derive._str_to_bool("maybe")


## _validate_potcar_pair


def test_validate_potcar_pair_element_only() -> None:
    """A bare element symbol (e.g. C:C) is a valid potential name."""
    mrf_derive._validate_potcar_pair("C", "C")


def test_validate_potcar_pair_suffix() -> None:
    """An element with a _suffix (Li_sv) is valid."""
    mrf_derive._validate_potcar_pair("Li", "Li_sv")


def test_validate_potcar_pair_wrong_element() -> None:
    """A symbol that does not start with the element is rejected."""
    with pytest.raises(ValueError, match="does not start with"):
        mrf_derive._validate_potcar_pair("Li", "Na_sv")


def test_validate_potcar_pair_missing_underscore() -> None:
    """A suffix without the _ separator is rejected."""
    with pytest.raises(ValueError, match="not a valid potential name"):
        mrf_derive._validate_potcar_pair("Li", "LiPBE")


## Recording fake for MRFDerivedInputSet


def _patch_inputset(monkeypatch, structure=None):
    """Replace MRFDerivedInputSet with a recording fake.

    The fake captures constructor kwargs and exposes the writable
    attributes the derive subcommands mutate.  It also patches
    :func:`dataclasses.replace`, which :func:`strain` and :func:`insert`
    use to copy an inputset.

    Returns the list of created fake instances.
    """
    records: list[_FakeInputSet] = []

    class _FakeInputSet:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.structure = structure
            self.name = ""
            self.prev_kpoints = None
            self.prev_incar = None
            self._config_dict: dict = {}
            self.files_to_transfer: dict = {}
            self.written = None
            records.append(self)

        def write_input(self, output_dir=None, **kw) -> None:
            self.written = output_dir

    def _fake_replace(obj, **changes):
        new = copy.copy(obj)
        for key, value in changes.items():
            setattr(new, key, value)
        return new

    monkeypatch.setattr(mrf_derive, "MRFDerivedInputSet", _FakeInputSet)
    monkeypatch.setattr(dataclasses, "replace", _fake_replace)
    return records


def _ns(**kwargs) -> argparse.Namespace:
    """Build a Namespace with the common derive args pre-filled."""
    defaults = {"input_directory": ".", "inherit_prev_incarpy": False}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


## add_args


def test_add_args_registers_subcommands() -> None:
    """Every subcommand is registered with its func_derive default."""
    parser = argparse.ArgumentParser()
    mrf_derive.add_args(parser)

    cases = [
        ([".", "fix", "{'Na': [True, True, True]}"], mrf_derive.fix),
        ([".", "incar", "ENCUT:520"], mrf_derive.incar),
        ([".", "potcar", "Li:Li_sv"], mrf_derive.potcar),
        ([".", "supercell", "2x2x2"], mrf_derive.supercell),
        ([".", "functional", "PBE"], mrf_derive.functional),
        ([".", "relax", "RELAX_POS"], mrf_derive.relax),
        ([".", "kpoints", "--grid", "1x1x5"], mrf_derive.kpoints),
        ([".", "strain"], mrf_derive.strain),
        ([".", "perturb"], mrf_derive.perturb),
        ([".", "scf"], mrf_derive.scf),
        ([".", "insert", "Li"], mrf_derive.insert),
        ([".", "ins", "Li"], mrf_derive.insert),
        ([".", "del", "Na"], mrf_derive.delete),
        ([".", "fill", "--uniq_sites", "a", "b"], mrf_derive.fill),
        ([".", "neb", "target_dir"], mrf_derive.neb),
        ([".", "neb_diffusion", "--diffusion_points", "a", "b"], mrf_derive.neb_diffusion),
        ([".", "atat", "str.out", "--kpoints", "1000"], mrf_derive.atat),
    ]
    for argv, func in cases:
        args = parser.parse_args(argv)
        assert args.func_derive is func, argv


def test_add_args_input_directory_positional() -> None:
    """The input_directory positional precedes the subcommand."""
    parser = argparse.ArgumentParser()
    mrf_derive.add_args(parser)
    args = parser.parse_args(["some_dir", "scf"])
    assert args.input_directory == "some_dir"
    assert args.func_derive is mrf_derive.scf


## perturb


def test_perturb_calls_structure_perturb(monkeypatch) -> None:
    """Perturb forwards distance/frac_tol and names the output dir."""
    records = _patch_inputset(monkeypatch, structure=_make_structure())
    calls: dict = {}
    monkeypatch.setattr(
        mrf_derive,
        "structure_perturb",
        lambda **kw: calls.update(kw),
    )

    args = _ns(distance=0.2, frac_tol=0.4)
    result = mrf_derive.perturb(args)

    assert calls["distance"] == 0.2
    assert calls["frac_tol"] == 0.4
    assert calls["min_distance"] is None
    assert calls["structure"] is records[0].structure
    assert records[0].name == "PERTURB.0.2"
    assert result["inputsets"] == [records[0]]


## functional


def test_functional_sets_name_and_kwarg(monkeypatch) -> None:
    """Functional passes the choice through and names the output dir."""
    records = _patch_inputset(monkeypatch)
    result = mrf_derive.functional(_ns(functional_type="optB88-vdW"))
    assert records[0].kwargs["functional"] == "optB88-vdW"
    assert records[0].name == "optB88-vdW"
    assert result["inputsets"] == [records[0]]


## supercell


def test_supercell_scales_and_rescales(monkeypatch) -> None:
    """Supercell scales the structure and sets the k-point density."""
    records = _patch_inputset(monkeypatch, structure=_make_structure())
    args = _ns(supercell_size="2x1x1", kpoint_density=8000)
    result = mrf_derive.supercell(args)

    assert records[0].kwargs["user_kpoints_settings"] == {"grid_density": 8000}
    assert len(records[0].structure) == 8  # 4 * 2 * 1 * 1
    assert records[0].name == "2x1x1"
    assert result["inputsets"] == [records[0]]


## kpoints


def test_kpoints_grid_path(monkeypatch) -> None:
    """--grid builds an explicit Kpoints from the grid string."""
    records = _patch_inputset(monkeypatch)
    args = _ns(grid="1x1x5", density=10000)
    result = mrf_derive.kpoints(args)

    assert "user_kpoints_settings" not in records[0].kwargs
    assert records[0].prev_kpoints == Kpoints(kpts=[(1, 1, 5)])
    assert records[0].name == "KPOINTS.1x1x5"
    assert result["inputsets"] == [records[0]]


def test_kpoints_density_path(monkeypatch) -> None:
    """Without --grid, the density is passed to the inputset."""
    records = _patch_inputset(monkeypatch)
    args = _ns(grid=None, density=8000)
    result = mrf_derive.kpoints(args)

    assert records[0].kwargs["user_kpoints_settings"] == {"grid_density": 8000}
    assert records[0].name == "KPOINTS.8000"
    assert result["inputsets"] == [records[0]]


## scf


def test_scf_sets_static_incar(monkeypatch) -> None:
    """SCF sets the static-calculation INCAR overrides."""
    records = _patch_inputset(monkeypatch)
    result = mrf_derive.scf(_ns())

    assert records[0].kwargs["user_incar_settings"] == {
        "NSW": 0, "IBRION": -1, "ISMEAR": -5,
        "IMAGES": None, "SPRING": None,
    }
    assert records[0].name == "SCF"
    assert result["inputsets"] == [records[0]]


## relax


def test_relax_position_only(monkeypatch) -> None:
    """Position-only relaxation maps ISIF and sets the relax overrides."""
    records = _patch_inputset(monkeypatch)
    args = _ns(isif="RELAX_POS", auto_encut=False)
    result = mrf_derive.relax(args)

    overrides = records[0].kwargs["user_incar_settings"]
    assert overrides["ISTART"] == 0
    assert overrides["NSW"] == 500
    assert overrides["IBRION"] == 2
    assert overrides["ISIF"] == 2
    assert overrides["EDIFF"] == pytest.approx(1e-6)
    assert overrides["EDIFFG"] == pytest.approx(-0.01)
    assert "ENCUT" not in overrides
    assert records[0].name == "relax.RELAX_POS"
    assert result["inputsets"] == [records[0]]


def test_relax_auto_encut_shape_volume(monkeypatch) -> None:
    """Shape/volume relaxation bumps ENCUT to 550 and warns."""
    records = _patch_inputset(monkeypatch)
    args = _ns(isif="RELAX_POS_SHAPE_VOL", auto_encut=True)
    with pytest.warns(UserWarning, match="Shape/volume relaxation"):
        mrf_derive.relax(args)

    assert records[0].kwargs["user_incar_settings"]["ENCUT"] == 550.0


def test_relax_auto_encut_fixed(monkeypatch) -> None:
    """Fixed shape/volume relaxation sets ENCUT to 500 and warns."""
    records = _patch_inputset(monkeypatch)
    args = _ns(isif="RELAX_POS", auto_encut=True)
    with pytest.warns(UserWarning, match="Shape and volume are fixed"):
        mrf_derive.relax(args)

    assert records[0].kwargs["user_incar_settings"]["ENCUT"] == 500.0


## incar


def test_incar_sets_and_unsets(monkeypatch) -> None:
    """PARAM:VALUE pairs become overrides; VALUE=None removes a tag."""
    records = _patch_inputset(monkeypatch)
    args = _ns(parameters=["ENCUT:520", "ISMEAR:None"])
    result = mrf_derive.incar(args)

    assert records[0].kwargs["user_incar_settings"] == {
        "ENCUT": "520", "ISMEAR": None,
    }
    assert records[0].name == "ENCUT.520,ISMEAR.None"
    assert result["inputsets"] == [records[0]]


def test_incar_no_parameters_warns(monkeypatch) -> None:
    """A None parameters value warns and produces a copy."""
    records = _patch_inputset(monkeypatch)
    args = _ns(parameters=None)
    with pytest.warns(UserWarning, match="No INCAR settings"):
        mrf_derive.incar(args)
    assert records[0].kwargs["user_incar_settings"] == {}


## potcar


def test_potcar_sets_config(monkeypatch) -> None:
    """ELEMENT:POTCAR pairs are validated and written to config."""
    records = _patch_inputset(monkeypatch)
    args = _ns(parameters=["Li:Li_sv", "C:C"])
    result = mrf_derive.potcar(args)

    assert records[0]._config_dict["POTCAR"] == {"Li": "Li_sv", "C": "C"}
    assert records[0].name == "POTCAR.Li.Li_sv,C.C"
    assert result["inputsets"] == [records[0]]


def test_potcar_empty_warns(monkeypatch) -> None:
    """No POTCAR settings warns and uses the bare POTCAR suffix."""
    records = _patch_inputset(monkeypatch)
    args = _ns(parameters=[])
    with pytest.warns(UserWarning, match="No POTCAR settings"):
        mrf_derive.potcar(args)
    assert records[0].name == "POTCAR"


def test_potcar_missing_colon_raises(monkeypatch) -> None:
    """An override without a colon is rejected."""
    _patch_inputset(monkeypatch)
    with pytest.raises(ValueError, match="Invalid POTCAR setting"):
        mrf_derive.potcar(_ns(parameters=["LiLi_sv"]))


def test_potcar_empty_value_raises(monkeypatch) -> None:
    """An empty POTCAR value is rejected."""
    _patch_inputset(monkeypatch)
    with pytest.raises(ValueError, match="Empty POTCAR value"):
        mrf_derive.potcar(_ns(parameters=["Li:"]))


## fix


def test_fix_applies_constraints(monkeypatch) -> None:
    """Constraints are applied to matching species."""
    structure = _make_structure()
    records = _patch_inputset(monkeypatch, structure=structure)
    args = _ns(
        constraints="{'Na': [True, True, True], 'C': [False, False, False]}",
        discard_previous=False,
    )
    result = mrf_derive.fix(args)

    na_sites = [s for s in structure if s.specie.symbol == "Na"]
    c_sites = [s for s in structure if s.specie.symbol == "C"]
    assert all(
        s.properties["selective_dynamics"] == [True, True, True] for s in na_sites
    )
    assert all(
        s.properties["selective_dynamics"] == [False, False, False] for s in c_sites
    )
    assert records[0].name == "selective_dynamics"
    assert result["inputsets"] == [records[0]]


def test_fix_bad_syntax_raises(monkeypatch) -> None:
    """A non-dict string raises a parse error."""
    _patch_inputset(monkeypatch, structure=_make_structure())
    args = _ns(constraints="not a dict", discard_previous=False)
    with pytest.raises(ValueError, match="Could not parse constraints"):
        mrf_derive.fix(args)


def test_fix_non_dict_raises(monkeypatch) -> None:
    """A parseable but non-dict literal is rejected."""
    _patch_inputset(monkeypatch, structure=_make_structure())
    args = _ns(constraints="[1, 2, 3]", discard_previous=False)
    with pytest.raises(ValueError, match="Could not parse constraints"):
        mrf_derive.fix(args)


def test_fix_wrong_length_raises(monkeypatch) -> None:
    """A constraint list of the wrong length is rejected."""
    _patch_inputset(monkeypatch, structure=_make_structure())
    args = _ns(constraints="{'Na': [True, True]}", discard_previous=False)
    with pytest.raises(ValueError, match="Invalid constraint format for Na"):
        mrf_derive.fix(args)


def test_fix_keeps_existing_warns(monkeypatch) -> None:
    """Constraints on non-target species are kept (with a warning)."""
    structure = _make_structure()
    # Pre-existing constraint on a C site (index 2), which the new
    # constraints dict does not mention, so it must be kept.
    structure[2].properties["selective_dynamics"] = [False, False, False]
    _patch_inputset(monkeypatch, structure=structure)
    args = _ns(
        constraints="{'Na': [True, True, True]}",
        discard_previous=False,
    )
    with pytest.warns(UserWarning, match="Keeping previous"):
        mrf_derive.fix(args)
    # Na sites pick up the new constraint; the C site's is untouched.
    na_sites = [s for s in structure if s.specie.symbol == "Na"]
    c_sites = [s for s in structure if s.specie.symbol == "C"]
    assert all(
        s.properties["selective_dynamics"] == [True, True, True] for s in na_sites
    )
    assert c_sites[0].properties["selective_dynamics"] == [False, False, False]


def test_fix_discard_previous(monkeypatch) -> None:
    """--discard_previous clears constraints on non-target species."""
    structure = _make_structure()
    # Same pre-existing constraint as the keep test, but discard_previous
    # must now clear it instead of keeping it.
    structure[2].properties["selective_dynamics"] = [False, False, False]
    _patch_inputset(monkeypatch, structure=structure)
    args = _ns(
        constraints="{'Na': [True, True, True]}",
        discard_previous=True,
    )
    mrf_derive.fix(args)
    na_sites = [s for s in structure if s.specie.symbol == "Na"]
    c_sites = [s for s in structure if s.specie.symbol == "C"]
    assert all(
        s.properties["selective_dynamics"] == [True, True, True] for s in na_sites
    )
    assert c_sites[0].properties["selective_dynamics"] is None


## strain


def _strain_args(**over):
    base = dict(
        input_directory=".",
        inherit_prev_incarpy=False,
        selective_dynamics=None,
        amin="100%", amax="100%", asteps=1,
        bmin="100%", bmax="100%", bsteps=1,
        cmin="100%", cmax="100%", csteps=1,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_strain_percent_multi_step(monkeypatch) -> None:
    """Percent values are converted to relative strain and enumerated."""
    records = _patch_inputset(monkeypatch, structure=_make_structure())
    args = _strain_args(amin="98%", amax="100%", asteps=2)
    result = mrf_derive.strain(args)

    outputs = result["inputsets"]
    assert [o.name for o in outputs] == [
        "strain.a.-0.02.c.0.00",
        "strain.c.0.00",
    ]
    assert outputs[0].structure.lattice.a == pytest.approx(3.92)
    assert outputs[1].structure.lattice.a == pytest.approx(4.0)
    # The original inputset structure is not mutated.
    assert records[0].structure.lattice.a == pytest.approx(4.0)


def test_strain_absolute(monkeypatch) -> None:
    """Absolute values are normalised by the initial lattice parameter."""
    _patch_inputset(monkeypatch, structure=_make_structure())
    args = _strain_args(amin="8.0", amax="8.0", asteps=1)
    result = mrf_derive.strain(args)

    outputs = result["inputsets"]
    assert outputs[0].name == "strain.a.1.00.c.0.00"
    assert outputs[0].structure.lattice.a == pytest.approx(8.0)


def test_strain_selective_dynamics(monkeypatch) -> None:
    """--selective-dynamics is applied to every site before straining."""
    structure = _make_structure()
    _patch_inputset(monkeypatch, structure=structure)
    args = _strain_args(selective_dynamics=[True, False, True])
    mrf_derive.strain(args)

    for site in structure:
        assert site.properties["selective_dynamics"] == [True, False, True]


## delete


def test_delete_removes_species(monkeypatch) -> None:
    """Matching species are removed and the output dir is named."""
    structure = _make_structure()
    records = _patch_inputset(monkeypatch, structure=structure)
    args = _ns(what=["Na"])
    result = mrf_derive.delete(args)

    assert len(structure) == 2
    assert all(s.specie.symbol != "Na" for s in structure)
    assert records[0].name == "del.Na"
    assert result["inputsets"] == [records[0]]


def test_delete_missing_species_warns(monkeypatch) -> None:
    """Deleting a species that is absent warns and keeps the structure."""
    structure = _make_structure()
    records = _patch_inputset(monkeypatch, structure=structure)
    args = _ns(what=["K"])
    with pytest.warns(UserWarning, match="Nothing was deleted"):
        mrf_derive.delete(args)
    assert len(structure) == 4
    assert records[0].name == "del.K"


## insert


def _patch_transformer(monkeypatch, structures):
    """Replace InsertMoleculeTransformation with a recording fake."""
    instances: list = []

    class _FakeTransformer:
        def __init__(self, atom, **kwargs) -> None:
            self.atom = atom
            self.kwargs = kwargs
            instances.append(self)

        def all_inserts(self, structure, limit=None):
            return structures

    monkeypatch.setattr(
        mrf_derive, "InsertMoleculeTransformation", _FakeTransformer)
    return instances


def test_insert_write_path(monkeypatch) -> None:
    """Insert writes one inputset per returned structure."""
    host = _make_structure()
    _patch_inputset(monkeypatch, structure=host)
    structures = [_make_structure(), _make_structure()]
    instances = _patch_transformer(monkeypatch, structures)

    args = _ns(atom="Li", limit=5, step=0.5, step_noise=0.1,
               threshold=0.75, no_matcher=False, count=False,
               multithread=False)
    result = mrf_derive.insert(args)

    assert instances[0].atom == "Li"
    assert instances[0].kwargs["step"] == 0.5
    assert instances[0].kwargs["step_noise"] == 0.1
    assert instances[0].kwargs["proximity_threshold"] == 0.75
    assert instances[0].kwargs["selective_dynamics"] == [True, True, True]
    assert "matcher" not in instances[0].kwargs
    assert [i.name for i in result["inputsets"]] == ["ins.Li.0", "ins.Li.1"]


def test_insert_no_matcher(monkeypatch) -> None:
    """--no-matcher passes matcher=None to the transformation."""
    _patch_inputset(monkeypatch, structure=_make_structure())
    instances = _patch_transformer(monkeypatch, [_make_structure()])

    args = _ns(atom="Li", limit=None, step=None, step_noise=None,
               threshold=0.75, no_matcher=True, count=False,
               multithread=False)
    mrf_derive.insert(args)

    assert instances[0].kwargs["matcher"] is None


def test_insert_count_prints(monkeypatch, capsys) -> None:
    """--count prints the number of structures and writes nothing."""
    _patch_inputset(monkeypatch, structure=_make_structure())
    _patch_transformer(monkeypatch, [_make_structure(), _make_structure()])

    args = _ns(atom="Li", limit=None, step=None, step_noise=None,
               threshold=0.75, no_matcher=False, count=True,
               multithread=False)
    result = mrf_derive.insert(args)

    assert capsys.readouterr().out.strip() == "2"
    assert result["inputsets"] == []


## derive (orchestrator)


class _DeriveInputSet:
    """Minimal inputset stand-in for the derive() orchestrator."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.written = None

    def write_input(self, output_dir=None, **kw) -> None:
        self.written = output_dir


def _derive_args(input_dir, **over) -> argparse.Namespace:
    base = argparse.Namespace(
        input_directory=str(input_dir),
        force_running=False,
        output=None,
        output_prefix=None,
        subdir=None,
        overwrite_output="True",
        func_derive=lambda args: {"inputsets": []},
    )
    for key, value in over.items():
        setattr(base, key, value)
    return base


def test_derive_missing_input_dir(tmp_path) -> None:
    """A non-existent input directory is rejected."""
    args = _derive_args(tmp_path / "nope")
    with pytest.raises(ValueError, match="Input path does not exist"):
        mrf_derive.derive(args)


def test_derive_refuses_running(tmp_path) -> None:
    """A RUNNING file blocks the run unless forced."""
    (tmp_path / "RUNNING").write_text("")
    args = _derive_args(tmp_path)
    with pytest.raises(IOError, match="RUNNING"):
        mrf_derive.derive(args)


def test_derive_force_running(tmp_path) -> None:
    """--force_running bypasses the RUNNING check."""
    (tmp_path / "RUNNING").write_text("")
    ins = _DeriveInputSet("SCF")
    args = _derive_args(tmp_path, force_running=True,
                        func_derive=lambda a: {"inputsets": [ins]})
    assert mrf_derive.derive(args) == 0
    assert ins.written is not None


def test_derive_empty_output(tmp_path) -> None:
    """An empty --output is rejected."""
    ins = _DeriveInputSet("SCF")
    args = _derive_args(tmp_path, output="",
                        func_derive=lambda a: {"inputsets": [ins]})
    with pytest.raises(ValueError, match="--output cannot be empty"):
        mrf_derive.derive(args)


def test_derive_single_default_prefix(tmp_path, monkeypatch) -> None:
    """A single inputset with no --output uses <input>.<name>."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    ins = _DeriveInputSet("SCF")
    args = _derive_args(run, func_derive=lambda a: {"inputsets": [ins]})
    assert mrf_derive.derive(args) == 0
    assert ins.written == "run.SCF"


def test_derive_single_explicit_output(tmp_path, monkeypatch) -> None:
    """A single inputset with --output writes directly there."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    ins = _DeriveInputSet("SCF")
    args = _derive_args(run, output="out",
                        func_derive=lambda a: {"inputsets": [ins]})
    assert mrf_derive.derive(args) == 0
    assert ins.written == "out"


def test_derive_multiple_output(tmp_path, monkeypatch) -> None:
    """Multiple inputsets are written under --output/<prefix><name>."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    ins1 = _DeriveInputSet("SCF")
    ins2 = _DeriveInputSet("relax.X")
    args = _derive_args(run, output="out",
                        func_derive=lambda a: {"inputsets": [ins1, ins2]})
    assert mrf_derive.derive(args) == 0
    assert ins1.written == os.path.join("out", "run.SCF")
    assert ins2.written == os.path.join("out", "run.relax.X")


def test_derive_output_prefix(tmp_path, monkeypatch) -> None:
    """--output-prefix replaces the input-dir-based prefix."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    ins = _DeriveInputSet("SCF")
    args = _derive_args(run, output_prefix="pre_",
                        func_derive=lambda a: {"inputsets": [ins]})
    assert mrf_derive.derive(args) == 0
    assert ins.written == "pre_SCF"


def test_derive_subdir(tmp_path, monkeypatch) -> None:
    """--subdir nests the output one level deeper."""
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    ins = _DeriveInputSet("SCF")
    args = _derive_args(run, subdir="sub",
                        func_derive=lambda a: {"inputsets": [ins]})
    assert mrf_derive.derive(args) == 0
    assert ins.written == os.path.join("run.SCF", "sub")


def test_derive_overwrite(tmp_path, monkeypatch) -> None:
    """A non-empty output dir is overwritten (with a warning) by default."""
    run = tmp_path / "run"
    run.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "file").write_text("x")
    monkeypatch.chdir(tmp_path)
    ins = _DeriveInputSet("SCF")
    args = _derive_args(run, output="out",
                        func_derive=lambda a: {"inputsets": [ins]})
    with pytest.warns(UserWarning, match="Overwriting"):
        assert mrf_derive.derive(args) == 0
    assert ins.written == "out"


def test_derive_skip_non_empty(tmp_path, monkeypatch) -> None:
    """overwrite_output=False skips a non-empty output dir."""
    run = tmp_path / "run"
    run.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "file").write_text("x")
    monkeypatch.chdir(tmp_path)
    ins = _DeriveInputSet("SCF")
    args = _derive_args(run, output="out", overwrite_output="False",
                        func_derive=lambda a: {"inputsets": [ins]})
    with pytest.warns(UserWarning, match="Skipping"):
        assert mrf_derive.derive(args) == 0
    assert ins.written is None
