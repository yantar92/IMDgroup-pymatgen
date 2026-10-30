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


"""Tests for IMDgroup.pymatgen.core.structure.

These helpers extend pymatgen's Structure, so the tests focus on the
values we compute and the decisions we make around the pymatgen calls
(see the testing principle in tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import DummySpecies, Lattice, Structure
from pymatgen.util.testing import MatSciTest

import IMDgroup.pymatgen.core.structure as structure_module
from IMDgroup.pymatgen.core.structure import (
    FRAMEWORK_SYMPREC,
    IMDStructure,
    StructureDuplicateWarning,
    filter_framework,
    get_framework_elements,
    get_framework_space_group,
    get_matched_structure,
    get_supercell_size,
    merge_structures,
    reduce_supercell,
    structure_diff,
    structure_distance,
    structure_interpolate2,
    structure_is_valid2,
    structure_matches,
    structure_perturb,
    structure_remove_duplicates,
    structure_strain,
)


@pytest.fixture
def cscl() -> Structure:
    """Two-site CsCl structure from pymatgen's curated set."""
    return MatSciTest().get_structure("CsCl")


def _scale_lattice(
        structure: Structure, factor: float, axis: int = 0) -> Structure:
    """Return a copy of ``structure`` stretched along one lattice axis."""
    matrix = structure.lattice.matrix.copy()
    matrix[axis, axis] *= factor
    return Structure(Lattice(matrix), structure.species, structure.frac_coords)


# --- framework helpers ------------------------------------------------------


def _carbon_lithium() -> Structure:
    """A carbon framework with one intercalated Li atom."""
    return Structure(
        lattice=Lattice.cubic(4.0),
        species=['C', 'C', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def test_get_framework_elements_most_common() -> None:
    """The framework element is the most common species."""
    assert get_framework_elements(_carbon_lithium()) == ['C']


def test_filter_framework_removes_mobile_species() -> None:
    """Filtering keeps only the framework elements."""
    framework = filter_framework(_carbon_lithium(), ['C'])
    assert len(framework.species) == 2
    assert set(str(s.specie) for s in framework) == {'C'}


def test_get_framework_space_group_ignores_mobile_species() -> None:
    """The framework space group excludes the intercalated species.

    The C2 framework is body-centred cubic (Im-3m); the Li intercalant
    lowers the full structure to R-3m.
    """
    assert get_framework_space_group(_carbon_lithium()) == 'Im-3m'


def test_get_framework_space_group_explicit_elements() -> None:
    """An explicit element list overrides the most-common-element default."""
    assert get_framework_space_group(
        _carbon_lithium(), framework_elements=['Li']) == 'Pm-3m'


def test_framework_symprec_shared_default() -> None:
    """FRAMEWORK_SYMPREC is the loose tolerance shared by framework checks."""
    assert FRAMEWORK_SYMPREC == 0.1


# --- structure_strain -------------------------------------------------------


def test_structure_strain_uniaxial(cscl: Structure) -> None:
    """A uniaxial stretch produces a symmetric diagonal strain tensor."""
    deformed = _scale_lattice(cscl, 1.02)
    strain = structure_strain(cscl, deformed)
    assert strain.shape == (3, 3)
    assert np.allclose(strain, strain.T)
    assert np.isclose(strain[0, 0], 0.02)
    assert np.isclose(strain[1, 1], 0.0)


def test_structure_strain_identical_is_zero(cscl: Structure) -> None:
    """Identical structures have zero strain."""
    assert np.allclose(structure_strain(cscl, cscl), 0.0)


# --- structure_distance -----------------------------------------------------


def test_structure_distance_identical_is_zero(cscl: Structure) -> None:
    """Identical structures are at distance zero."""
    assert structure_distance(cscl, cscl) == 0.0


def test_structure_distance_measures_displacement(cscl: Structure) -> None:
    """A single displaced site contributes its cartesian displacement."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    assert np.isclose(structure_distance(cscl, moved), cscl.lattice.a * 0.2)


def test_structure_distance_norm_divides_by_displaced_count(
        cscl: Structure) -> None:
    """With norm=True the distance is divided by the displaced-site count."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    moved.translate_sites([1], [0, 0.3, 0], frac_coords=True)
    assert np.isclose(
        structure_distance(cscl, moved, norm=True),
        structure_distance(cscl, moved) / 2,
    )


def test_structure_distance_max_dist_returns_early(cscl: Structure) -> None:
    """max_dist stops accumulation once the partial distance exceeds it."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    moved.translate_sites([1], [0, 0.3, 0], frac_coords=True)
    full = structure_distance(cscl, moved)
    assert structure_distance(cscl, moved, max_dist=5.0) == full
    assert structure_distance(cscl, moved, max_dist=0.5) < full


def test_structure_distance_tol_ignores_small_displacement(
        cscl: Structure) -> None:
    """A displacement below tol is ignored; above tol it contributes."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.05, 0, 0], frac_coords=True)
    assert structure_distance(cscl, moved, tol=0.5) == 0.0
    assert structure_distance(cscl, moved, tol=0.1) > 0.0


def test_structure_distance_handles_shuffled_order(cscl: Structure) -> None:
    """A shuffled site order does not change the distance."""
    shuffled = Structure(
        cscl.lattice,
        [cscl[1].species, cscl[0].species],
        [cscl[1].frac_coords, cscl[0].frac_coords],
    )
    assert structure_distance(cscl, shuffled) == 0.0


def test_structure_distance_match_first_false_skips_matching(
        cscl: Structure, monkeypatch) -> None:
    """match_first=False computes the distance without calling the matcher."""
    def _boom(*args, **kwargs):
        raise AssertionError("get_matched_structure should not be called")
    monkeypatch.setattr(structure_module, "get_matched_structure", _boom)

    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    assert np.isclose(
        structure_distance(cscl, moved, match_first=False),
        cscl.lattice.a * 0.2,
    )


# --- structure_diff ---------------------------------------------------------


def test_structure_diff_reports_per_site_vectors(cscl: Structure) -> None:
    """Displacements are returned as one cartesian vector per site."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    vectors = structure_diff(cscl, moved)
    assert len(vectors) == len(cscl)
    assert np.isclose(vectors[0][0], cscl.lattice.a * 0.2)
    assert np.allclose(vectors[1], 0.0)


def test_structure_diff_zeroes_small_displacements(cscl: Structure) -> None:
    """Displacements below tol are reported as the zero vector."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.001, 0, 0], frac_coords=True)
    vectors = structure_diff(cscl, moved, tol=0.1)
    assert np.allclose(vectors[0], 0.0)


def test_structure_diff_match_first_false_skips_matching(
        cscl: Structure, monkeypatch) -> None:
    """match_first=False computes vectors without calling the matcher."""
    def _boom(*args, **kwargs):
        raise AssertionError("get_matched_structure should not be called")
    monkeypatch.setattr(structure_module, "get_matched_structure", _boom)

    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    vectors = structure_diff(cscl, moved, match_first=False)
    assert len(vectors) == len(cscl)
    assert np.isclose(vectors[0][0], cscl.lattice.a * 0.2)


# --- get_matched_structure --------------------------------------------------


def test_get_matched_structure_reorders_sites(cscl: Structure) -> None:
    """Sites are reordered so target[idx] corresponds to reference[idx]."""
    reversed_struct = Structure(
        cscl.lattice,
        [cscl[1].species, cscl[0].species],
        [cscl[1].frac_coords, cscl[0].frac_coords],
    )
    matched = get_matched_structure(cscl, reversed_struct)
    assert isinstance(matched, IMDStructure)
    assert matched.species == cscl.species
    assert np.allclose(matched.frac_coords, cscl.frac_coords)


def test_get_matched_structure_appends_extra_sites(cscl: Structure) -> None:
    """Sites beyond the reference length are appended at the end."""
    extra = Structure(
        cscl.lattice,
        [cscl[0].species, cscl[1].species, "Na"],
        [cscl[0].frac_coords, cscl[1].frac_coords, [0.25, 0.25, 0.25]],
    )
    matched = get_matched_structure(cscl, extra)
    assert len(matched) == 3
    assert matched[0].species_string == cscl[0].species_string
    assert matched[1].species_string == cscl[1].species_string
    assert matched[2].species_string == "Na"


def test_get_matched_structure_too_few_sites(cscl: Structure) -> None:
    """A target shorter than the reference is rejected."""
    with pytest.raises(ValueError, match="too few sites"):
        get_matched_structure(cscl, Structure(cscl.lattice, ["Cs"], [[0, 0, 0]]))


def test_get_matched_structure_different_lattices(cscl: Structure) -> None:
    """Structures with different lattices are rejected."""
    other = Structure(Lattice.cubic(5.0), cscl.species, cscl.frac_coords)
    with pytest.raises(ValueError, match="different lattices"):
        get_matched_structure(cscl, other)


def test_get_matched_structure_pbc_false() -> None:
    """pbc=False uses non-periodic distances.

    The two Cs sites in the target sit at [0.45, 0, 0] and [0.95, 0, 0].
    With periodic distances the reference site at [0, 0, 0] matches the
    wrapped [0.95, 0, 0] (distance 0.05); without wrapping it matches
    [0.45, 0, 0] (distance 0.45).  The assertion therefore fails if the
    code falls back to periodic distances.
    """
    lattice = Lattice.cubic(10.0)
    reference = Structure(lattice, ["Cs", "Cs"], [[0, 0, 0], [0.5, 0, 0]])
    target = Structure(lattice, ["Cs", "Cs"], [[0.95, 0, 0], [0.45, 0, 0]])
    matched = get_matched_structure(reference, target, pbc=False)
    assert np.allclose(matched[0].frac_coords, [0.45, 0, 0])


def test_get_matched_structure_match_species_false(cscl: Structure) -> None:
    """match_species=False pairs sites by proximity regardless of species."""
    mismatched = Structure(
        cscl.lattice, ["Na", "Cl"],
        [cscl[0].frac_coords, cscl[1].frac_coords],
    )
    matched = get_matched_structure(cscl, mismatched, match_species=False)
    assert [s.species_string for s in matched] == ["Na", "Cl"]


def test_get_matched_structure_unable_to_match_raises(cscl: Structure) -> None:
    """A species absent from the target makes matching fail."""
    mismatched = Structure(
        cscl.lattice, ["Na", "Cl"],
        [cscl[0].frac_coords, cscl[1].frac_coords],
    )
    with pytest.raises(ValueError, match="Unable to reliably match"):
        get_matched_structure(cscl, mismatched)


def test_get_matched_structure_already_matched_fast_path(
        cscl: Structure) -> None:
    """An already-ordered target returns an IMDStructure copy unchanged."""
    matched = get_matched_structure(cscl, cscl.copy())
    assert isinstance(matched, IMDStructure)
    assert matched.species == cscl.species


def test_get_matched_structure_appends_extra_after_reorder(
        cscl: Structure) -> None:
    """Extra sites are appended even when matching also reorders."""
    target = Structure(
        cscl.lattice,
        [cscl[1].species, cscl[0].species, "Na"],
        [cscl[1].frac_coords, cscl[0].frac_coords, [0.25, 0.25, 0.25]],
    )
    matched = get_matched_structure(cscl, target)
    assert [s.species_string for s in matched] == ["Cs", "Cl", "Na"]


# --- merge_structures -------------------------------------------------------


def test_merge_structures_combines_sites(cscl: Structure) -> None:
    """Sites from all inputs land in a single structure with shared lattice."""
    cs = Structure(cscl.lattice, ["Cs"], [[0, 0, 0]])
    cl = Structure(cscl.lattice, ["Cl"], [[0.5, 0.5, 0.5]])
    merged = merge_structures([cs, cl])
    assert merged.composition == cscl.composition
    assert len(merged) == 2


def test_merge_structures_empty_raises() -> None:
    """An empty input list is rejected."""
    with pytest.raises(AssertionError):
        merge_structures([])


def test_merge_structures_lattice_mismatch_raises(cscl: Structure) -> None:
    """Inputs with different lattices are rejected."""
    cs = Structure(cscl.lattice, ["Cs"], [[0, 0, 0]])
    cl = Structure(Lattice.cubic(5.0), ["Cl"], [[0.5, 0.5, 0.5]])
    with pytest.raises(AssertionError):
        merge_structures([cs, cl])


# --- structure_is_valid2 ----------------------------------------------------


def test_structure_is_valid2_accepts_well_spaced(cscl: Structure) -> None:
    """A structure with well-separated atoms is valid."""
    assert structure_is_valid2(cscl)


def test_structure_is_valid2_rejects_close_atoms(cscl: Structure) -> None:
    """Atoms below the radius threshold are rejected."""
    close = Structure(cscl.lattice, ["Cs", "Cl"], [[0, 0, 0], [0.05, 0, 0]])
    assert not structure_is_valid2(close)


# --- reduce_supercell / get_supercell_size ----------------------------------


def test_reduce_supercell_returns_primitive(cscl: Structure) -> None:
    """A supercell reduces to its primitive cell."""
    reduced = reduce_supercell(cscl * (2, 3, 4))
    assert len(reduced) == len(cscl)
    assert np.allclose(reduced.lattice.parameters, cscl.lattice.parameters)


def test_get_supercell_size_primitive(cscl: Structure) -> None:
    """A primitive cell reports a 1x1x1 supercell size."""
    assert get_supercell_size(cscl) == (1, 1, 1)


def test_get_supercell_size_counts_repetitions(cscl: Structure) -> None:
    """The A x B x C repetition factors are recovered from a supercell."""
    assert get_supercell_size(cscl * (2, 3, 4)) == (2, 3, 4)


# --- structure_remove_duplicates --------------------------------------------


def test_structure_remove_duplicates_replaces_copy_with_none(
        cscl: Structure) -> None:
    """A duplicate is replaced with None, preserving list length."""
    result = structure_remove_duplicates([cscl, cscl.copy()])
    assert len(result) == 2
    assert result[0] is cscl
    assert result[1] is None


def test_structure_remove_duplicates_replaces_supercell_with_none(
        cscl: Structure) -> None:
    """A supercell of a kept structure is replaced with None."""
    result = structure_remove_duplicates([cscl, cscl * (2, 1, 1)])
    assert len(result) == 2
    assert result[1] is None


def test_structure_remove_duplicates_preserves_none(cscl: Structure) -> None:
    """None entries are preserved in place and order."""
    result = structure_remove_duplicates([None, cscl, None])
    assert result[0] is None
    assert result[1] is cscl
    assert result[2] is None


# --- structure_perturb ------------------------------------------------------


def test_structure_perturb_returns_same_object(cscl: Structure) -> None:
    """The perturbed structure is modified in place and stays valid."""
    result = structure_perturb(cscl, distance=0.1)
    assert result is cscl
    assert structure_is_valid2(cscl)


def test_structure_perturb_honours_selective_dynamics(
        cscl: Structure) -> None:
    """Sites fixed by selective dynamics are not moved."""
    struct = cscl.copy()
    struct.add_site_property(
        "selective_dynamics",
        [[True, True, True], [False, False, False]],
    )
    fixed = struct[1].frac_coords.copy()
    with pytest.warns(UserWarning, match="selective_dynamics"):
        structure_perturb(struct, distance=0.1)
    assert np.allclose(struct[1].frac_coords, fixed)
    assert structure_is_valid2(struct)


def test_structure_perturb_gives_up_after_100_attempts(
        cscl: Structure, monkeypatch) -> None:
    """A never-valid perturbation raises after 100 attempts."""
    calls: list[int] = []

    def _always_invalid(structure, frac_tol: float = 0.5) -> bool:
        # The initial assertion must pass; every loop check then fails.
        calls.append(1)
        return len(calls) == 1

    monkeypatch.setattr(
        structure_module, "structure_is_valid2", _always_invalid)
    with pytest.raises(ValueError, match="100 attempts"):
        structure_perturb(cscl, distance=0.1)
    assert len(calls) == 102  # 1 assert + 101 loop iterations


# --- IMDStructure -----------------------------------------------------------


def test_imdstructure_to_writes_vac(tmp_path) -> None:
    """The X dummy species is written as Vac in str.out."""
    struct = IMDStructure.from_structure(
        Structure(Lattice.cubic(4.0), ["Li", DummySpecies("X")],
                  [[0, 0, 0], [0.5, 0, 0]]))
    text = struct.to_file(str(tmp_path / "str.out"), fmt="atat")
    assert "Vac" in text
    assert "X0+" not in text


def test_imdstructure_to_atat_without_filename() -> None:
    """fmt="atat" with no filename returns the string without writing."""
    struct = IMDStructure.from_structure(
        Structure(Lattice.cubic(4.0), ["Li", DummySpecies("X")],
                  [[0, 0, 0], [0.5, 0, 0]]))
    text = struct.to(fmt="atat")
    assert "Vac" in text
    assert "X0+" not in text


def test_imdstructure_from_file_reads_vac(tmp_path) -> None:
    """A str.out with Vac is read back as an IMDStructure with X species."""
    struct = IMDStructure.from_structure(
        Structure(Lattice.cubic(4.0), ["Li", DummySpecies("X")],
                  [[0, 0, 0], [0.5, 0, 0]]))
    path = tmp_path / "str.out"
    struct.to_file(str(path), fmt="atat")
    parsed = IMDStructure.from_file(str(path))
    assert isinstance(parsed, IMDStructure)
    assert len(parsed) == 2
    assert parsed[0].species_string == "Li"
    assert isinstance(parsed[1].specie, DummySpecies)
    assert np.allclose(parsed[1].frac_coords, [0.5, 0, 0])


def test_imdstructure_from_structure_preserves_sites(cscl: Structure) -> None:
    """from_structure copies an existing structure into an IMDStructure."""
    converted = IMDStructure.from_structure(cscl)
    assert isinstance(converted, IMDStructure)
    assert converted.species == cscl.species
    assert np.allclose(converted.frac_coords, cscl.frac_coords)


def test_imdstructure_from_file_non_atat(cscl: Structure, tmp_path) -> None:
    """Reading a non-str.out file delegates to pymatgen's reader.

    The parsed result is still cast to IMDStructure (the subclass), not
    returned as a plain Structure.
    """
    struct = IMDStructure.from_structure(cscl)
    path = tmp_path / "POSCAR"
    struct.to_file(str(path), fmt="poscar")
    parsed = IMDStructure.from_file(str(path))
    assert isinstance(parsed, IMDStructure)
    assert parsed.composition == cscl.composition


def test_imdstructure_to_non_atat(cscl: Structure) -> None:
    """A non-atat format is written by pymatgen's POSCAR writer.

    The output is parsed back as POSCAR, which the atat writer would not
    produce, so this also distinguishes the two writer branches.
    """
    struct = IMDStructure.from_structure(cscl)
    out = struct.to(fmt="poscar")
    assert Structure.from_str(out, fmt="poscar").composition == cscl.composition


def test_imdstructure_from_file_sort_and_merge(tmp_path) -> None:
    """The .out reader honours sort and merge_tol."""
    struct = IMDStructure.from_structure(
        Structure(Lattice.cubic(4.0), ["Li", "Na"], [[0, 0, 0], [0.4, 0, 0]]))
    path = tmp_path / "str.out"
    struct.to_file(str(path), fmt="atat")
    parsed = IMDStructure.from_file(str(path), sort=True, merge_tol=0.01)
    assert isinstance(parsed, IMDStructure)
    assert len(parsed) == 2


# --- structure_interpolate2 -------------------------------------------------


def test_structure_interpolate2_basic(cscl: Structure) -> None:
    """Interpolation returns nimages+1 structures including both endpoints."""
    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    images = structure_interpolate2(cscl, moved, nimages=2, center=0.0)
    assert len(images) == 3
    assert all(isinstance(img, Structure) for img in images)
    # With centre alignment disabled, the endpoints are returned verbatim.
    assert np.allclose(images[0].frac_coords, cscl.frac_coords)
    assert np.allclose(images[-1].frac_coords, moved.frac_coords)


def _single_atom(frac_x: float) -> Structure:
    """A single-Li structure whose centre of mass is frac_x along a."""
    return Structure(Lattice.cubic(10.0), ["Li"], [[frac_x, 0, 0]])


def test_structure_interpolate2_center_true_aligns() -> None:
    """center=True always aligns the two centres before interpolation."""
    images = structure_interpolate2(
        _single_atom(0.1), _single_atom(0.4), nimages=2, center=True)
    centers = [np.mean(img.frac_coords, axis=0)[0] for img in images]
    assert all(np.isclose(c, 0.1) for c in centers)


def test_structure_interpolate2_center_float_aligns_when_close() -> None:
    """A float center aligns when the centre-to-centre distance is below it."""
    images = structure_interpolate2(
        _single_atom(0.1), _single_atom(0.4), nimages=2, center=1.0)
    centers = [np.mean(img.frac_coords, axis=0)[0] for img in images]
    assert all(np.isclose(c, 0.1) for c in centers)


def test_structure_interpolate2_center_float_skips_when_far() -> None:
    """A float center below the centre-to-centre distance skips alignment."""
    images = structure_interpolate2(
        _single_atom(0.1), _single_atom(0.4), nimages=2, center=0.001)
    centers = [np.mean(img.frac_coords, axis=0)[0] for img in images]
    assert np.isclose(centers[0], 0.1)
    assert np.isclose(centers[-1], 0.4)


def test_structure_interpolate2_center_falsy_skips() -> None:
    """A falsy center (0.0) skips centre alignment entirely."""
    images = structure_interpolate2(
        _single_atom(0.1), _single_atom(0.4), nimages=2, center=0.0)
    centers = [np.mean(img.frac_coords, axis=0)[0] for img in images]
    assert np.isclose(centers[0], 0.1)
    assert np.isclose(centers[-1], 0.4)


def test_structure_interpolate2_match_first_false_skips_matching(
        cscl: Structure, monkeypatch) -> None:
    """match_first=False interpolates without calling the matcher."""
    def _boom(*args, **kwargs):
        raise AssertionError("get_matched_structure should not be called")
    monkeypatch.setattr(structure_module, "get_matched_structure", _boom)

    moved = cscl.copy()
    moved.translate_sites([0], [0.2, 0, 0], frac_coords=True)
    images = structure_interpolate2(
        cscl, moved, nimages=2, match_first=False, center=0.0)
    assert len(images) == 3


def _colliding_endpoints() -> tuple[Structure, Structure]:
    """Two structures whose naive linear midpoint collides.

    Li and Na swap positions, so the half-way image places both at the
    same point.
    """
    lattice = Lattice.cubic(10.0)
    beg = Structure(lattice, ["Li", "Na"], [[0.0, 0, 0], [0.5, 0, 0]])
    end = Structure(lattice, ["Li", "Na"], [[0.5, 0, 0], [0.0, 0, 0]])
    return beg, end


def test_structure_interpolate2_frac_tol_zero_skips_validation() -> None:
    """frac_tol=0 returns the raw interpolation without the validity check."""
    beg, end = _colliding_endpoints()
    images = structure_interpolate2(beg, end, nimages=2, frac_tol=0.0)
    assert len(images) == 3
    # The midpoint still collides, proving the check was skipped.
    assert not structure_is_valid2(images[1], 0.5)


def test_structure_interpolate2_avoids_collision() -> None:
    """Colliding images are re-spaced so every image is valid."""
    beg, end = _colliding_endpoints()
    images = structure_interpolate2(beg, end, nimages=2, frac_tol=0.5)
    assert len(images) == 3
    assert all(structure_is_valid2(img, 0.5) for img in images)


# --- structure_matches ------------------------------------------------------


def test_structure_matches_warns_on_duplicate(cscl: Structure) -> None:
    """warn=True emits StructureDuplicateWarning with origin paths."""
    struct = cscl.copy()
    struct.properties["origin_path"] = "a"
    known = cscl.copy()
    known.properties["origin_path"] = "b"

    with pytest.warns(StructureDuplicateWarning, match="a and b"):
        assert structure_matches(struct, [known], warn=True) is True


@pytest.mark.parametrize("multithread", [False, True])
def test_structure_matches_finds_duplicate(
        cscl: Structure, multithread) -> None:
    """A duplicate is found in both single- and multi-threaded modes."""
    assert structure_matches(
        cscl, [cscl, cscl * (2, 1, 1)], multithread=multithread) is True


@pytest.mark.parametrize("multithread", [False, True])
def test_structure_matches_no_match(cscl: Structure, multithread) -> None:
    """No match is reported in both single- and multi-threaded modes."""
    other = Structure(Lattice.cubic(5.0), ["Li"], [[0, 0, 0]])
    assert structure_matches(cscl, [other], multithread=multithread) is False


def test_structure_matches_custom_cmp_fun(cscl: Structure) -> None:
    """A caller-supplied comparator is used instead of StructureMatcher."""
    calls: list[tuple] = []

    def cmp(a, b) -> bool:
        calls.append((a, b))
        return True

    assert structure_matches(cscl, [cscl], cmp_fun=cmp) is True
    assert calls == [(cscl, cscl)]


def test_worker_wrapper(cscl: Structure, monkeypatch) -> None:
    """_worker_wrapper unpacks args and calls the global worker."""
    monkeypatch.setattr(
        structure_module, "_GLOBAL_WORKER", lambda a, b: a is b)
    assert structure_module._worker_wrapper((cscl, cscl)) is True
