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

from IMDgroup.pymatgen.core.structure import (
    IMDStructure,
    get_matched_structure,
    get_supercell_size,
    merge_structures,
    reduce_supercell,
    structure_diff,
    structure_distance,
    structure_is_valid2,
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


# --- IMDStructure str.out round trip ----------------------------------------


def test_imdstructure_to_writes_vac(tmp_path) -> None:
    """The X dummy species is written as Vac in str.out."""
    struct = IMDStructure.from_structure(
        Structure(Lattice.cubic(4.0), ["Li", DummySpecies("X")],
                  [[0, 0, 0], [0.5, 0, 0]]))
    text = struct.to_file(str(tmp_path / "str.out"), fmt="atat")
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
