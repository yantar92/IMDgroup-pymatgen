"""Tests for IMDgroup.pymatgen.transformations.insert_molecule.

These exercise :class:`InsertMoleculeTransformation` and its helpers,
focusing on the values we compute and the decisions we make around the
pymatgen calls (see the testing principle in tests/conftest.py).

The graphite matrix fixture (a 63-atom defective carbon cell) comes
from a production Na-insertion scan.  A loosened ``step`` of 0.5 A
(vs 0.1 A in production) keeps the integration test fast while still
exercising the full grid scan + symmetry dedup path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Molecule, Structure

from IMDgroup.pymatgen.core.structure import structure_is_valid2
from IMDgroup.pymatgen.transformations.insert_molecule import (
    InsertMoleculeTransformation,
    get_all_molecule_inserts,
)

MATRIX_POSCAR = (
    Path(__file__).parent / "fixtures" / "insert_molecule" / "matrix.fixC.POSCAR"
)


@pytest.fixture
def graphite_matrix() -> Structure:
    """The defective 63-atom graphite supercell used for Na insertion."""
    return Structure.from_file(str(MATRIX_POSCAR))


# --- _check_proximity -------------------------------------------------------


def test_check_proximity_detects_close_and_far() -> None:
    """A site close to another is rejected; a distant one is accepted."""
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, proximity_threshold=0.75)
    far = Structure(
        Lattice.cubic(8.0), ["C", "C"], [[0, 0, 0], [4, 0, 0]],
        coords_are_cartesian=True)
    close = Structure(
        Lattice.cubic(8.0), ["C", "C"], [[0, 0, 0], [0.5, 0, 0]],
        coords_are_cartesian=True)
    assert transformer._check_proximity(far, 0) is True
    assert transformer._check_proximity(close, 0) is False


def test_check_proximity_detects_overlapping_atoms() -> None:
    """Two same-species atoms on one site should be reported as too close."""
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, proximity_threshold=0.75)
    overlap = Structure(
        Lattice.cubic(8.0), ["Na", "Na"], [[0, 0, 0], [0, 0, 0]],
        coords_are_cartesian=True)
    assert transformer._check_proximity(overlap, 0) is False


# --- _get_site_grid ---------------------------------------------------------


def test_get_site_grid_spacing() -> None:
    """The grid is a regular lattice in fractional coordinates."""
    structure = Structure(
        Lattice.cubic(4.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    grid = transformer._get_site_grid(structure)
    assert len(grid) == 4 ** 3
    coords = np.array(grid)
    for axis in range(3):
        assert np.allclose(
            np.unique(coords[:, axis]), [0.0, 0.25, 0.5, 0.75])


def test_get_site_grid_random_sampling() -> None:
    """A negative step_noise requests fully random sampling."""
    structure = Structure(
        Lattice.cubic(4.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0, step_noise=-5)
    grid = transformer._get_site_grid(structure)
    assert len(grid) == 5
    coords = np.array(grid)
    assert np.all(coords >= 0.0)
    assert np.all(coords < 1.0)


# --- _get_angle_grid / rotate_molecule_euler --------------------------------


def test_single_atom_rejects_anglestep() -> None:
    """A single atom cannot be rotated."""
    with pytest.raises(AssertionError, match="Cannot rotate"):
        InsertMoleculeTransformation("Na", step=1.0, anglestep=1.0)


def test_get_angle_grid() -> None:
    """Angle triplets are Euler angles within [0, 2*pi)."""
    molecule = Molecule(["Na", "Na"], [[0, 0, 0], [3.0, 0, 0]])
    transformer = InsertMoleculeTransformation(
        molecule, step=1.0, anglestep=1.0)
    angles = transformer._get_angle_grid()
    assert angles
    arr = np.array(angles)
    assert arr.shape[1] == 3
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 2 * np.pi)


def test_rotate_molecule_euler_preserves_bond_and_anchor() -> None:
    """Rotation keeps the anchor atom fixed and the bond length intact."""
    molecule = Molecule(["Na", "Na"], [[0, 0, 0], [3.0, 0, 0]])
    transformer = InsertMoleculeTransformation(
        molecule, step=1.0, anglestep=1.0)
    rotated = transformer.rotate_molecule_euler([1.0, 0.5, 0.3])
    assert np.isclose(rotated.get_distance(0, 1), 3.0)
    assert np.allclose(rotated[0].coords, [0.0, 0.0, 0.0])
    assert not np.allclose(rotated[1].coords, [3.0, 0.0, 0.0])


# --- _check_molecule_proximity -----------------------------------------------


def test_check_molecule_proximity_detects_identical_vs_shifted() -> None:
    """An unchanged molecule is not distinct; a displaced one is."""
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, proximity_threshold=0.75)
    molecule = Molecule(["Na", "Na"], [[0, 0, 0], [2.0, 0, 0]])
    same = Structure(
        Lattice.cubic(10.0), molecule.species, molecule.cart_coords,
        coords_are_cartesian=True)
    shifted = Structure(
        Lattice.cubic(10.0), molecule.species,
        [[5.0, 0, 0], [7.0, 0, 0]], coords_are_cartesian=True)
    assert transformer._check_molecule_proximity(
        same, same.copy(), 5.0) is False
    assert transformer._check_molecule_proximity(same, shifted, 5.0) is True


# --- _insert_molecule ---------------------------------------------------------


def test_insert_molecule_places_and_labels() -> None:
    """The atom lands at the target site with label and selective dynamics."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    host.add_site_property("selective_dynamics", [[False, False, False]])
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, label="insert", selective_dynamics=[True, True, True])
    result = transformer._insert_molecule(host.copy(), [0.5, 0.5, 0.5])
    assert result is not None
    assert len(result) == 2
    assert result[0].species_string == "Na"
    assert np.allclose(result[0].frac_coords, [0.5, 0.5, 0.5])
    assert result[0].label == "insert-Na0"
    assert list(result[0].properties["selective_dynamics"]) == \
        [True, True, True]
    assert result[1].species_string == "C"


def test_insert_molecule_rejects_too_close() -> None:
    """An insertion that clashes with the host returns None."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    assert transformer._insert_molecule(host.copy(), [0.1, 0.1, 0.1]) is None


# --- all_inserts / apply_transformation / wrapper -----------------------------


@pytest.mark.heavy
def test_all_inserts_into_graphite_matrix(graphite_matrix: Structure) -> None:
    """A loosened Na scan yields the expected valid, distinct structures."""
    transformer = InsertMoleculeTransformation(
        "Na", step=0.5, proximity_threshold=0.75,
        selective_dynamics=[True, True, True])
    inserts = transformer.all_inserts(graphite_matrix, limit=None)

    # 14 distinct sites at step=0.5 (production found 13 at step=0.1).
    assert len(inserts) == 14

    for structure in inserts:
        assert structure_is_valid2(structure, 0.75)
        na_sites = [site for site in structure if site.specie.symbol == "Na"]
        assert len(na_sites) == 1
        assert len(structure) == 64
        assert na_sites[0].label == "insert-Na0"
        assert list(na_sites[0].properties["selective_dynamics"]) == \
            [True, True, True]

    # Distinctness: no two inserts are symmetry-equivalent.
    matcher = StructureMatcher(attempt_supercell=True, scale=False)
    for i in range(len(inserts)):
        for j in range(i + 1, len(inserts)):
            assert not matcher.fit(inserts[i], inserts[j])


@pytest.mark.heavy
def test_sequential_insertion_has_no_overlaps(
        graphite_matrix: Structure) -> None:
    """Inserting a second Na must not overlap the first."""
    transformer = InsertMoleculeTransformation(
        "Na", step=0.5, proximity_threshold=0.75,
        selective_dynamics=[True, True, True])
    first = transformer.all_inserts(graphite_matrix, limit=1)[0]

    second_transformer = InsertMoleculeTransformation(
        "Na", step=0.5, proximity_threshold=0.75,
        selective_dynamics=[True, True, True])
    second = second_transformer.all_inserts(first, limit=None)

    assert all(structure_is_valid2(s, 0.75) for s in second)
    for structure in second:
        na_sites = [site for site in structure if site.specie.symbol == "Na"]
        assert len(na_sites) == 2
        assert not np.allclose(
            na_sites[0].frac_coords, na_sites[1].frac_coords)


def test_apply_transformation_returns_single_structure() -> None:
    """apply_transformation returns one inserted structure by default."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    result = transformer.apply_transformation(host)
    assert isinstance(result, Structure)
    assert len(result) == 2


def test_apply_transformation_ranked_list() -> None:
    """return_ranked_list returns a list of dictionaries."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    results = transformer.apply_transformation(host, return_ranked_list=3)
    # Returns up to ``return_ranked_list`` structures, capped by how many
    # distinct insertion sites exist (2 for a single C in an 8 A cube).
    assert 1 <= len(results) <= 3
    assert all("structure" in d for d in results)


def test_get_all_molecule_inserts_wrapper() -> None:
    """The convenience wrapper returns a list of Structures."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    inserts = get_all_molecule_inserts("Na", host, step=1.0)
    assert inserts
    assert all(isinstance(s, Structure) for s in inserts)


def test_is_one_to_many() -> None:
    """The transformation is one-to-many."""
    assert InsertMoleculeTransformation("Na", step=1.0).is_one_to_many is True
