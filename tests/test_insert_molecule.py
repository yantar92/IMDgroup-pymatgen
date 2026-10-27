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


def _write_molecule(tmp_path: Path, molecule: Molecule) -> Path:
    """Write a Molecule to an XYZ file with pymatgen and return its path."""
    path = tmp_path / "molecule.xyz"
    molecule.to(filename=str(path), fmt="xyz")
    return path


def _write_structure(tmp_path: Path, structure: Structure) -> Path:
    """Write a Structure to a POSCAR file and return its path."""
    path = tmp_path / "host.POSCAR"
    structure.to(filename=str(path), fmt="poscar")
    return path


def _cubic_supercell() -> Structure:
    """A 2x1x1 supercell of a simple-cubic primitive cell."""
    primitive = Structure(
        Lattice.cubic(4.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    return primitive * (2, 1, 1)


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
    """A negative step_noise requests uniform random sampling over the cell."""
    structure = Structure(
        Lattice.cubic(4.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0, step_noise=-1000)
    grid = transformer._get_site_grid(structure)
    assert len(grid) == 1000
    coords = np.array(grid)
    assert np.all(coords >= 0.0)
    assert np.all(coords < 1.0)
    # Uniform on [0, 1): mean 0.5, std sqrt(1 / 12).
    assert np.allclose(np.mean(coords), 0.5, atol=0.02)
    assert np.allclose(np.std(coords), np.sqrt(1.0 / 12.0), atol=0.02)


def test_get_site_grid_positive_step_noise_statistics() -> None:
    """A positive step_noise adds Gaussian jitter with the expected std."""
    structure = Structure(
        Lattice.cubic(4.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    clean = np.array(
        InsertMoleculeTransformation("Na", step=1.0)._get_site_grid(structure))
    np.random.seed(1234)
    noisy = np.array(
        InsertMoleculeTransformation(
            "Na", step=1.0, step_noise=0.5)._get_site_grid(structure))
    assert noisy.shape == clean.shape

    # The noise added per coordinate is
    # ``step_noise * (step / a) * standard_normal()``.
    # With step=1.0 and a=4.0, that is 0.5 * 0.25 = 0.125.
    noise = noisy - clean
    expected_std = 0.5 * (1.0 / 4.0)
    assert np.allclose(np.mean(noise), 0.0, atol=0.05)
    assert abs(np.std(noise) - expected_std) < 0.03


def test_get_site_grid_reduces_supercell() -> None:
    """reduce_supercell=True limits the grid to the primitive cell volume."""
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, reduce_supercell=True)
    grid = np.array(transformer._get_site_grid(_cubic_supercell()))

    # The primitive cell spans half the supercell along a, so the scan is
    # confined to 0 <= x < 0.5 and yields 4 x 4 x 4 points.
    assert grid.shape == (4 * 4 * 4, 3)
    assert np.all(grid[:, 0] < 0.5)


def test_get_site_grid_keeps_full_supercell_when_not_reduced() -> None:
    """reduce_supercell=False scans the full supercell volume."""
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, reduce_supercell=False)
    grid = np.array(transformer._get_site_grid(_cubic_supercell()))

    assert grid.shape == (8 * 4 * 4, 3)
    assert np.all(grid[:, 0] < 1.0)
    assert np.any(grid[:, 0] > 0.5)


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


# --- molecule file input -----------------------------------------------------


def test_init_parses_molecule_file(tmp_path: Path) -> None:
    """A path to a molecule file is parsed into a multi-atom Molecule."""
    water = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    mol_file = _write_molecule(tmp_path, water)
    transformer = InsertMoleculeTransformation(str(mol_file), step=1.0)
    assert isinstance(transformer.molecule, Molecule)
    assert len(transformer.molecule) == 3
    assert [site.specie.symbol for site in transformer.molecule] == \
        ["O", "H", "H"]


def test_molecule_rotation_end_to_end(tmp_path: Path) -> None:
    """A real molecule file flows through __init__ -> all_inserts with rotation."""
    water = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    mol_file = _write_molecule(tmp_path, water)
    transformer = InsertMoleculeTransformation(
        str(mol_file), step=4.0, anglestep=np.pi / 2, matcher=None)

    # anglestep populates the Euler-angle grid for the multi-atom molecule.
    assert transformer._candidate_angles
    assert len(transformer._candidate_angles) == 8

    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    inserts = transformer.all_inserts(host)

    # matcher=None skips the StructureMatcher dedup, but the molecule
    # proximity check (known_inserts) still collapses redundant rotations:
    # 7 valid grid positions x 7 distinct Euler orientations = 49.  The 8
    # angle triplets collapse to 7 because beta=0 makes the two z-rotations
    # (alpha and gamma) degenerate.
    assert len(inserts) == 49
    orientations = set()
    for structure in inserts:
        assert len(structure) == 4  # host C + O + 2 H
        o_sites = [site for site in structure if site.specie.symbol == "O"]
        h_sites = [site for site in structure if site.specie.symbol == "H"]
        assert len(o_sites) == 1 and len(h_sites) == 2
        assert o_sites[0].label == "insert-O0"
        assert {site.label for site in h_sites} == {"insert-H1", "insert-H2"}
        # Relative H coordinates (wrapped) characterise the orientation.
        o = o_sites[0]
        rel = tuple(sorted(
            tuple(np.round(
                (np.array(h.frac_coords) - np.array(o.frac_coords) + 0.5)
                % 1.0 - 0.5, 3))
            for h in h_sites
        ))
        orientations.add(rel)
    # Euler rotations produce 7 distinct orientations (one of the 8 triplets
    # is degenerate with another).
    assert len(orientations) == 7


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
        same, same.copy()) is False
    assert transformer._check_molecule_proximity(same, shifted) is True


def test_check_molecule_proximity_single_atom() -> None:
    """A single-atom placement is distinct only when displaced."""
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    single = Structure(
        Lattice.cubic(10.0), ["Na"], [[0, 0, 0]], coords_are_cartesian=True)
    shifted = Structure(
        Lattice.cubic(10.0), ["Na"], [[5.0, 0, 0]], coords_are_cartesian=True)
    assert transformer._check_molecule_proximity(
        single, single.copy()) is False
    assert transformer._check_molecule_proximity(
        single, shifted) is True


def test_check_molecule_proximity_uses_proximity_threshold() -> None:
    """Distinctness is gated by ``proximity_threshold * (r1 + r2)``.

    A molecule compared to itself is never distinct, regardless of its
    internal bond length.  Distinctness only arises when an atom moves
    farther than the scaled sum of radii.
    """
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, proximity_threshold=0.75)
    base = Structure(
        Lattice.cubic(10.0), ["Na", "Na"], [[0, 0, 0], [3.0, 0, 0]],
        coords_are_cartesian=True)

    # Na radius is 1.8 A, so the threshold is 0.75 * (1.8 + 1.8) = 2.7 A.
    threshold = 0.75 * (1.8 + 1.8)
    near = Structure(
        Lattice.cubic(10.0), ["Na", "Na"],
        [[threshold - 0.5, 0, 0], [3.0 + threshold - 0.5, 0, 0]],
        coords_are_cartesian=True)
    far = Structure(
        Lattice.cubic(10.0), ["Na", "Na"],
        [[threshold + 0.5, 0, 0], [3.0 + threshold + 0.5, 0, 0]],
        coords_are_cartesian=True)

    assert transformer._check_molecule_proximity(
        base, base.copy()) is False
    assert transformer._check_molecule_proximity(base, near) is False
    assert transformer._check_molecule_proximity(base, far) is True


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
    # The host's own selective dynamics flags are preserved.
    assert list(result[1].properties["selective_dynamics"]) == \
        [False, False, False]


def test_insert_molecule_defaults_free_flags_for_host_with_sd() -> None:
    """selective_dynamics=None defaults inserted atoms to free flags."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    host.add_site_property("selective_dynamics", [[False, False, False]])
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    result = transformer._insert_molecule(host.copy(), [0.5, 0.5, 0.5])
    assert result is not None
    na_site = [site for site in result if site.specie.symbol == "Na"][0]
    assert list(na_site.properties["selective_dynamics"]) == \
        [True, True, True]
    c_site = [site for site in result if site.specie.symbol == "C"][0]
    assert list(c_site.properties["selective_dynamics"]) == \
        [False, False, False]


def test_insert_molecule_adds_free_flags_to_host_without_sd() -> None:
    """Explicit selective_dynamics adds free flags to a host lacking SD."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, selective_dynamics=[True, True, False])
    result = transformer._insert_molecule(host.copy(), [0.5, 0.5, 0.5])
    assert result is not None
    na_site = [site for site in result if site.specie.symbol == "Na"][0]
    assert list(na_site.properties["selective_dynamics"]) == \
        [True, True, False]
    c_site = [site for site in result if site.specie.symbol == "C"][0]
    assert list(c_site.properties["selective_dynamics"]) == \
        [True, True, True]


def test_insert_molecule_all_free_flags_on_host_without_sd_is_noop() -> None:
    """An all-free flag on a constraint-free host writes no constraints."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation(
        "Na", step=1.0, selective_dynamics=[True, True, True])
    result = transformer._insert_molecule(host.copy(), [0.5, 0.5, 0.5])
    assert result is not None
    # No site carries a selective_dynamics property: the all-free flags
    # were a no-op and must not produce an all-"T T T" POSCAR.
    for site in result:
        assert "selective_dynamics" not in site.properties


def test_insert_molecule_rejects_too_close() -> None:
    """An insertion that clashes with the host returns None."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0)
    assert transformer._insert_molecule(host.copy(), [0.1, 0.1, 0.1]) is None


def test_insert_molecule_skips_label_when_none() -> None:
    """label=None leaves the inserted atom with its default species label."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=1.0, label=None)
    result = transformer._insert_molecule(host.copy(), [0.5, 0.5, 0.5])
    assert result is not None
    na_site = [site for site in result if site.specie.symbol == "Na"][0]
    assert na_site.label == "Na"


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


def test_all_inserts_accepts_structure_path(tmp_path: Path) -> None:
    """A path to a structure file is parsed by all_inserts."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    path = _write_structure(tmp_path, host)
    inserts = InsertMoleculeTransformation("Na", step=4.0).all_inserts(
        str(path))
    assert inserts
    assert all(isinstance(s, Structure) for s in inserts)
    assert all(len(s) == 2 for s in inserts)


def test_apply_transformation_accepts_structure_path(tmp_path: Path) -> None:
    """A path to a structure file is parsed by apply_transformation."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    path = _write_structure(tmp_path, host)
    result = InsertMoleculeTransformation("Na", step=4.0).apply_transformation(
        str(path))
    assert isinstance(result, Structure)
    assert len(result) == 2


def test_get_all_molecule_inserts_wrapper() -> None:
    """The convenience wrapper returns a list of Structures."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    inserts = get_all_molecule_inserts("Na", host, step=1.0)
    assert inserts
    assert all(isinstance(s, Structure) for s in inserts)


def test_get_all_molecule_inserts_forwards_label_and_limit() -> None:
    """The wrapper passes label and limit through to the transformer."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    inserts = get_all_molecule_inserts(
        "Na", host, step=4.0, label="myinsert", limit=2)
    assert 1 <= len(inserts) <= 2
    for structure in inserts:
        na_sites = [site for site in structure if site.specie.symbol == "Na"]
        assert len(na_sites) == 1
        assert na_sites[0].label == "myinsert-Na0"


def test_matcher_none_skips_symmetry_deduplication() -> None:
    """matcher=None keeps symmetry-equivalent inserts."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=4.0, matcher=None)
    inserts = transformer.all_inserts(host)

    # step=4.0 on an 8 A cube yields 8 grid points; the one at the origin
    # overlaps the host C and is rejected, leaving 7.  Without a
    # StructureMatcher, none of the symmetry-equivalent positions collapse.
    assert len(inserts) == 7
    positions = sorted(
        tuple(float(c) for c in site.frac_coords)
        for structure in inserts
        for site in structure
        if site.specie.symbol == "Na"
    )
    assert positions == [
        (0.0, 0.0, 0.5),
        (0.0, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 0.5),
    ]


def test_multithread_matches_serial_dedup() -> None:
    """The multithreaded StructureMatcher path agrees with the serial one."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    serial = InsertMoleculeTransformation(
        "Na", step=4.0).all_inserts(host, limit=3)
    parallel = InsertMoleculeTransformation(
        "Na", step=4.0, multithread=True).all_inserts(host, limit=3)

    def positions(structures: list[Structure]) -> list[list[tuple]]:
        return [
            sorted(
                tuple(float(c) for c in site.frac_coords)
                for site in structure
                if site.specie.symbol == "Na"
            )
            for structure in structures
        ]

    assert positions(parallel) == positions(serial)


def test_negative_limit_randomly_samples_subset() -> None:
    """A negative limit returns a random subset of the candidate inserts."""
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation("Na", step=4.0, matcher=None)
    valid = {
        tuple(float(c) for c in coords)
        for coords in transformer._get_site_grid(host)
        if not np.allclose(coords, [0.0, 0.0, 0.0])
    }

    np.random.seed(0)
    inserts = transformer.all_inserts(host, limit=-3)
    assert len(inserts) == 3
    positions = [
        tuple(float(c) for c in site.frac_coords)
        for structure in inserts
        for site in structure
        if site.specie.symbol == "Na"
    ]
    # Every sampled position is a valid grid point, with no repeats.
    assert set(positions) <= valid
    assert len(positions) == len(set(positions)) == 3


def test_negative_limit_shuffles_angle_grid(monkeypatch) -> None:
    """A negative limit shuffles the molecule's Euler-angle grid."""
    molecule = Molecule(["Na", "Na"], [[0, 0, 0], [3.0, 0, 0]])
    host = Structure(
        Lattice.cubic(8.0), ["C"], [[0, 0, 0]], coords_are_cartesian=True)
    transformer = InsertMoleculeTransformation(
        molecule, step=4.0, anglestep=np.pi / 2, matcher=None)

    shuffled = []
    real_shuffle = np.random.shuffle

    def spy_shuffle(array) -> None:
        shuffled.append(array)
        real_shuffle(array)

    monkeypatch.setattr(np.random, "shuffle", spy_shuffle)

    transformer.all_inserts(host, limit=-2)

    assert any(array is transformer._candidate_angles for array in shuffled)


def test_is_one_to_many() -> None:
    """The transformation is one-to-many."""
    assert InsertMoleculeTransformation("Na", step=1.0).is_one_to_many is True
