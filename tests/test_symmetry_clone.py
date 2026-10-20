"""Tests for IMDgroup.pymatgen.transformations.symmetry_clone.

These transformations generate symmetry-equivalent clones of sites, so
the tests focus on the decisions we make around pymatgen's SymmOp and
SpacegroupAnalyzer calls (see tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Element, Lattice, Structure, SymmOp

from IMDgroup.pymatgen.transformations.symmetry_clone import (
    SymmetryCloneTransformation,
    SymmetryFillTransformation,
    apply_operation_keep_lattice,
)


IDENTITY = SymmOp.from_xyz_str("x,y,z")
HALF_X = SymmOp.from_xyz_str("x+1/2,y,z")


@pytest.fixture
def host() -> Structure:
    """Single atom at the origin of a cubic cell: full cubic symmetry."""
    return Structure(Lattice.cubic(4.0), ["Na"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def interstitial() -> Structure:
    """Host atom plus one interstitial atom at a general position."""
    return Structure(
        Lattice.cubic(4.0),
        ["Na", "Na"],
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
    )


# --- SymmetryFillTransformation --------------------------------------------

def test_fill_constructor_from_structure(host) -> None:
    """A Structure input derives operations via SpacegroupAnalyzer."""
    fill = SymmetryFillTransformation(host, ["Na"])
    assert len(fill.sym_operations) == 48
    assert fill.element_set == {Element("Na")}


def test_fill_constructor_from_op_list() -> None:
    """A list of SymmOp is used as-is."""
    fill = SymmetryFillTransformation([IDENTITY, HALF_X], ["Na"])
    assert fill.sym_operations == [IDENTITY, HALF_X]


def test_fill_constructor_rejects_invalid() -> None:
    """A non-Structure, non-list input raises ValueError."""
    with pytest.raises(ValueError):
        SymmetryFillTransformation("not a structure or op list", ["Na"])


def test_fill_clones_element_and_sets_symop() -> None:
    """Selected sites are cloned and each site records its origin op."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.0, 0.0, 0.0]])
    fill = SymmetryFillTransformation([IDENTITY, HALF_X], ["Na"])
    result = fill.apply_transformation(structure)

    assert len(result) == 2
    by_coord = {tuple(np.round(site.frac_coords, 6)): site for site in result}
    assert set(by_coord) == {(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)}
    assert by_coord[(0.0, 0.0, 0.0)].properties["symop"] == IDENTITY
    assert by_coord[(0.5, 0.0, 0.0)].properties["symop"] == HALF_X


def test_fill_skips_overlapping_clone() -> None:
    """The identity op maps a site onto itself and is skipped."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.0, 0.0, 0.0]])
    fill = SymmetryFillTransformation([IDENTITY], ["Na"])
    assert len(fill.apply_transformation(structure)) == 1


def test_fill_only_clones_selected_elements() -> None:
    """Sites of non-selected elements are left untouched."""
    structure = Structure(
        Lattice.cubic(4.0),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    fill = SymmetryFillTransformation([HALF_X], ["Na"])
    result = fill.apply_transformation(structure)

    symbols = [site.specie.symbol for site in result]
    assert symbols.count("Na") == 2  # original + clone
    assert symbols.count("Cl") == 1  # untouched


def test_fill_accepts_structure_path(tmp_path) -> None:
    """A path string is read through Structure.from_file."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.0, 0.0, 0.0]])
    path = tmp_path / "struct.cif"
    structure.to(filename=str(path))

    fill = SymmetryFillTransformation([IDENTITY, HALF_X], ["Na"])
    assert len(fill.apply_transformation(str(path))) == 2


# --- apply_operation_keep_lattice ------------------------------------------

def test_apply_operation_keep_lattice_preserves_lattice() -> None:
    """The lattice vectors are unchanged after the operation."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.1, 0.2, 0.3]])
    result = apply_operation_keep_lattice(structure, HALF_X)
    assert result.lattice == structure.lattice
    assert len(result) == len(structure)


def test_apply_operation_keep_lattice_normalizes_coords() -> None:
    """Fractional coordinates are wrapped into the unit cell."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.6, 0.2, 0.3]])
    result = apply_operation_keep_lattice(structure, HALF_X)
    # 0.6 + 0.5 -> 1.1, wrapped to 0.1
    assert np.allclose(result.frac_coords[0], [0.1, 0.2, 0.3])


def test_apply_operation_keep_lattice_sets_symop() -> None:
    """The applied op is recorded in the structure properties."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.1, 0.2, 0.3]])
    result = apply_operation_keep_lattice(structure, HALF_X)
    assert result.properties["symop"] == HALF_X


# --- SymmetryCloneTransformation -------------------------------------------

def test_clone_constructor_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        SymmetryCloneTransformation("not a structure or op list")


def test_clone_enumerates_distinct_positions(host, interstitial) -> None:
    """48 cubic ops map the general-position interstitial to 48 sites."""
    trans = SymmetryCloneTransformation(host, tol=0.01)
    clones = trans.get_all_clones(interstitial, progress_bar=False)

    assert len(clones) == 48
    interstitial_coords = {
        tuple(np.round(clone.frac_coords[1], 4)) for clone in clones
    }
    assert len(interstitial_coords) == 48


def test_clone_sorted_by_distance(host, interstitial) -> None:
    """Clones are sorted so the identity op (distance 0) comes first."""
    trans = SymmetryCloneTransformation(host, tol=0.01)
    clones = trans.get_all_clones(interstitial, progress_bar=False)
    assert clones[0].properties["symop"] == IDENTITY


def test_clone_dedups_duplicate_ops() -> None:
    """Two identical operations collapse to a single clone."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.1, 0.2, 0.3]])
    trans = SymmetryCloneTransformation([IDENTITY, IDENTITY], tol=0.01)
    clones = trans.get_all_clones(structure, progress_bar=False)
    assert len(clones) == 1


def test_clone_keeps_distinct_ops() -> None:
    """Distinct operations produce distinct clones."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.1, 0.2, 0.3]])
    trans = SymmetryCloneTransformation([IDENTITY, HALF_X], tol=0.01)
    clones = trans.get_all_clones(structure, progress_bar=False)
    assert len(clones) == 2


class _AcceptFirstThree:
    """Filter that keeps only the first three candidate clones."""

    def filter(self, clone, clones):
        return len(clones) < 3


class _TruncateToTwo:
    """Filter whose final_filter keeps only the first two clones."""

    def __init__(self):
        self.filter_calls = 0

    def filter(self, clone, clones):
        self.filter_calls += 1
        return True

    def final_filter(self, clones):
        return clones[:2]


def test_clone_filter_gates_candidates(host, interstitial) -> None:
    """filter() rejecting candidates caps the number of clones."""
    trans = SymmetryCloneTransformation(
        host, filter_cls=_AcceptFirstThree(), tol=0.01)
    clones = trans.get_all_clones(interstitial, progress_bar=False)
    assert len(clones) == 3


def test_clone_final_filter_truncates(host, interstitial) -> None:
    """final_filter() is applied after all candidates are collected."""
    filt = _TruncateToTwo()
    trans = SymmetryCloneTransformation(host, filter_cls=filt, tol=0.01)
    clones = trans.get_all_clones(interstitial, progress_bar=False)
    assert len(clones) == 2
    assert filt.filter_calls == 48


def test_clone_apply_returns_first_clone(host, interstitial) -> None:
    """Single-mode apply returns the first (identity) clone."""
    trans = SymmetryCloneTransformation(host, tol=0.01)
    single = trans.apply_transformation(interstitial)
    assert single.properties["symop"] == IDENTITY


def test_clone_apply_ranked_list(host, interstitial) -> None:
    """return_ranked_list produces a list of structure dicts."""
    trans = SymmetryCloneTransformation(host, tol=0.01)
    ranked = trans.apply_transformation(interstitial, return_ranked_list=3)
    assert isinstance(ranked, list)
    assert len(ranked) == 3
    assert all(set(item) == {"structure"} for item in ranked)


def test_clone_is_one_to_many(host) -> None:
    assert SymmetryCloneTransformation(host).is_one_to_many is True


def test_clone_apply_accepts_structure_path(tmp_path) -> None:
    """A path string is read through Structure.from_file."""
    structure = Structure(Lattice.cubic(4.0), ["Na"], [[0.1, 0.2, 0.3]])
    path = tmp_path / "struct.cif"
    structure.to(filename=str(path))

    trans = SymmetryCloneTransformation([IDENTITY, HALF_X], tol=0.01)
    single = trans.apply_transformation(str(path))
    assert single.properties["symop"] == IDENTITY
