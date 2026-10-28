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


"""Tests for IMDgroup.pymatgen.io.atat."""

from __future__ import annotations

import numpy as np
from pymatgen.core import DummySpecies, Lattice, Structure

from IMDgroup.pymatgen.io.atat import (
    check_sublattice_flip,
    check_volume_distortion,
    fit_sublattice_to_structure,
)


def _sublattice() -> Structure:
    """A two-site Li/X sublattice used across the ATAT tests."""
    return Structure(
        Lattice.cubic(4.0), ["Li", DummySpecies("X")],
        [[0, 0, 0], [0.5, 0, 0]])


def _scale_lattice(
        structure: Structure, factor: float, axis: int = 0) -> Structure:
    """Return a copy of ``structure`` stretched along one lattice axis."""
    matrix = structure.lattice.matrix.copy()
    matrix[axis, axis] *= factor
    return Structure(Lattice(matrix), structure.species, structure.frac_coords)


# --- check_volume_distortion -------------------------------------------------


def test_check_volume_distortion_accepts_small_strain() -> None:
    """A small lattice distortion is below the default threshold."""
    sublattice = _sublattice()
    assert check_volume_distortion(sublattice, _scale_lattice(sublattice, 1.02))


def test_check_volume_distortion_rejects_large_strain() -> None:
    """A large lattice distortion exceeds the default threshold."""
    sublattice = _sublattice()
    assert not check_volume_distortion(
        sublattice, _scale_lattice(sublattice, 1.30))


def test_check_volume_distortion_threshold_override() -> None:
    """A custom threshold changes the accept/reject decision."""
    sublattice = _sublattice()
    deformed = _scale_lattice(sublattice, 1.30)
    assert check_volume_distortion(sublattice, deformed, threshold=0.5)


# --- check_sublattice_flip ---------------------------------------------------


def test_check_sublattice_flip_preserved() -> None:
    """A slightly relaxed structure keeps its sublattice configuration."""
    sublattice = _sublattice()
    after = Structure(
        sublattice.lattice, ["Li", DummySpecies("X")],
        [[0.02, 0, 0], [0.5, 0.01, 0]])
    assert check_sublattice_flip(sublattice, after, sublattice)


def test_check_sublattice_flip_detected() -> None:
    """Swapping Li and the vacancy is detected as a flip."""
    sublattice = _sublattice()
    after = Structure(
        sublattice.lattice, ["Li", DummySpecies("X")],
        [[0.5, 0, 0], [0.0, 0, 0]])
    assert not check_sublattice_flip(sublattice, after, sublattice)


# --- fit_sublattice_to_structure --------------------------------------------


def test_fit_sublattice_to_structure_reassigns_species() -> None:
    """The fitted sublattice reflects the relaxed species configuration."""
    sublattice = _sublattice()
    flipped = Structure(
        sublattice.lattice, ["Li", DummySpecies("X")],
        [[0.5, 0, 0], [0.0, 0, 0]])
    fitted = fit_sublattice_to_structure(sublattice, flipped)
    assert isinstance(fitted[0].specie, DummySpecies)
    assert np.allclose(fitted[0].frac_coords, [0, 0, 0])
    assert fitted[1].species_string == "Li"
    assert np.allclose(fitted[1].frac_coords, [0.5, 0, 0])


def test_fit_sublattice_fills_missing_sites_with_dummy() -> None:
    """A relaxed structure with a vacancy gets the empty site filled with X."""
    sublattice = Structure(
        Lattice.cubic(4.0),
        ["Li", DummySpecies("X"), DummySpecies("X")],
        [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]])
    # Relaxed structure is missing the (0, 0.5, 0) site entirely.
    structure = Structure(
        Lattice.cubic(4.0), ["Li", DummySpecies("X")],
        [[0, 0, 0], [0.5, 0, 0]])

    fitted = fit_sublattice_to_structure(sublattice, structure)

    assert len(fitted) == 3
    assert fitted[0].species_string == "Li"
    assert np.allclose(fitted[0].frac_coords, [0, 0, 0])
    # The missing site is restored as an X dummy species.
    assert isinstance(fitted[2].specie, DummySpecies)
    assert np.allclose(fitted[2].frac_coords, [0, 0.5, 0])
