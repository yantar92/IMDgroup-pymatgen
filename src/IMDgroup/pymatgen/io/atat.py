# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
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


"""This module implements helper functions to work with ATAT.
"""

import numpy as np
from IMDgroup.pymatgen.core.structure import\
    structure_strain, structure_distance, get_matched_structure
from IMDgroup.pymatgen.core.structure import IMDStructure as Structure
from pymatgen.core import DummySpecies


def check_volume_distortion(
        str_before: Structure,
        str_after: Structure,
        # 0.1 is what is done by ATAT in checkcell subroutine
        threshold: float = 0.1) -> bool:
    """Check whether lattice distortion between two structures is acceptable.

    The distortion is the norm of the engineering strain tensor.
    A distortion below ``threshold`` is considered acceptable.
    The default threshold follows ATAT's ``checkcell`` subroutine.

    Args:
        str_before: Initial structure.
        str_after: Deformed structure.
        threshold: Max allowed distortion (default: 0.1).

    Returns:
        bool: True if distortion is below threshold, False otherwise.
    """
    strain = structure_strain(str_before, str_after)
    distortion = np.linalg.norm(strain)
    if distortion < threshold:
        return True
    return False


def check_sublattice_flip(
        str_before: Structure,
        str_after: Structure,
        sublattice: Structure) -> bool:
    """Check whether the relaxed sublattice configuration is preserved.

    Returns True when ``str_after`` occupies the same sublattice
    configuration as ``str_before``, when compared against the
    reference ``sublattice``.

    The species scanned by cluster expansion must be marked with the
    same dummy species name (e.g. X) in all arguments.  For example,
    in an ATAT Li,Vac system, both Li and Vac should be replaced with
    X.

    Args:
        str_before: Structure before relaxation.
        str_after: Structure after relaxation.
        sublattice: Full sublattice with all sites occupied (as in str.in).

    Returns:
        bool: True if the sublattice configuration is preserved.
    """
    # First, scale STR_AFTER lattice to fit STR_BEFORE and SUBLATTICE
    # Assume that STR_BEFORE and SUBLATTICE have the same lattices
    str_after_normalized = str_after.copy()
    str_after_normalized.lattice = str_before.lattice
    dist_relax = structure_distance(str_before, str_after_normalized)

    # Replace all species with X to compare with anonymous sublattice
    dist_sublattice = structure_distance(
        str_after_normalized, sublattice,
        # Compare specie-insensitively
        match_first=True,
        match_species=False)

    if np.isclose(dist_relax, dist_sublattice, rtol=0.001):
        return True
    return False


def fit_sublattice_to_structure(
        sublattice: Structure, structure: Structure) -> Structure:
    """Adjust a reference sublattice to match a relaxed structure.

    Useful for building a new ``str.out`` when the structure has
    flipped away from the initial sublattice guess (see
    :func:`check_sublattice_flip`).  Use ``'X'`` dummy species in place
    of vacancies.

    Args:
        sublattice: Reference sublattice (as from str.in).
        structure: Relaxed structure with possible sublattice flip.

    Returns:
        Structure: Adjusted sublattice matching the relaxed structure.
    """
    structure = structure.copy()
    # Force the lattice to match.  Needed for get_matched_structure.
    structure.lattice = sublattice.lattice
    sublattice2 = get_matched_structure(
        structure, sublattice, match_species=False)
    for idx, site in enumerate(sublattice2):
        if idx < len(structure):
            site.species = structure[idx].species
        else:
            site.species = DummySpecies('X')
    # keep the original site order
    sublattice2 = get_matched_structure(sublattice, sublattice2, match_species=False)
    return sublattice2
