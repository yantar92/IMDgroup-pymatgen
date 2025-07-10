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
    structure_strain, structure_distance
from IMDgroup.pymatgen.core.structure import IMDStructure as Structure


def check_volume_distortion(
        str_before: Structure,
        str_after: Structure,
        # 0.1 is what is done by ATAT in checkcell subroutine
        threshold: float = 0.1) -> bool:
    """Return False when lattice distortion is too large for STR_BEFORE and STR_AFTER.
    The lattice distortion is a norm of engineering strain tensor.
    The distortion is considered "too large" when it is no less than THRESHOLD.
    The default 0.1 threshold is following ATAT source code.
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
    """Check if STR_AFTER flipped its SUBLATTICE sites compared to STR_BEFORE.
    Return True when STR_AFTER occupies the same sublattice configuration as
    STR_BEFORE.  The SUBLATTICE is full sublattice with all the sites
    occupied (as per str.in).

    Note that specie sites that are scanned by cluster expansion must be
    marked with the same specie name in all the arguments.  For
    example, if ATAT is running on Li, Vac system, both Li and Vac species
    should be replaced with, say X dummy specie.
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
