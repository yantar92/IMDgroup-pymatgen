"""Extension for pymatgen.core.structure
"""
import logging
import numpy as np
from pymatgen.core import Structure
from pymatgen.util.coord import pbc_shortest_vectors

logger = logging.getLogger(__name__)


def merge_structures(
        structs: list[Structure],
        tol: float = 0.01,
        ) -> Structure:
    """Merge all STRUCTS into a single Structure.
    Return structure.
    tol is tolerance passed to Structure.merge_sites method.
    """
    assert len(structs) > 0
    for struct in structs[1:]:
        assert struct.lattice == structs[0].lattice

    merged = structs[0].copy()
    sites_before = sum(len(s) for s in structs)
    for struct in structs[1:]:
        for site in struct:
            merged.append(
                site.species,
                site.frac_coords,
                properties=site.properties)
        merged.merge_sites(mode='average', tol=tol)
    logger.debug(
        "Merged %d structures (%d -> %d atoms)",
        len(structs), sites_before, len(merged))
    return merged


def structure_distance(
        structure1: Structure, structure2: Structure,
        tol: float = 0.1) -> float:
    """Return distance between two similar structures.
    The structures must have the same number of sites and species.
    The returned value is a sum of distances between the nearest
    lattice sites.  Distances below TOL do not contribute to the
    sum.
    """
    diffs = structure_diff(structure1, structure2)

    return sum(x for x in [np.linalg.norm(diff) for diff in diffs]
               if x > tol)


def structure_diff(
        structure1: Structure, structure2: Structure) -> float:
    """Return difference between two similar structures as list of vectors.
    The structures must have the same number of sites and species.
    The returned value is a list of vectors to be applied to the first
    structure nodes to obtain the second structure (after sorting to
    make sites consistent).
    """
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    str2 = structure1.interpolate(
        structure2, 2, autosort_tol=0.5)[2]

    return pbc_shortest_vectors(
        str1.lattice, str1.frac_coords, str2.frac_coords)
