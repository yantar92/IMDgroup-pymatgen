"""Extension for pymatgen.core.structure
"""
import logging
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


def merge_structures(
        structs: list[Structure],
        tol: float = 0.01,
        ) -> Structure:
    """Merge all STRUCTS into a single Structure.
    Return structure.
    tol is tolerance passed to Structure.merge_sites method.
    """
    logger.debug("Merging %d structures...", len(structs))
    assert len(structs) > 0
    for struct in structs[1:]:
        assert struct.lattice == structs[0].lattice

    merged = structs[0].copy()
    for struct in structs[1:]:
        for site in struct:
            merged.append(
                site.species,
                site.frac_coords,
                properties=site.properties)
    sites_before = len(merged)
    merged.merge_sites(mode='average', tol=tol)
    logger.debug(
        "Merging %d structures... done (%d -> %d atoms)",
        len(structs), sites_before, len(merged))
    return merged
