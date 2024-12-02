"""Extension for pymatgen.core.structure
"""
import logging
import numpy as np
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
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    str2 = structure1.interpolate(
        structure2, 2, autosort_tol=0.5)[2]

    tot_distance = 0
    for node1, node2 in zip(str1, str2):
        distance = node1.distance(node2)
        if distance > tol:
            tot_distance += distance

    return tot_distance


def structure_diff(
        structure1: Structure, structure2: Structure) -> float:
    """Return difference between two similar structures as list of vectors.
    The vectors are in fractional coordinates.

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

    start_coords = np.array(str1.frac_coords)
    end_coords = np.array(str2.frac_coords)

    diff = end_coords - start_coords
    # Account for periodic boundary conditions
    diff -= np.round(diff)  # this works because fractional coordinates

    return diff


def structure_interpolate2(
        structure1: Structure, structure2: Structure,
        nimages: int = 10,
        **kwargs) -> list[Structure]:
    """Like Structure.interpolate, but make sure that images are valid.
    Valid means that no atoms in the images are very close
    (Structure.is_valid).
    NIMAGES can only be the number of images, not a list.
    **KWARGS are the other arguments passed to Structure.interpolate,
    which see.
    Return a list of interpolated structures, possibly adjusted to
    avoid atom collisions by changing distances between images.
    """
    assert structure1.is_valid()
    assert structure2.is_valid()
    images = structure1.interpolate(structure2, nimages=nimages, **kwargs)

    def all_valid(images):
        """Return True if all IMAGES are valid.
        Otherwise, return invalid image index."""
        for idx, image in enumerate(images):
            if not image.is_valid():
                return idx
        return True

    invalid_idx = all_valid(images)
    # Normal interpolation works fine.  Return immediately.
    if invalid_idx is True:
        return images

    # Otherwise, adjust the spacing manually to avoid collisions.
    nimages = np.arange(nimages + 1) / nimages

    def get_image(coord):
        """Get image at COORD."""
        return structure1.interpolate(
            structure2, nimages=[coord], **kwargs)[0]

    def search_valid(valid_coord, invalid_coord, tol=1E-3):
        """Find valid interpolation coordinate between VALID_COORD and INVALID_COORD.
        Assume that INVALID_COORD is an invalid image and that
        VALID_COORD is valid.
        """
        while np.abs(valid_coord - invalid_coord) > tol:
            trial_coord = (valid_coord + invalid_coord) / 2.0
            trial_image = get_image(trial_coord)
            if trial_image.is_valid():
                valid_coord = trial_coord
            else:
                invalid_coord = trial_coord
        return valid_coord

    while invalid_idx is not True:
        left_coord = search_valid(
            nimages[invalid_idx - 1], nimages[invalid_idx])
        if np.abs(nimages[invalid_idx - 1] - left_coord) > 1E-3:
            nimages[invalid_idx] = left_coord
        else:  # No valid point to the left.  Search right.
            next_valid_idx = invalid_idx + 1
            while not get_image(nimages[next_valid_idx]).is_valid():
                next_valid_idx += 1
            right_coord = search_valid(
                nimages[next_valid_idx], nimages[invalid_idx])
            rescaled = np.linspace(
                right_coord, nimages[-1], num=len(nimages[invalid_idx:]))
            for idx, coord in enumerate(rescaled):
                nimages[idx + invalid_idx] = coord
        images = structure1.interpolate(structure2, nimages, **kwargs)
        invalid_idx = all_valid(images)
    logger.info("Adjusted interpolation coordinates to %s", nimages)
    return images
