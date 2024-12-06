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

    prop_values = {}

    def store_array_props(struct):
        """Convert array properties to strings.
        This is to work around pymatgen bug#4197 where merging array
        properties fails.
        """
        for site in struct:
            for name, val in site.properties.items():
                if isinstance(val, np.ndarray):
                    site.properties[name] = str(val)
                    prop_values[str(val)] = val

    def restore_array_props(struct):
        """Convert back strings to arrays.
        This is to work around pymatgen bug#4197 where merging array
        properties fails.
        """
        for site in struct:
            for name, val in site.properties.items():
                if isinstance(val, str) and val in prop_values:
                    site.properties[name] = prop_values[val]

    merged = structs[0].copy()

    sites_before = sum(len(s) for s in structs)
    for struct in structs[1:]:
        for site in struct:
            merged.append(
                site.species,
                site.frac_coords,
                properties=site.properties)
        store_array_props(merged)
        merged.merge_sites(mode='average', tol=tol)
    restore_array_props(merged)
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
        frac_tol: float = 0.5,
        **kwargs) -> list[Structure]:
    """Like Structure.interpolate, but make sure that images are valid.
    Valid means that no atoms in the images are very close
    (structure_is_valid2), no closer than FRAC_TOL*sum of specie radiuses.
    NIMAGES can only be the number of images, not a list.
    **KWARGS are the other arguments passed to Structure.interpolate,
    which see.
    Return a list of interpolated structures, possibly adjusted to
    avoid atom collisions by changing distances between images.
    """
    assert structure_is_valid2(structure1, frac_tol)
    assert structure_is_valid2(structure2, frac_tol)
    images = structure1.interpolate(structure2, nimages=nimages, **kwargs)

    def all_valid(images):
        """Return True if all IMAGES are valid.
        Otherwise, return invalid image index."""
        for idx, image in enumerate(images):
            if not structure_is_valid2(image, frac_tol):
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

    def search_valid(valid_coord, invalid_coord):
        """Find valid interpolation coordinate between VALID_COORD and INVALID_COORD.
        Assume that INVALID_COORD is an invalid image and that
        VALID_COORD is valid.
        """
        while np.abs(valid_coord - invalid_coord) > 1E-3:
            trial_coord = (valid_coord + invalid_coord) / 2.0
            trial_image = get_image(trial_coord)
            if structure_is_valid2(trial_image, frac_tol):
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
            while not structure_is_valid2(get_image(
                    nimages[next_valid_idx]), frac_tol):
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


def structure_is_valid2(structure: Structure, frac_tol: float = 0.5) -> bool:
    """True if Structure does not contains atoms that are too close.

    The atoms are considered too close when distance between the atoms
    is less than sum of their radiuses times FRAC_TOL (default: 0.5).

    Return True if STRUCTURE does not contain atoms that are too close.
    """
    if len(structure) == 1:
        return True
    all_dists = structure.distance_matrix[np.triu_indices(len(structure), 1)]
    for i, dists in enumerate(all_dists):
        for j, dist in enumerate(dists):
            max_dist = frac_tol * (
                structure[i].specie.atomic_radius
                + structure[j].specie.atomic_radius
            )
            if dist > max_dist:
                return False
    return True
