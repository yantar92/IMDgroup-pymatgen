"""Extension for pymatgen.core.structure
"""
import logging
import warnings
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
        # Need to use copy to avoid merge_sites modifying site
        # properties by side effect.
        for site in struct.copy():
            merged.append(
                site.species,
                site.frac_coords,
                properties=site.properties)
        merged.merge_sites(mode='average', tol=tol)
    # logger.debug(
    #     "Merged %d structures (%d -> %d atoms)",
    #     len(structs), sites_before, len(merged))
    return merged


def structure_distance(
        structure1: Structure, structure2: Structure,
        tol: float = 0.1,
        autosort_tol: float | None = 0.5) -> float:
    """Return distance between two similar structures.
    The structures must have the same number of sites and species.
    The returned value is a square root of sum of squared distances
    between the nearest lattice sites.  Distances below TOL do not
    contribute to the sum.

    AUTOSOR_TOL (default: 0.5) is passed to
    pymatgen.core.Structure.interpolate.  When it is a float an
    attempt is made to match site to site via heuristics.  When None,
    compute 1-to-1 distance mapping.
    """
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    try:
        str2 = structure_interpolate2(
            structure1, structure2, 2,
            center=tol,
            frac_tol=0,
            autosort_tol=autosort_tol)[2]
    except ValueError:
        # Fall back to direct interpolation
        # Structures are way too different, so the best we can do is
        # assuming that the site order is right
        warnings.warn("Computing distance between dissimilar structures")
        # At least, make sure that we are not mapping one species into another.
        for site1, site2 in zip(structure1, structure2):
            assert site1.species == site2.species
        str2 = structure_interpolate2(
            structure1, structure2, 2, frac_tol=0, center=tol)[2]

    tot_distance_square = 0
    for node1, node2 in zip(str1, str2):
        distance = node1.distance(node2)
        if distance > tol:
            tot_distance_square += distance * distance

    return np.sqrt(tot_distance_square)


def structure_interpolate2(
        structure1: Structure, structure2: Structure,
        nimages: int = 10,
        frac_tol: float = 0.5,
        center: bool | float = 0.5,
        **kwargs) -> list[Structure]:
    """Like Structure.interpolate, but make sure that images are valid.
    Valid means that no atoms in the images are very close
    (structure_is_valid2), no closer than FRAC_TOL*sum of specie radiuses.

    A new parameter CENTER (default: 0.5Å) will adjust STRUCTURE2 to
    match STRUCTURE1 geometric center of mass before interpolation.
    CENTER may either be a boolean (True to adjust structure centers)
    or a float to adjust structure centers only when the distance
    between the centers does not exceed the float value.

    NIMAGES can only be the number of images, not a list.
    **KWARGS are the other arguments passed to Structure.interpolate,
    which see.
    Return a list of interpolated structures, possibly adjusted to
    avoid atom collisions by changing distances between images.
    """
    assert structure_is_valid2(structure1, frac_tol)
    assert structure_is_valid2(structure2, frac_tol)

    if center:
        _images = structure1.interpolate(structure2, nimages=1, **kwargs)
        center1 = np.mean(np.array(_images[0].frac_coords), axis=0)
        center2 = np.mean(np.array(_images[1].frac_coords), axis=0)
        diff = center1 - center2
        diff_len = np.linalg.norm(diff)
        # logger.debug("Interpolating: drift %fÅ", diff_len)
        if center is True or (isinstance(center, float) and diff_len < center):
            structure2 = _images[1].copy()
            # logger.info("Interpolating: adjusting centers by %fÅ", diff_len)
            structure2.translate_sites(
                list(range(len(structure2))),
                diff, frac_coords=True, to_unit_cell=True
            )

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
    all_dists = structure.distance_matrix
    for i, dists in enumerate(all_dists):
        for j, dist in enumerate(dists):
            if i == j:
                continue
            max_dist = frac_tol * (
                structure[i].specie.atomic_radius
                + structure[j].specie.atomic_radius
            )
            if dist < max_dist:
                return False
    return True


def reduce_supercell(structure):
    """Return reduced supercell for STRUCTURE.
    Do not modify STRUCTURE.
    """
    reduced_structure = structure.copy()
    reduced_structure = reduced_structure.get_primitive_structure(
        constrain_latt=['alpha', 'beta', 'gamma'])
    return reduced_structure


def get_supercell_size(structure):
    """Get supercell size for STRUCTURE.
    Return a tuple of intergers (A, B, C) for AxBxC supercell.
    """
    reduced_structure = reduce_supercell(structure)
    a = structure.lattice.a/reduced_structure.lattice.a
    b = structure.lattice.b/reduced_structure.lattice.b
    c = structure.lattice.c/reduced_structure.lattice.c
    return (round(a), round(b), round(c))

