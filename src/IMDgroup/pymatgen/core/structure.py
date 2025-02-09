"""Extension for pymatgen.core.structure
"""
import logging
import warnings
from multiprocessing import Pool
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
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


def get_matched_structure(
        reference_struct: Structure,
        target_struct: Structure,
        pbc: bool = True
        ):
    """Find the best site match between REFERENCE_STRUCT and TARGET_STRUCT.
    Return modified TARGET_STRUCT with sites rearranged in such a way that
    reference_struct[idx] is close to return_value[idx] and have the
    same species.

    TARGET_STRUCT must have the same lattice with REFERENCE_STRUCT and
    must contain sites of REFERENCE_STRUCT as a subset.

    When PBC is False, do not use boundary conditions to compute distances.
    """
    # Check length of structures
    if len(target_struct) < len(reference_struct):
        raise ValueError("Target structure has too few sites!")

    if not reference_struct.lattice == target_struct.lattice:
        raise ValueError("Structures with different lattices!")

    start_coords = np.array(reference_struct.frac_coords)
    end_coords = np.array(target_struct.frac_coords)

    if pbc:
        dist_matrix = reference_struct.lattice.get_all_distances(
            start_coords, end_coords)
    else:
        diff = start_coords[:, np.newaxis, :] - end_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    matched = np.full(len(start_coords), False)

    result_sites = []

    # Assign the closest site with the same species
    for idx, row in enumerate(dist_matrix):
        ind = np.argsort(row)
        found_mapping = False
        for matched_idx in ind:
            if not matched[matched_idx] and\
               reference_struct[idx].species ==\
               target_struct[matched_idx].species:
                matched[matched_idx] = True
                result_sites.append(target_struct[matched_idx])
                found_mapping = True
                break
        if not found_mapping:
            raise ValueError("Unable to reliably match structures")

    # If there are more sites in target_struct, add them to the end.
    for idx, site_matched in enumerate(matched):
        if not site_matched:
            result_sites.append(target_struct[idx])

    return Structure.from_sites(
        result_sites,
        charge=target_struct.charge,
        properties=target_struct.properties
    )


def structure_diff(
        structure1: Structure, structure2: Structure,
        tol: float = 0.1,
        match_first = True
        ):
    """Return translation vectors between two similar structures.
    The structures must have the same number of sites and species.
    Trnalations shorter than TOL angstrem do not contribute to the
    result.

    When MATCH_FIRST is True (default), call get_matched_structure
    first.
    """
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    if match_first:
        str2 = get_matched_structure(structure1, structure2)
    else:
        str2 = structure2

    vectors = []
    diff_frac = [
        pbc_shortest_vectors(str1.lattice, c1, c2)[0][0]
        for c1, c2 in zip(str1.frac_coords, str2.frac_coords)
    ]
    diff = str1.lattice.get_cartesian_coords(diff_frac)
    for v in diff:
        if np.linalg.norm(v) > tol:
            vectors.append(v)
        else:
            vectors.append(np.array([0, 0, 0]))

    return vectors


def structure_distance(
        structure1: Structure, structure2: Structure,
        tol: float = 0.1,
        match_first=True) -> float:
    """Return tuple distance between two similar structures.
    The structures must have the same number of sites and species.
    The returned value is a square root of sum of squared distances
    between the nearest lattice sites.  Distances below TOL do not
    contribute to the sum.

    When MATCH_FIRST is True (default), call get_matched_structure
    first.
    """
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    if match_first:
        str2 = get_matched_structure(structure1, structure2)
    else:
        str2 = structure2

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
    if center:
        center1 = np.mean(np.array(structure1.frac_coords), axis=0)
        center2 = np.mean(np.array(structure2.frac_coords), axis=0)
        diff = center1 - center2
        diff_len = np.linalg.norm(diff)
        # logger.debug("Interpolating: drift %fÅ", diff_len)
        if center is True or (isinstance(center, float) and diff_len < center):
            structure2 = structure2.copy()
            # logger.info("Interpolating: adjusting centers by %fÅ", diff_len)
            structure2.translate_sites(
                list(range(len(structure2))),
                diff, frac_coords=True, to_unit_cell=True
            )

    structure2 = get_matched_structure(structure1, structure2)

    images = structure1.interpolate(structure2, nimages=nimages, **kwargs)

    if np.isclose(frac_tol, 0):
        return images

    assert structure_is_valid2(structure1, frac_tol)
    assert structure_is_valid2(structure2, frac_tol)

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

