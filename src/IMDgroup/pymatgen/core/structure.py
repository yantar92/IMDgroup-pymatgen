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


"""Extension for pymatgen.core.structure
"""
import logging
import warnings
import os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from monty.io import zopen
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.util.typing import PathLike
from typing_extensions import Self

logger = logging.getLogger(__name__)


# FIXME: Contribute upstream
# FIXME: cannot read lat.in because it will have >1 occupancies
# when calling Mcsqs.structure_from_str.  Need to modify Mcsqs
class IMDStructure(Structure):
    """IMDGroup variant of Structure.
    New features:
    1. Read structure from str.out ATAT's file.
    2. Write structure to str.out ATAT's file replacing X0+ with Vac
    """
    @classmethod
    def from_file(
            cls,
            filename: PathLike,
            primitive: bool = False,
            sort: bool = False,
            merge_tol: float = 0.0,
            **kwargs,
    ) -> Self:
        """Read a structure from a file.
        Support everything from pymatgen.Structure and also
        ATAT's structures.  ATAT's structures will contain
        vacancies (Vac) as dummy X species.

        Args:
            filename (PathLike): The file to read.
            primitive (bool): Whether to convert to a primitive cell. Defaults to False.
            sort (bool): Whether to sort sites. Default to False.
            merge_tol (float): If this is some positive number, sites that are within merge_tol from each other will be
                merged. Usually 0.01 should be enough to deal with common numerical issues.
            kwargs: Passthrough to relevant reader. E.g. if the file has CIF format, the kwargs will be passed
                through to CifParser.

        Returns:
            Structure.
        """
        filename = str(filename)
        if os.path.basename(filename) in ("str.out"):  # , "lat.in"
            from pymatgen.io.atat import Mcsqs
            # We manually replace Vac with X instances that can be
            # read by pymatgen.
            atat_structure_text = Path(filename).read_text(encoding='utf-8')
            atat_structure_text = atat_structure_text.replace("Vac", "X")
            struct = Mcsqs.structure_from_str(atat_structure_text)
            if sort:
                struct = struct.get_sorted_structure()
            if merge_tol:
                struct.merge_sites(merge_tol)
            struct.__class__ = cls
            return struct
        ret = super().from_file(filename, primitive, sort, merge_tol, **kwargs)
        ret.__class__ = cls
        return ret

    def to_file(self, filename: str = "", fmt="") -> str | None:
        """A more intuitive alias for .to()."""
        return self.to(filename, fmt)

    def to(self, filename: PathLike = "", fmt="", **kwargs) -> str:
        """Output the structure to a file or string.
        In addition to what pymatgen provides, write "str.out" file suitable
        for ATAT, replacing X0+ species with Vac and dropping occupancies.
        This corresponds to fmt="atat".

        Args:
            filename (PathLike): If provided, output will be written to a file. If
                fmt is not specified, the format is determined from the
                filename. Defaults is None, i.e. string output.
            fmt (str): Format to output to. Defaults to JSON unless filename
                is provided. If fmt is specifies, it overrides whatever the
                filename is. Options include "cif", "poscar", "cssr", "json",
                "xsf", "mcsqs", "prismatic", "yaml", "yml", "fleur-inpgen", "pwmat",
                "aims".
                Non-case sensitive.
            **kwargs: Kwargs passthru to relevant methods. e.g. This allows
                the passing of parameters like symprec to the
                CifWriter.__init__ method for generation of symmetric CIFs.

        Returns:
            str: String representation of molecule in given format. If a filename
                is provided, the same string is written to the file.
        """
        filename, fmt = str(filename), fmt.lower()
        if fmt == "atat" or os.path.basename(filename) in ("str.out"):
            from pymatgen.io.atat import Mcsqs
            res_str = Mcsqs(self).to_str().replace('X0+', 'Vac').replace('=1.0','').replace('=1', '')
            with zopen(filename, mode="wt", encoding="utf8") as file:
                file.write(res_str)
            return res_str
        return super().to(filename, fmt, **kwargs)


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

    # sites_before = sum(len(s) for s in structs)
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
        pbc: bool = True,
        match_species: bool = True
        ):
    """Find the best site match between REFERENCE_STRUCT and TARGET_STRUCT.
    Return modified TARGET_STRUCT with sites rearranged in such a way that
    reference_struct[idx] is close to return_value[idx] and have the
    same species.

    TARGET_STRUCT must have the same lattice with REFERENCE_STRUCT and
    must contain sites of REFERENCE_STRUCT as a subset.

    When PBC is False, do not use boundary conditions to compute distances.

    When MATCH_SPECIES is False, do not require species to match.
    """
    # Check length of structures
    if len(target_struct) < len(reference_struct):
        raise ValueError("Target structure has too few sites!")

    if not np.all(np.isclose(
            reference_struct.lattice.parameters,
            target_struct.lattice.parameters,
            rtol=1e-05)):
        raise ValueError("Structures with different lattices!")

    start_coords = np.array(reference_struct.frac_coords)
    end_coords = np.array(target_struct.frac_coords)

    if pbc:
        dist_matrix = reference_struct.lattice.get_all_distances(
            start_coords, end_coords)
    else:
        diff = start_coords[:, np.newaxis, :] - end_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    matched = np.full(len(end_coords), False)

    result_sites = []

    already_matched = True
    # Assign the closest site with the same species
    for idx, row in enumerate(dist_matrix):
        ind = np.argsort(row)
        found_mapping = False
        for matched_idx in ind:
            if not matched[matched_idx] and\
               ((not match_species) or
               reference_struct[idx].species ==
               target_struct[matched_idx].species):
                matched[matched_idx] = True
                result_sites.append(target_struct[matched_idx])
                found_mapping = True
                if idx != matched_idx:
                    already_matched = False
                break
        if not found_mapping:
            raise ValueError("Unable to reliably match structures")

    if already_matched:
        return target_struct.copy()

    # If there are more sites in target_struct, add them to the end.
    for idx, site_matched in enumerate(matched):
        if not site_matched:
            result_sites.append(target_struct[idx])

    return IMDStructure.from_sites(
        result_sites,
        properties=target_struct.properties
    )


def structure_diff(
        structure1: Structure, structure2: Structure,
        tol: float = 0.1,
        match_first: bool = True,
        match_species: bool = True
        ):
    """Return translation vectors between two similar structures.
    The structures must have the same number of sites and species.
    Trnalations shorter than TOL angstrem do not contribute to the
    result.

    When MATCH_FIRST is True (default), call get_matched_structure
    first.  When MATCH_FIRST is True and MATCH_SPECIES is false, match
    structures ignoring species.
    """
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    if match_first:
        str2 = get_matched_structure(
            structure1, structure2, match_species=match_species)
    else:
        str2 = structure2

    vectors = []
    # Returns cartesian!
    diff_matrix = pbc_shortest_vectors(
        str1.lattice, str1.frac_coords, str2.frac_coords,
        # Only compute diagonal elements (1-to-1 matching)
        mask=~np.eye(len(str1), dtype=bool),
        return_d2=False)
    diff = [diff_matrix[idx][idx] for idx in range(len(str1))]

    for v in diff:
        if np.linalg.norm(v) > tol:
            vectors.append(v)
        else:
            vectors.append(np.array([0, 0, 0]))

    return vectors


def structure_distance(
        structure1: Structure, structure2: Structure,
        tol: float = 0.1,
        match_first=True,
        max_dist=None,
        norm=False,
        match_species: bool = True) -> float:
    """Return tuple distance between two similar structures.
    The structures must have the same number of sites and species.
    The returned value is a square root of sum of squared distances
    between the nearest lattice sites.  Distances below TOL do not
    contribute to the sum.

    When structures have similar, but not the same lattices, the
    comparison is done by mapping the fractional site positions in
    STRUCTURE2 onto lattice vectors of STRUCTURE1.

    When NORM is True (default: False), norm the distance by the number
    of displacement above threshold.

    When MATCH_FIRST is True (default), call get_matched_structure
    first.  MATCH_SPECIES (default: True) controls whether to
    assert that species should match during structure matching.

    When MAX_DIST is provided, return immediately when computed
    distance exceeds MAX_DIST.
    """
    str1 = structure1
    # interpolate knows how to match similar sites, spitting out
    # re-ordered (to match structure1) final structure as output
    # This also performs the necessary assertions about structure
    # similarity
    if match_first:
        str2 = get_matched_structure(
            structure1, structure2, match_species=match_species)
    else:
        str2 = structure2

    # Returns cartesian!
    _, dist2_matrix = pbc_shortest_vectors(
        str1.lattice,
        str1.frac_coords, str2.frac_coords,
        # Only compute diagonal elements (1-to-1 matching)
        mask=~np.eye(len(str1), dtype=bool),
        return_d2=True)

    displaced_sites = 0
    tot_distance_square = 0
    max_dist_square = None
    if max_dist is not None:
        max_dist_square = max_dist * max_dist
    for idx, _ in enumerate(str1):
        distance_square = dist2_matrix[idx][idx]
        if np.sqrt(distance_square) > tol:
            displaced_sites += 1
            tot_distance_square += distance_square
            if max_dist_square is not None and\
               tot_distance_square > max_dist_square:
                return np.sqrt(tot_distance_square)

    if norm and displaced_sites > 0:
        return np.sqrt(tot_distance_square)/displaced_sites
    return np.sqrt(tot_distance_square)


def structure_interpolate2(
        structure1: Structure, structure2: Structure,
        nimages: int = 10,
        frac_tol: float = 0.5,
        center: bool | float = 0.5,
        match_first: bool = True,
        **kwargs) -> list[Structure]:
    """Like Structure.interpolate, but make sure that images are valid.
    Valid means that no atoms in the images are very close
    (structure_is_valid2), no closer than FRAC_TOL*sum of specie radiuses.

    With CENTER set to non-False (default: 0.5Å), adjust STRUCTURE2 to
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

    if match_first:
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


class StructureDuplicateWarning(UserWarning):
    """Warning class for duplicate input structures."""


# Global variable to hold the worker function.
_global_worker = None


def _worker_wrapper(args):
    """Top-level function that calls the global worker with unpacked arguments."""
    return _global_worker(*args)


def structure_matches(
        struct: Structure,
        known_structs: list[Structure | None],
        cmp_fun=None,
        warn=False,
        multithread=False):
    """Return True when STRUCT is equivalent to any KNOWN_STRUCTS.
    Otherwise, return False.
    CMP_FUN is the function to be used to judge the equivalence
    (default: None - use StructureMatcher.fit).  It must accept two arguments
    - structures to compare.
    When WARN is True, display warning when duplicate is found.
    When MULTITHREAD is True, use multithreading.
    """
    if cmp_fun is None:
        cmp_fun = StructureMatcher(attempt_supercell=True, scale=False).fit

    known_structs = [known for known in known_structs if known is not None]

    def _warn(duplicate_of):
        if warn:
            origin_path = struct.properties.get('origin_path')
            origin_path_2 = duplicate_of.properties.get('origin_path')
            warnings.warn(
                "Duplicate structures found" +
                f" ({origin_path} and {origin_path_2})"
                if origin_path and origin_path_2 else "",
                StructureDuplicateWarning
            )

    if multithread:
        with Pool() as pool:
            global _global_worker
            _global_worker = cmp_fun
            equivs = pool.imap(
                _worker_wrapper,
                [(struct, known) for known in known_structs]
            )
            for idx, val in enumerate(equivs):
                if val:
                    pool.terminate()
                    _warn(known_structs[idx])
                    return True
    else:
        for known in known_structs:
            if cmp_fun(struct, known):
                _warn(known)
                return True
    return False


def structure_perturb(
        structure: Structure,
        distance: float,
        min_distance: float | None = None,
        frac_tol: float = 0.5,
):
    """Perform a random perturbation of the sites in STRUCTURE to break
        symmetries. Modify structure in place.

        Unlike pymatgen.core.Structure.perturb, honor selective dynamics.
        Also, make sure that the resulting structure does not have sites
        too close from one another.

        Args:
            distance (float): Distance in angstroms by which to perturb each site.
            min_distance (None, int, or float): if None, all displacements will
                be equal amplitude. If int or float, perturb each site a
                distance drawn from the uniform distribution between
                'min_distance' and 'distance'.
            frac_tol (float): Fracture tolerance for site proximity.
            The value is the minimal allowed distance in the units of sum
            of atomic radii of site species.

        Returns:
            Structure: self with perturbed sites.
        """
    assert structure_is_valid2(structure, frac_tol)
    orig_structure = structure.copy()

    counter = 0
    while True:
        structure.perturb(distance, min_distance)
        if 'selective_dynamics' in orig_structure[0].properties:
            warnings.warn(
                "Not perturbing site coordinates restricted by selective_dynamics"
            )
            for orig_site, new_site in zip(orig_structure, structure):
                for coord_idx, move in enumerate(
                        orig_site.properties['selective_dynamics']):
                    if not move:
                        new_site.frac_coords[coord_idx] =\
                            orig_site.frac_coords[coord_idx]
        if structure_is_valid2(structure, frac_tol):
            break
        counter += 1
        if counter > 100:
            raise ValueError(
                "Cannot generate sufficiently sparse"
                " perturbed structure after 100 attempts ::"
                f" distance={distance}; min_distance={min_distance};"
                f" frac_tol={frac_tol}")
        logger.debug("structure_perturb: Re-generating unlucky perturbation")
        for orig_site, new_site in zip(orig_structure, structure):
            new_site.frac_coords = orig_site.frac_coords
    return structure


def structure_strain(structure1: Structure, structure2: Structure):
    """Compute strain required to deform STRUCTURE1 lattice into STRUCTURE2.
    Return strain, as a matrix.
    """
    lat_before = structure1.lattice.matrix
    lat_after = structure2.lattice.matrix
    transform = np.dot(np.linalg.inv(lat_before), lat_after) - np.eye(3)
    strain = (transform + transform.transpose())/2.0
    return strain
