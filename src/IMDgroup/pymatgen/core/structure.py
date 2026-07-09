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


"""Extension for pymatgen.core.structure."""
import logging
import warnings
import os
from pathlib import Path
import multiprocessing
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
    """IMDGroup variant of pymatgen Structure.

    Adds the ability to read and write ATAT ``str.out`` files and
    handle vacancies (Vac) as dummy species X.
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
        if Path(filename).suffix == ".out":  # "str.out"
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
            res_str = Mcsqs(self).to_str().replace('X0+', 'Vac').replace('=1.0', '').replace('=1', '')
            with zopen(filename, mode="wt", encoding="utf8") as file:
                file.write(res_str)
            return res_str
        return super().to(filename, fmt, **kwargs)


def merge_structures(
        structs: list[Structure],
        tol: float = 0.01,
) -> Structure:
    """Merge multiple structures into a single Structure.

    All structures must share the same lattice.  Sites are merged
    with the given tolerance.

    Args:
        structs: List of structures to merge.  Must be non-empty.
        tol: Tolerance in Angstrom for merging sites (passed to
            ``Structure.merge_sites``).

    Returns:
        Structure: A new structure containing merged sites from all inputs.
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
    """Rearrange sites in target_struct to best match reference_struct.

    Returns a modified target_struct with sites reordered so that
    reference_struct[idx] is close to the returned structure's [idx]
    and they share the same species.  Extra sites (beyond the
    reference length) are appended at the end.

    Args:
        reference_struct: The reference structure to match against.
        target_struct: The target structure to reorder.  Must have the
            same lattice and contain reference_struct sites as a subset.
        pbc: When True (default), use periodic boundary conditions for
            distance calculations.
        match_species: When False, ignore species when matching sites.

    Returns:
        Structure: Reordered target structure with one-to-one site
        correspondence to reference_struct.

    Raises:
        ValueError: If target_struct has too few sites or the lattices
            differ, or if matching fails.
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
    """Compute translation vectors between two similar structures.

    Both structures must have the same number of sites and species.
    Each vector in the result connects corresponding sites.
    Displacements below ``tol`` Angstrom are zeroed.

    Args:
        structure1: First structure.
        structure2: Second structure.
        tol: Displacements below this threshold (Angstrom) are set to
            the zero vector.
        match_first: When True (default), call
            :func:`get_matched_structure` before computing vectors.
        match_species: When False and match_first is True, ignore
            species during structure matching.

    Returns:
        list[np.ndarray]: List of 3D cartesian displacement vectors,
        one per site.
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
    """Compute distance between two similar structures.

    The distance is the square root of the sum of squared distances
    between corresponding sites.  Displacements below ``tol`` Angstrom
    do not contribute.

    When the structures have similar but not identical lattices,
    fractional site positions of ``structure2`` are mapped onto the
    lattice vectors of ``structure1``.

    Args:
        structure1: First structure.
        structure2: Second structure.
        tol: Displacement threshold below which contributions are
            ignored (Angstrom).
        match_first: When True (default), call
            :func:`get_matched_structure` before computing distances.
        match_species: When False and match_first is True, ignore
            species during matching.
        max_dist: When set, return early if the accumulating distance
            exceeds this value.
        norm: When True, divide the result by the count of sites
            displaced above threshold.

    Returns:
        float: Structure distance.
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
        return np.sqrt(tot_distance_square) / displaced_sites
    return np.sqrt(tot_distance_square)


def structure_interpolate2(
        structure1: Structure, structure2: Structure,
        nimages: int = 10,
        frac_tol: float = 0.5,
        center: bool | float = 0.5,
        match_first: bool = True,
        **kwargs) -> list[Structure]:
    """Interpolate between structures, avoiding atom collisions.

    Like ``Structure.interpolate``, but ensures no atoms in the
    interpolated images are too close.  "Too close" means less than
    ``frac_tol * (radius1 + radius2)``.

    Args:
        structure1: Starting structure.
        structure2: Ending structure.
        nimages: Number of interpolated images (excludes endpoints).
        frac_tol: Proximity tolerance as a fraction of atomic radii sum.
            Use 0 to skip validity checks.
        center: When True or a float, align geometric centers of mass
            before interpolation.  When a float, only align if the
            center-to-center distance is below that value.
        match_first: When True (default), call
            :func:`get_matched_structure` before interpolation.
        **kwargs: Forwarded to ``Structure.interpolate``.

    Returns:
        list[Structure]: Interpolated structures, possibly with
        adjusted spacing to avoid collisions.
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
    """Check whether a structure contains no atoms that are too close.

    Atoms are considered too close when the distance between them is
    less than ``frac_tol * (atomic_radius1 + atomic_radius2)``.

    Args:
        structure: Structure to validate.
        frac_tol: Threshold multiplier for the sum of atomic radii.

    Returns:
        bool: True if all pairwise distances are above threshold.
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
    """Return the primitive cell of a supercell structure.

    Constrains alpha, beta, and gamma angles during reduction.

    Args:
        structure: Input structure (possibly a supercell).

    Returns:
        Structure: Primitive cell.  The input is not modified.
    """
    reduced_structure = structure.copy()
    reduced_structure = reduced_structure.get_primitive_structure(
        constrain_latt=['alpha', 'beta', 'gamma'])
    return reduced_structure


def get_supercell_size(structure):
    """Determine supercell dimensions relative to the primitive cell.

    Args:
        structure: Supercell structure.

    Returns:
        tuple[int, int, int]: (A, B, C) factors such that the input is
        an A x B x C supercell of its primitive.
    """
    reduced_structure = reduce_supercell(structure)
    a = structure.lattice.a / reduced_structure.lattice.a
    b = structure.lattice.b / reduced_structure.lattice.b
    c = structure.lattice.c / reduced_structure.lattice.c
    return (round(a), round(b), round(c))


class StructureDuplicateWarning(UserWarning):
    """Warning emitted when duplicate input structures are detected."""


# Global variable to hold the worker function.
_GLOBAL_WORKER = None


def _worker_wrapper(args):
    """Unpack arguments and call the global worker function.

    Args:
        args: Tuple of arguments forwarded to ``_GLOBAL_WORKER``.

    Returns:
        Result of ``_GLOBAL_WORKER(*args)``.
    """
    return _GLOBAL_WORKER(*args)


def structure_matches(
        struct: Structure,
        known_structs: list[Structure | None],
        cmp_fun=None,
        warn=False,
        multithread=False):
    """Check whether a structure is equivalent to any in a known list.

    Args:
        struct: Structure to test.
        known_structs: List of known structures.  None entries are skipped.
        cmp_fun: Callable that takes two structures and returns True if
            they match.  Defaults to ``StructureMatcher(attempt_supercell=True,
            scale=False).fit``.
        warn: When True, emit ``StructureDuplicateWarning`` on match.
        multithread: When True, use ``cpu_count - 1`` workers.  When an
            integer, use that many workers (capped at available CPUs).

    Returns:
        bool: True if a match is found, False otherwise.
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

    if multithread is not False:
        global _GLOBAL_WORKER
        _GLOBAL_WORKER = cmp_fun
        cpus = int(os.environ.get(
            'SLURM_CPUS_ON_NODE',
            multiprocessing.cpu_count()))
        # experimental: leave some buffer to avoid process being stuck
        cpus = min(1, cpus - 1)
        if isinstance(multithread, int):
            cpus = min(multithread, cpus)
        with Pool(processes=cpus) as pool:
            tasks = [(struct, known) for known in known_structs]
            for idx, is_match in enumerate(pool.imap(_worker_wrapper, tasks)):
                if is_match:
                    _warn(known_structs[idx])
                    return True
    else:
        for known in known_structs:
            if cmp_fun(struct, known):
                _warn(known)
                return True
    return False


def structure_remove_duplicates(
        structs: list[Structure | None],
        cmp_fun=None,
        warn=False,
        multithread=False):
    """Remove duplicate structures from a list, preserving order.

    Uses :func:`structure_matches` to test each structure against
    previously kept structures.  The first occurrence of each unique
    structure is kept; subsequent duplicates are replaced with None.

    Args:
        structs: List of structures to deduplicate.
        cmp_fun: Comparison function passed to
            :func:`structure_matches`.  Defaults to None, which uses
            ``StructureMatcher(attempt_supercell=True, scale=False).fit``.
        warn: When True, emit ``StructureDuplicateWarning`` for each
            duplicate found.
        multithread: When True, use ``cpu_count - 1`` workers.  When an
            integer, use that many workers (capped at available CPUs).

    Returns:
        list[Structure | None]: Input list with duplicates replaced by
        None, preserving order.
    """
    result: list[Structure | None] = []
    for struct in structs:
        if struct is None:
            result.append(None)
        elif not structure_matches(struct, result,
                                   cmp_fun=cmp_fun,
                                   warn=warn,
                                   multithread=multithread):
            result.append(struct)
    return result


def structure_perturb(
        structure: Structure,
        distance: float,
        min_distance: float | None = None,
        frac_tol: float = 0.5,
):
    """Perturb sites randomly while respecting selective dynamics.

    Unlike ``pymatgen.core.Structure.perturb``, this function honours
    ``selective_dynamics`` site properties and ensures the perturbed
    structure has no sites that are too close.

    Args:
        structure: Structure to perturb.  Modified in place.
        distance: Maximum perturbation amplitude in Angstrom.
        min_distance: When set, each perturbation is drawn uniformly
            from [min_distance, distance].
        frac_tol: Proximity tolerance as a fraction of atomic radii sum.

    Returns:
        Structure: The perturbed structure (same object).

    Raises:
        ValueError: If a valid perturbation cannot be found after 100
            attempts.
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
    """Compute the engineering strain to deform structure1 into structure2.

    Args:
        structure1: Initial structure.
        structure2: Deformed structure.

    Returns:
        np.ndarray: 3x3 symmetric strain tensor.
    """
    lat_before = structure1.lattice.matrix
    lat_after = structure2.lattice.matrix
    transform = np.dot(np.linalg.inv(lat_before), lat_after) - np.eye(3)
    strain = (transform + transform.transpose()) / 2.0
    return strain
