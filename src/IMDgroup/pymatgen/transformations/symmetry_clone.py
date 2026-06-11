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


"""Generate all the symmetrically equivalent clones of a site in structure."""

import logging
from multiprocessing import Pool, cpu_count
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.core import (SymmOp, Structure, get_el_sp)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.typing import SpeciesLike
from alive_progress import alive_bar
from IMDgroup.pymatgen.core.structure import structure_distance, get_matched_structure

__author__ = "Ihor Radchenko <yantar92@posteo.net>"
logger = logging.getLogger(__name__)


class SymmetryFillTransformation(AbstractTransformation):
    """Clone selected sites according to symmetry operations.

    Applies a list of symmetry operations to all sites of a given
    element set, adding new sites when they are not too close to
    existing ones.

    Attributes:

    - **sym_operations**: List of fractional SymmOp objects.
    - **element_set**: Set of species to clone.
    """

    def __init__(
            self,
            sym_operations: list[SymmOp] | Structure,
            element_list: list[SpeciesLike]):
        """Create structure with sites cloned according to symmetry.

        Args:
            sym_operations: List of SymmOp *fractional* operations to be
                applied or a reference Structure to be used to generate the
                operations.
            element_list: List of species to be cloned.
        """
        self.element_set = set(map(get_el_sp, element_list))

        if isinstance(sym_operations, Structure):
            analyzer = SpacegroupAnalyzer(sym_operations)
            # Fractional
            self.sym_operations = analyzer.get_symmetry_operations()
        elif (isinstance(sym_operations, list) and
              isinstance(sym_operations[0], SymmOp)):
            self.sym_operations = sym_operations
        else:
            raise ValueError(
                "sym_operations must be Structure of a list of SymmOp")

    def apply_transformation(
            self,
            structure: Structure | str,
            _return_ranked_list: bool | int = False):
        """Apply the symmetry fill transformation.

        Args:
            structure: Structure to fill, or path to a structure file.
            _return_ranked_list: Unused (one-to-one transformation).

        Returns:
            Structure: Filled structure.  Each cloned site has a
            ``symop`` property set to the symmetry operation used.
        """
        all_elements = map(get_el_sp, structure.species)
        elements_to_remove = set(all_elements) - self.element_set
        filled_structure = structure.copy()
        identity_op = SymmOp.from_xyz_str('x,y,z')
        for site in filled_structure:
            site.properties.update({'symop': identity_op})
        clean_structure = structure.copy()
        clean_structure.remove_species(list(elements_to_remove))
        for op in self.sym_operations:
            tmp_structure = clean_structure.copy()
            tmp_structure.apply_operation(op, fractional=True)
            for site in tmp_structure:
                props = site.properties
                props.update({'symop': op})
                try:
                    filled_structure.append(
                        site.species, site.coords,
                        coords_are_cartesian=True,
                        properties=props, validate_proximity=True)
                except ValueError:
                    # Too close, skip.
                    pass
        return filled_structure


def apply_operation_keep_lattice(structure, op):
    """Apply a symmetry operation while preserving lattice vectors.

    The modified structure will have atom-to-atom match and all
    fractional coordinates normalized within 0..1 range.

    Args:
        structure: Structure to transform.
        op: SymmOp to apply.

    Returns:
        Structure: Modified copy with unchanged lattice.
    """
    tmp = structure.copy()
    tmp.apply_operation(op, fractional=True)
    # Operation might change the lattice vectors.
    # Force them back into STRUCTURE by enforcing periodic
    # conditions
    result = structure.copy()
    result.remove_sites(indices=range(len(result)))  # empty
    for site in tmp:
        result.append(
            site.species, site.coords,
            coords_are_cartesian=True,
            properties=site.properties)
    for site in result:
        site.to_unit_cell(in_place=True)
    # Make life easier for the callees.  Align sites
    # between clone and structure
    props = result.properties  # preserve properties
    result = get_matched_structure(structure, result)
    result.properties = props
    result.properties['symop'] = op
    return result


def _structure_distance_wrapper(args):
    return structure_distance(*args)


class SymmetryCloneTransformation(AbstractTransformation):
    """Generate symmetrically equivalent clones of a structure.

    Applies all symmetry operations to produce a list of distinct
    configurations, filtering duplicates by structure distance.

    Attributes:
        sym_operations: List of fractional SymmOp objects.
        tol: Distance threshold for considering two clones equivalent.
        filter_cls: Optional filter with ``filter`` and ``final_filter``
            methods.
    """

    def __init__(
            self,
            sym_operations: list[SymmOp] | Structure,
            filter_cls=None,
            tol: float = 0.5
            ):
        """Initialise symmetry clone transformation.

        Args:
            sym_operations: List of fractional SymmOp objects, or a
                Structure used to derive them via
                ``SpacegroupAnalyzer``.
            filter_cls: Optional filter object.  Must implement
                ``filter(trial, clones) -> bool`` and may implement
                ``final_filter(clones) -> list``.
            tol: Distance tolerance for equivalence.  Two clones are
                considered identical if the sum of site distances is
                below ``tol``.
        """
        self.tol = tol
        self.filter_cls = filter_cls
        if isinstance(sym_operations, Structure):
            analyzer = SpacegroupAnalyzer(sym_operations)
            # Fractional
            self.sym_operations = analyzer.get_symmetry_operations()
        elif (isinstance(sym_operations, list) and
              isinstance(sym_operations[0], SymmOp)):
            self.sym_operations = sym_operations
        else:
            raise ValueError(
                "sym_operations must be Structure of a list of SymmOp")

    def get_all_clones(
            self,
            structure: Structure,
            progress_bar: bool = True,
            multithread: bool = False) -> list[Structure]:
        """Generate all distinct symmetry clones of a structure.

        Clones are sorted by distance from the input structure.

        Args:
            structure: Structure to clone.
            progress_bar: Whether to display a progress bar.
            multithread: Whether to use multithreading for duplicate
                detection.

        Returns:
            list[Structure]: Distinct clones with one-to-one site
            matching and ``symop`` properties.
        """
        clones = []

        def _member(structure, clones):
            """Check whether a structure is equivalent to any in a list.

            Returns:
                bool: True if the structure is within ``self.tol`` of
                any clone.
            """
            # For some reason, Python sometimes hangs here when there
            # are too few CLONES.
            # FIXME: Multithreaded version appear to be slower in practice.
            # if multithread and len(clones) > cpu_count():
            #     with Pool() as pool:
            #         distances = pool.imap_unordered(
            #             _structure_distance_wrapper,
            #             [(structure, clone, 0.1, False, self.tol)
            #              for clone in clones]
            #         )
            #         for dist in distances:
            #             if dist < self.tol:
            #                 pool.terminate()
            #                 return True
            for clone in clones:
                dist = structure_distance(
                    structure, clone, match_first=False, max_dist=self.tol)
                if dist < self.tol:
                    return True
            return False

        with alive_bar(len(self.sym_operations),
                       title='Searching clones',
                       disable=not progress_bar) as progress_bar:
            for op in self.sym_operations:
                progress_bar()  # pylint: disable=not-callable
                clone = structure.copy()
                clone = apply_operation_keep_lattice(clone, op)
                if not _member(clone, clones)\
                   and (self.filter_cls is None
                        or self.filter_cls.filter(clone, clones)):
                    assert clone.is_valid(), "Given symmetry operations lead to invalid structure"
                    clones.append(clone)

        # Apply additional filters
        if self.filter_cls is not None and\
           getattr(self.filter_cls, "final_filter", False):
            clones = self.filter_cls.final_filter(clones)

        # Sort structures by distance from reference STRUCTURE
        clones = sorted(
            clones,
            key=lambda clone: structure_distance(clone, structure, match_first=False))

        return clones

    def apply_transformation(
            self,
            structure: Structure | str,
            return_ranked_list: bool | int = False):
        """Apply symmetry clone transformation.

        Args:
            structure: Structure to clone, or path to a structure file.
            return_ranked_list: If an int, return that many structures
                as ranked dictionaries.

        Returns:
            Structure or list[dict]: Single clone when
            ``return_ranked_list`` is False, otherwise a list of
            ``{'structure': ...}`` dictionaries.
        """
        all_clones = self.get_all_clones(structure)
        if not return_ranked_list:
            return all_clones[0]
        return [{"structure": structure}
                for structure in all_clones[:return_ranked_list]]

    @property
    def is_one_to_many(self) -> bool:
        """Whether the transformation is one-to-many (always True)."""
        return True
