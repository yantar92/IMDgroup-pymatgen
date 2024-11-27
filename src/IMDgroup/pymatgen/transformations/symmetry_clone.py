"""Generate all the symmetrically equivalent clones of a site in structure.
"""

import logging
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.core import (SymmOp, Structure, Element)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.typing import SpeciesLike
from alive_progress import alive_bar
from IMDgroup.pymatgen.core.structure import structure_distance

__author__ = "Ihor Radchenko <yantar92@posteo.net>"
logger = logging.getLogger(__name__)


class SymmetryFillTransformation(AbstractTransformation):
    """Create structures with given sites cloned according to symmetry."""

    def __init__(
            self,
            sym_operations: list[SymmOp] | Structure,
            element_list: list[SpeciesLike]):
        """Create structure with sites cloned according to symmetry.
        Args:
         sym_operations: List of SymmOp *fractional* operations to be
           applied or a reference Structure to be used to generate the
           operations.
         species_list: List of species to be cloned.
        """
        self.element_set = set(map(Element, element_list))

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
        """Transform structure, filling symmetrically equivalent sites.

        Args:
          structure (Structure or filename):
            Structure to insert into.
          _return_ranked_list (bool | int, optional):
            Unused (this is one-to-one transformation).

        Returns:
          Transformed structure.  Each cloned site will have its
          property 'symop' be set to symmetry operation used to
          generate the clone.
        """
        all_elements = map(Element, structure.species)
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


class SymmetryCloneTransformation(AbstractTransformation):
    """Create equivalent structures according to symmetry."""

    def __init__(
            self,
            sym_operations: list[SymmOp] | Structure,
            filter_cls=None,
            tol: float = 0.5
            ):
        """Create eqivalent structures cloned according to symmetry.
        Args:
         sym_operations: List of SymmOp *fractional* operations to be
           applied or a reference Structure to be used to generate the
           operations.
         filter_cls: A class instance with method filter accepting two
           arguments: trial structure and list of structure clones
           known so far.  When function returns False, the trial
           structure is rejected.
           In addition, the class may have final_filter method
           accepting a list of structure clones.  It must return the
           filtered structure.
         tol: Tolerance when comparing equivalent clones.
           If sum of distances between all site positions in two
           clones is less than tol, they are considered the same.
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

    def get_all_clones(self, structure):
        """Generate a list of all clones for STRUCTURE.
        """
        clones = []

        def _member(structure, clones):
            """Return True when STRUCTURE is in CLONES.
            Return False otherwise.
            """
            for clone in clones:
                dist = structure_distance(structure, clone)
                if dist < self.tol:
                    return True
            return False

        with alive_bar(len(self.sym_operations)) as progress_bar:
            for op in self.sym_operations:
                progress_bar()  # pylint: disable=not-callable
                tmp = structure.copy()
                tmp.apply_operation(op, fractional=True)
                # Operation might change the lattice vectors.
                # Force them back into STRUCTURE by enforcing periodic
                # conditions
                clone = structure.copy()
                clone.remove_sites(indices=range(len(clone)))  # empty
                for site in tmp:
                    clone.append(
                        site.species, site.coords,
                        coords_are_cartesian=True,
                        properties=site.properties)
                # Make life easier for the collees.  Align sites
                # between clone and structure
                clone = structure.interpolate(clone, 2, autosort_tol=0.5)[2]
                clone.properties['symop'] = op
                if not _member(clone, clones)\
                   and (self.filter_cls is None
                        or self.filter_cls.filter(clone, clones)):
                    clones.append(clone)

        # Apply additional filters
        if self.filter_cls is not None and\
           getattr(self.filter_cls, "final_filter", False):
            clones = self.filter_cls.final_filter(clones)

        # Sort structures by distance from reference STRUCTURE
        clones = sorted(
            clones,
            key=lambda clone: structure_distance(clone, structure))

        return clones

    def apply_transformation(
            self,
            structure: Structure | str,
            return_ranked_list: bool | int = False):
        """Create symmetrically equivalent closed of STRUCTURE.

        Args:
          structure (Structure or filename):
            Structure to clone.
          return_ranked_list (bool | int, optional):
            If int, return that number of structures.

        Returns:
          Transformed list of structures.  Each cloned structure will
          have its property 'symop' be set to symmetry operation used
          to generate the clone.
        """
        all_clones = self.get_all_clones(structure)
        if not return_ranked_list:
            return all_clones[0]
        return [{"structure": structure}
                for structure in all_clones[:return_ranked_list]]

    @property
    def is_one_to_many(self) -> bool:
        """Transform one structure to many."""
        return True
