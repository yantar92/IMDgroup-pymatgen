"""Generate all the symmetrically equivalent clones of a site in structure.
"""

import logging
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.core import (SymmOp, Structure, Element)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.typing import SpeciesLike

__author__ = "Ihor Radchenko <yantar92@posteo.net>"
logger = logging.getLogger(__name__)


class SymmetryCloneTransformation(AbstractTransformation):
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
