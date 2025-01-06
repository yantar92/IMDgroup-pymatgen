"""Ase <> pymatgen data converters.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.ase import (AseAtomsAdaptor, MSONAtoms)

from ase.atoms import Atoms
from ase.constraints import FixAtoms, FixCartesian

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from pymatgen.core.structure import SiteCollection


class IMDGAseAtomsAdaptor (AseAtomsAdaptor):
    """Adaptor serves as a bridge between ASE Atoms and pymatgen objects.
Modified to support arbitrary VASP selective dynamics constrains.
See https://github.com/materialsproject/pymatgen/pull/4229"""

    @staticmethod
    def get_atoms(structure: SiteCollection, msonable: bool = True, **kwargs) -> MSONAtoms | Atoms:
        """Get ASE Atoms object from pymatgen structure or molecule.

        Args:
            structure (SiteCollection): pymatgen Structure or Molecule
            msonable (bool): Whether to return an MSONAtoms object, which is MSONable.
            **kwargs: passed to the ASE Atoms constructor

        Returns:
            Atoms: ASE Atoms object
        """
        atoms = AseAtomsAdaptor.get_atoms(structure, msonable, **kwargs)
        atoms.set_constraint(None)

        # Read in selective dynamics if present.
        # Note that FixCartesian class uses an opposite notion of
        # "fix" and "not fix" flags: in ASE True means fixed and False
        # means not fixed.
        fix_atoms: dict | None = None
        if "selective_dynamics" in structure.site_properties:
            fix_atoms = {
                str([xc, yc, zc]): ([xc, yc, zc], [])
                for xc in [True, False]
                for yc in [True, False]
                for zc in [True, False]
            }
            # [False, False, False] is free to move - no constraint in ASE.
            del fix_atoms[str([False, False, False])]
            for site in structure:
                selective_dynamics: ArrayLike = site.properties.get("selective_dynamics")  # type: ignore[assignment]
                for cmask_str in fix_atoms:
                    cmask_site = (~np.array(selective_dynamics)).tolist()
                    fix_atoms[cmask_str][1].append(cmask_str == str(cmask_site))
        else:
            fix_atoms = None

        # Set the selective dynamics with the FixCartesian class.
        if fix_atoms is not None:
            atoms.set_constraint(
                [
                    FixAtoms(indices) if cmask == [True, True, True] else FixCartesian(indices, mask=cmask)
                    for cmask, indices in fix_atoms.values()
                    # Do not add empty constraints
                    if any(indices)
                ]
            )

        return atoms

    @staticmethod
    def get_structure(atoms: Atoms, cls: type[Structure] = Structure, **cls_kwargs) -> Structure:
        """Get pymatgen structure from ASE Atoms.

        Args:
            atoms: ASE Atoms object
            cls: The Structure class to instantiate (defaults to pymatgen Structure)
            **cls_kwargs: Any additional kwargs to pass to the cls

        Returns:
            Structure: Equivalent pymatgen Structure
        """
        structure = AseAtomsAdaptor.get_structure(atoms, cls, **cls_kwargs)

        # If the ASE Atoms object has constraints, make sure that they are of the
        # FixAtoms or FixCartesian kind, which are the only ones that
        # can be supported in Pymatgen.
        # By default, FixAtoms fixes all three (x, y, z) dimensions.
        if atoms.constraints:
            unsupported_constraint_type = False
            constraint_indices: dict = {
                str([xc, yc, zc]): ([xc, yc, zc], [])
                for xc in [True, False]
                for yc in [True, False]
                for zc in [True, False]
            }
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    constraint_indices[str([False] * 3)][1].extend(constraint.get_indices().tolist())
                elif isinstance(constraint, FixCartesian):
                    cmask = (~np.array(constraint.mask)).tolist()
                    constraint_indices[str(cmask)][1].extend(constraint.get_indices().tolist())
                else:
                    unsupported_constraint_type = True
            if unsupported_constraint_type:
                warnings.warn(
                    "Only FixAtoms and FixCartesian is supported by Pymatgen. Other constraints will not be set.",
                    stacklevel=2,
                )
            sel_dyn = []
            for atom in atoms:
                constrained = False
                for mask, indices in constraint_indices.values():
                    if atom.index in indices:
                        sel_dyn.append(mask)
                        constrained = True
                        break  # Assume no duplicates
                if not constrained:
                    sel_dyn.append([False] * 3)
        else:
            sel_dyn = None

        if sel_dyn is not None and ~np.all(sel_dyn):
            structure.add_site_property("selective_dynamics", sel_dyn)

        return structure
