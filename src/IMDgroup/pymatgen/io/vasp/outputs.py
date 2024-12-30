"""This module implements extensions to pymatgen.io.vasp.outputs module.
"""

import warnings
from pymatgen.io.vasp.outputs import Vasprun as pmgVasprun
from IMDgroup.pymatgen.io.vasp.inputs import Incar


class VasprunWarning(Warning):
    """Warning for VASP run."""


class Vasprun(pmgVasprun):
    """Modified version of pymatgen's Vasprun class, which see.
    """

    @property
    def final_energy(self) -> float:
        """Final energy from the VASP run."""
        energy = super().final_energy

        if self.incar.get('IBRION') in Incar.IBRION_IONIC_RELAX_values and\
           self.incar.get('ISIF') != Incar.ISIF_FIX_SHAPE_VOL:
            warnings.warn(
                "Energy may not be accurate when using "
                f"ISIF({self.incar.get('ISIF')})!={Incar.ISIF_FIX_SHAPE_VOL}",
                VasprunWarning
            )

        return energy
