"""This module implements extensions to pymatgen.io.vasp.outputs module.
"""

import warnings
from pymatgen.io.vasp.outputs import Vasprun as pmgVasprun
from pymatgen.io.vasp.outputs import Outcar as pmgOutcar
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


class Outcar(pmgOutcar):
    """Modified version of pymatgen's Outcar class that stores all the fields.
    """

    def as_dict(self) -> dict:
        """MSONable dict."""
        dct = super().as_dict()
        for key, value in vars(self).items():
            if key not in dct:
                dct[key] = value
        return dct
