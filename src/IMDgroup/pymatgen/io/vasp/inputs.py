"""This module implements IMD group-specific extensions to
pymatgen.io.vasp.inputs module.
"""

import warnings
from pymatgen.io.vasp.inputs import Incar as pmgIncar
from pymatgen.io.vasp.inputs import BadIncarWarning


class Incar(pmgIncar):
    """Modified version of pymatgen's Incar class, which see.
    Extensions:
    1. Readable constants for Incar values.
    2. Warn when IBRION=-1 and NSW>0 (see
       https://www.vasp.at/wiki/index.php/IBRION)
    """

    # ISIF values
    ISIF_RELAX_POS = ISIF_FIX_SHAPE_VOL = 2
    ISIF_RELAX_POS_SHAPE_VOL = ISIF_FIX_NONE = 3
    ISIF_RELAX_POS_SHAPE = ISIF_FIX_VOL = 4
    ISIF_RELAX_SHAPE = IFIX_FIX_POS_VOL = 5
    ISIF_RELAX_SHAPE_VOL = ISIF_FIX_POS = 6
    ISIF_RELAX_VOL = ISIF_FIX_POS_SHAPE = 7
    ISIF_RELAX_POS_VOL = ISIF_FIX_SHAPE = 8

    # IBRION values
    IBRION_NONE = -1
    IBRION_MD = 0
    IBRION_IONIC_RELAX_FORCE_FAST = 1
    IBRION_IONIC_RELAX_CGA = 2
    IBRION_IONIC_RELAX_DAMPED_MD = 3

    IBRION_IONIC_RELAX_values = [
        IBRION_IONIC_RELAX_FORCE_FAST,
        IBRION_IONIC_RELAX_CGA,
        IBRION_IONIC_RELAX_DAMPED_MD]

    # FIXME: This should better be contributed upstream as I cannot
    # override the checks in the Incar instances used from pymatgen
    # internals.
    @classmethod
    def proc_val(cls, key: str, val: str) -> list | bool | float | int | str:
        """Helper method to convert INCAR parameters to proper types
        like ints, floats, lists, etc.

        Args:
            key (str): INCAR parameter key.
            val (str): Value of INCAR parameter.
        """
        result = super().proc_val(key, val)
        if cls.get("IBRION", None) == cls.IBRION_NONE and\
           cls.get("NSW", 0) > 0:
            warnings.warn(
                f"NSW ({cls.get('NSW', "N/A")}) > 0 is useless"
                f" with IBRION = {cls.IBRION_NONE}",
                BadIncarWarning)
        return result
