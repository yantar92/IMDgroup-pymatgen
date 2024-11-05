"""This module implements IMD group-specific extensions to
pymatgen.io.vasp.inputs module.
"""

import os
from monty.serialization import loadfn
from pymatgen.io.vasp.inputs import Incar as pmgIncar
from IMDgroup.common import groupby_cmp


MODULE_DIR = os.path.dirname(__file__)


# We copy this over from pymatgen.io.vasp.sets because we need our own
# MODULE_DIR
def _load_yaml_config(fname):
    config = loadfn(f"{MODULE_DIR}/{fname}.yaml")
    if "PARENT" in config:
        parent_config = _load_yaml_config(config["PARENT"])
        for k, v in parent_config.items():
            if k not in config:
                config[k] = v
            elif isinstance(v, dict):
                v_new = config.get(k, {})
                v_new.update(v)
                config[k] = v_new
    return config


class Incar(pmgIncar):
    """Modified version of pymatgen's Incar class, which see.
    Extensions:
    1. Readable constants for Incar values.
    2. Warn when IBRION=-1 and NSW>0 (see
       https://www.vasp.at/wiki/index.php/IBRION)
    3. Add methods to retreieve standard setting combinations
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
    # def proc_val(self, key: str, val: str) -> list | bool | float | int | str:
    #     """Helper method to convert INCAR parameters to proper types
    #     like ints, floats, lists, etc.

    #     Args:
    #         key (str): INCAR parameter key.
    #         val (str): Value of INCAR parameter.
    #     """
    #     result = pmgIncar.proc_val(key, val)
    #     if self.get("IBRION", None) == self.IBRION_NONE and\
    #        self.get("NSW", 0) > 0:
    #         warnings.warn(
    #             f"NSW ({self.get('NSW', "N/A")}) > 0 is useless"
    #             f" with IBRION = {self.IBRION_NONE}",
    #             BadIncarWarning)
    #     return result

    @staticmethod
    def get_recipe(setup: str, name: str):
        """Retrieve INCAR settings for SETUP with NAME.
        setup can be:
        1. "functional" to retrieve functional setup from names
           PBE, PBEsol, PBE+D2, PBE+TS, vdW-DF, vdW-DF2, optB88-vdW,
           optB86b-vdW .
        """
        settings = None
        if setup == "functional":
            functional_config = _load_yaml_config("functionals")
            if name not in functional_config:
                raise KeyError(
                    "Invalid or unsupported functional. " +
                    "Supported functionals are " +
                    ', '.join(functional_config) + "."
                )
            settings = functional_config.get(name)
        if settings:
            for key, val in settings.items():
                if val == "None":
                    settings[key] = None
            return settings
        raise ValueError(f"Unknown setup: {setup}")

    @staticmethod
    def group_incars(incars):
        """Group similar incars together.
        Returns: (common_incar, [[group1] [group2] ...]).
        """
        def _incar_eq(incar1, incar2):
            """Return True when INCAR1 is equal to INCAR2.
            SYSTEM keyword is ignored.
            """
            difference = incar1.diff(incar2)
            return difference["Different"] is None\
                or list(difference["Different"].keys()) == ["SYSTEM"]

        def _incar_name(incar):
            """Get INCAR name.
            Assume that name is stored in SYSTEM parameter.
            """
            return incar['SYSTEM']

        groups = groupby_cmp(incars, _incar_eq, _incar_name)

        common_incar = None
        for group in groups:
            if common_incar is None:
                common_incar = group[0].copy()
                common_incar.pop('SYSTEM')
            else:
                for key, val in group[0].items():
                    if common_incar.get(key, None) != val:
                        common_incar.pop(key, None)
                for key, val in common_incar.copy().items():
                    if group[0].get(key, None) != val:
                        common_incar.pop(key, None)

        return (common_incar, groups)
