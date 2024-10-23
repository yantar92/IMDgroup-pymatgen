"""This module implement useful VASP input sets to be used for the
group research.
"""

import os
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from monty.serialization import loadfn
from pymatgen.io.vasp.sets import VaspInputSet
from pymatgen.io.vasp.inputs import VaspInput
from pymatgen.io.vasp.sets import _dummy_structure
from pymatgen.util.due import Doi, due
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.calculators.vasp.setups \
    import setups_defaults as ase_potential_defaults
from pymatgen.util.typing import PathLike

# ase uses pairs of 'Si': '_suffix'.  Convert them into 'Si': 'Si_suffix'
POTCAR_RECOMMENDED = dict(
    (name, name + suffix)
    for name, suffix in ase_potential_defaults['recommended'].items())


__author__ = "Ihor Radchenko <yantar92@posteo.net>"
MODULE_DIR = os.path.dirname(__file__)


def _load_cif(fname):
    return Structure.from_file(f"{MODULE_DIR}/{fname}.cif")


def _load_mp(name):
    with MPRester() as m:
        structure = m.get_structure_by_material_id(name)  # carbon
        assert structure.is_valid()
    return structure


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


@dataclass
class IMDVaspInputSet(VaspInputSet):
    """IMDGroup variant of VaspInputSet.
    New features:
    1. Potentials do not have to be specified.  By default, use
       VASP-recommended potentials via ase.
    2. New argument FUNCTIONAL (see functionals.yaml) specifying
       functional to be used.  This is similar to vdw parameter in
       VaspInputSet, but also allows setting PBE/PBEsol and other
       non-vdw functionals.
    """
    functional = None
    CONFIG = {'INCAR':
              {
                  # Generic INCAR defaults independes from a given system
                  # Electronic minimization algo
                  'ALGO': 'Normal',
                  # Energy cutoff
                  'ENCUT': 550.0,  # energy cutoff
                  # Smearing, defaults suggested in
                  # https://www.vasp.at/wiki/index.php/ISMEAR
                  'ISMEAR': 0,
                  'SIGMA': 0.04,
                  # FIXME: May we calculate it automatically, from
                  # POTCAR + INCAR data?
                  'NCORE': 16
              },
              'KPOINTS': {'grid_density': 10000},
              'POTCAR_FUNCTIONAL': 'PBE_64',
              'POTCAR': POTCAR_RECOMMENDED}

    def __post_init__(self) -> None:
        assert self.structure.is_valid()

        # Setup default POTCAR.  If an element is missing from
        # POTCAR_RECOMMENED, assume that the potential name is the
        # same with element name.
        for element in self.structure.composition.elements:
            if element.symbol not in self.CONFIG['POTCAR']:
                self.CONFIG['POTCAR'][element.symbol] = element.symbol

        formula = self.structure.reduced_formula
        lattice_type = SpacegroupAnalyzer(self.structure).get_crystal_system()
        space_group =\
            SpacegroupAnalyzer(self.structure).get_space_group_number()
        if "mpid" in self.structure.properties:
            mpid = self.structure.properties["mpid"] + '.'
        else:
            mpid = ''

        self.CONFIG['INCAR']['SYSTEM'] =\
            f'{formula}.{mpid}{lattice_type}.{space_group}'

        super().__post_init__()

        # Do it after parent class initialization, when _config_dict
        # is set
        if isinstance(self.functional, str):
            self.functional = self.functional.lower()
        if self.functional:
            functional_config = _load_yaml_config("functionals")
            if params := functional_config.get(self.functional):
                self._config_dict["INCAR"].update(params)
            else:
                raise KeyError(
                    "Invalid or unsupported functional. " +
                    "Supported functionals are " +
                    ', '.join(functional_config) + "."
                )

    @classmethod
    def input_from_directory(cls, directory: PathLike, **kwargs):
        """Create VASP inputs from directory.
        This is like VaspInputSet.from_prev_calc, but allows reading
        vasp input directory that does not contain VASP outputs.
        """
        try:
            return VaspInputSet.from_prev_calc(directory, **kwargs)
        except Exception:
            vasp_input = VaspInput.from_directory(directory)
            input_set = cls(
                vasp_input['POSCAR'].structure,
                prev_incar=vasp_input['INCAR'],
                prev_kpoints=vasp_input['KPOINTS'],
                **kwargs
                )
            # Copy over POTCAR, if any
            files_to_transfer = {}
            if potcars := sorted(glob(str(Path(directory) / "POTCAR*"))):
                files_to_transfer['POTCAR'] = str(potcars[-1])
            input_set.files_to_transfer.update(files_to_transfer)
            return input_set
        pass


@due.dcite(
    Doi("10.1007/s10570-024-05754-7"),
    description="Understanding of dielectric properties of cellulose",
)
@dataclass
class IMDRelaxCellulose(VaspInputSet):
    """Relaxation input set for cellulose.

    Args:
      structure (Structure, "ibeta", or "ialpha")
        Structure to be used for input.  Either a Structure object, or
        string "ialpha"/"ibeta" for the corresponding cellulose phase,
        as computed in the paper.
      user_kpoints_settings (dict or Kpoints):
        Allow user to override kpoints setting by supplying a
        dict. e.g. {"reciprocal_density": 1000}. User can also supply
        Kpoints object.

    References:
      Yadav, A., Boström, M. & Malyi, O.I. Understanding of dielectric
      properties of cellulose. Cellulose 31, 2783–2794
      (2024). https://doi.org/10.1007/s10570-024-05754-7
    """
    CONFIG = _load_yaml_config("IMDRelaxCellulose")
    force_gamma: bool = True  # Must use gamma-centered k-point grid

    def __post_init__(self) -> None:
        if self.structure == 'ialpha':
            self.structure = _load_cif('cellulose_ialpha')
        if self.structure == 'ibeta':
            self.structure = _load_cif('cellulose_ibeta')
        super().__post_init__()


@dataclass
class IMDGraphite(VaspInputSet):
    """SCF input set for graphite.

    Args:
      user_kpoints_settings (dict or Kpoints):
        Allow user to override kpoints setting by supplying a
        dict. e.g. {"reciprocal_density": 1000}. User can also supply
        Kpoints object.
    """
    CONFIG = _load_yaml_config("IMDGraphite")
    force_gamma: bool = True  # Must use gamma-centered k-point grid

    def __post_init__(self) -> None:
        self.structure = _load_mp('mp-48')
        super().__post_init__()
