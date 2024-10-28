"""This module implement useful VASP input sets to be used for the
group research.
"""

import os
import warnings
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from monty.serialization import loadfn
from pymatgen.io.vasp.sets import VaspInputSet, BadInputSetWarning
from pymatgen.io.vasp.inputs import VaspInput, Potcar, Kpoints, Incar, Poscar
from pymatgen.util.due import Doi, due
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.calculators.vasp.setups \
    import setups_defaults as ase_potential_defaults

ISIF_RELAX_POS = ISIF_FIX_SHAPE_VOL = 2
ISIF_RELAX_POS_SHAPE_VOL = ISIF_FIX_NONE = 3
ISIF_RELAX_POS_SHAPE = ISIF_FIX_VOL = 4
ISIF_RELAX_SHAPE = IFIX_FIX_POS_VOL = 5
ISIF_RELAX_SHAPE_VOL = ISIF_FIX_POS = 6
ISIF_RELAX_VOL = ISIF_FIX_POS_SHAPE = 7
ISIF_RELAX_POS_VOL = ISIF_FIX_SHAPE = 8

IBRION_IONIC_RELAX_CGA = 2

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
    1. New argument FUNCTIONAL (see functionals.yaml) specifying
       functional to be used.  This is similar to vdw parameter in
       VaspInputSet, but also allows setting PBE/PBEsol and other
       non-vdw functionals.
    2. Automatic SYSTEM name generation.
    3. Structure and input validation
    # FIXME: pymatgen forces PBE, but it ought to be configurable via
    # pmg config. May file a bug report.
    4. Use the latest POTCAR_FUNCTIONAL PBE_64 by default.
    5. Complain when NCORE exceeds the number of sites in the system.
    """
    functional: str | None = None

    @property
    def incar_updates(self) -> dict:
        """Updates to the INCAR config according to funcational."""
        incar_updates = {}
        if isinstance(self.functional, str):
            self.functional = self.functional.lower()
        if self.functional:
            functional_config = _load_yaml_config("functionals")
            if params := functional_config.get(self.functional):
                incar_updates.update(params)
            else:
                raise KeyError(
                    "Invalid or unsupported functional. " +
                    "Supported functionals are " +
                    ', '.join(functional_config) + "."
                )
        return incar_updates

    @property
    def incar(self) -> Incar:
        """The INCAR.  Also, automatically derive SYSTEM name."""
        incar = super().incar

        formula = self.structure.reduced_formula
        lattice_type = SpacegroupAnalyzer(self.structure).get_crystal_system()
        space_group =\
            SpacegroupAnalyzer(self.structure).get_space_group_number()
        if "mpid" in self.structure.properties:
            mpid = self.structure.properties["mpid"] + '.'
        else:
            mpid = ''

        if 'SYSTEM' not in incar:
            incar['SYSTEM'] = f'{formula}.{mpid}{lattice_type}.{space_group}'

        incar.check_params()
        NCORE = incar['NCORE'] if 'NCORE' in incar else None
        if 'NPAR' in incar:
            # https://www.vasp.at/wiki/index.php/KPAR
            KPAR = incar['KPAR'] if 'KPAR' in incar else 1
            NCORE = incar['NPAR'] * KPAR
        if NCORE is not None and NCORE > len(self.structure):
            warnings.warn(
                "NCORE/NPAR parameter in the input set is too large"
                f" ({NCORE} > {len(self.structure)} atoms)",
                BadInputSetWarning,
            )
        return incar

    @property
    def poscar(self) -> Poscar:
        """Check structure and return POSCAR."""
        assert self.structure.is_valid()
        return super().poscar

    @property
    def potcar_symbols(self) -> list[str]:
        """List of POTCAR symbols.
        Auto-fill missing potentials for elements using VASP
        recommendations."""
        # Setup default POTCAR.  If an element is missing from
        # POTCAR_RECOMMENED, assume that the potential name is the
        # same with element name.
        elements = self.poscar.site_symbols
        for element in elements:
            if 'POTCAR' not in self._config_dict:
                self._config_dict['POTCAR'] = {}
            if element not in self._config_dict['POTCAR']:
                self._config_dict['POTCAR'][element] = \
                    POTCAR_RECOMMENDED[element] if element in POTCAR_RECOMMENDED\
                    else element
        return super().potcar_symbols


@dataclass
class IMDDerivedInputSet(IMDVaspInputSet):
    """Inputset derived from an existing Vasp output or input directory.
    Accepts mandatory argument DIRECTORY.
    """
    directory: str | None = None

    CONFIG = {'INCAR': {}, 'POTCAR_FUNCTIONAL': "PBE_64"}

    @property
    def kpoints_updates(self):
        """Call kpoints_updates from VaspInputSet, but prefer
        prev_kpoints unconditionally.
        """

        if self.prev_kpoints and isinstance(self.prev_kpoints, Kpoints):
            return self.prev_kpoints
        return super().kpoints_updates

    def __post_init__(self) -> None:
        # Directory settings take precedence.
        self.inherit_incar = True
        try:
            self.override_from_prev_calc(prev_calc_dir=self.directory)
            super().__post_init__()
            # override_from_prev_calc uses vasprun.xml
            # However, as it turns out vasprun.xml may not have all
            # the incar parameters. For example, it does not store NCORE.
            # Force using the actual INCAR file.
            vasp_input = VaspInput.from_directory(self.directory)
            self.prev_incar = vasp_input['INCAR']
        except ValueError:
            # No VASP output found.  Try to ingest VASP input.
            vasp_input = VaspInput.from_directory(self.directory)
            self.structure = vasp_input['POSCAR'].structure

            super().__post_init__()

            self.prev_incar = vasp_input['INCAR']
            self.prev_kpoints = vasp_input['KPOINTS']

        # self.override_from_prev_calc does not inherit POTCAR.  Force it.
        if potcars := sorted(glob(str(Path(self.directory) / "POTCAR*"))):
            # Override defaults with POTCAR data
            # We still want to transfer the file explicitly to
            # make sure that any non-standard POTCARS are not
            # going to be broken
            potcar = Potcar.from_file(str(potcars[-1]))
            potcar_dict = {}
            for el, symbol in \
                    zip(self.poscar.site_symbols,
                        potcar.symbols):
                potcar_dict[el] = symbol
            self._config_dict['POTCAR'] = potcar_dict
            # potcar.functional is actually not what pymatgen means by
            # POTCAR_FUNCTIONAL
            # self._config_dict['POTCAR_FUNCTIONAL'] = potcar.functional


@dataclass
class IMDStandardVaspInputSet(IMDVaspInputSet):
    """Standard input set for IMDGroup.
    New features:
    1. Potentials do not have to be specified.  By default, use
       VASP-recommended potentials via ase.
    """
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
                  'NCORE': 16,
                  # By default, do not write WAVECAR and CHGCAR - save space
                  'LWAVE': False,
                  'LCHARG': False,
              },
              'KPOINTS': {'grid_density': 10000},
              'POTCAR_FUNCTIONAL': 'PBE_64',
              'POTCAR': POTCAR_RECOMMENDED}


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
