"""This module implement useful VASP input sets to be used for the
group research.
"""

import os
import math
import warnings
import logging
from glob import glob
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Self
from pymatgen.io.vasp.sets import VaspInputSet, BadInputSetWarning
from pymatgen.io.vasp.inputs import Potcar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.due import Doi, due
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.calculators.vasp.setups \
    import setups_defaults as ase_potential_defaults
from ase.mep import idpp_interpolate
from IMDgroup.pymatgen.core.structure import\
    merge_structures, structure_interpolate2, structure_is_valid2
from IMDgroup.pymatgen.io.ase.ase import IMDGAseAtomsAdaptor
from IMDgroup.pymatgen.io.vasp.inputs import Incar, _load_yaml_config, nebp, neb_dirs
from IMDgroup.pymatgen.cli.imdg_visualize import\
    write_selective_dynamics_summary_maybe

# ase uses pairs of 'Si': '_suffix'.  Convert them into 'Si': 'Si_suffix'
POTCAR_RECOMMENDED = dict(
    (name, name + suffix)
    for name, suffix in ase_potential_defaults['recommended'].items())


__author__ = "Ihor Radchenko <yantar92@posteo.net>"
MODULE_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def _load_cif(fname):
    return Structure.from_file(f"{MODULE_DIR}/{fname}.cif")


def _load_mp(name):
    with MPRester() as m:
        structure = m.get_structure_by_material_id(name)  # carbon
        assert structure.is_valid()
    return structure


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
    6. Warn if KPOINT density is too high/low
    7. Visualize non-trivial selective dynamic constrains, if any as
       .cif file.
    8. New argument IMAGES defining additional inputsets to be written
       into 00 01 02 ... folders.  Useful when INCAR has IMAGES
       parameter.  When IMAGES are provided, inputset automatically
       uses structure from the first image.  Setting the structure to
       anything other then None or the first image structure will
       raise an error.
    9. New arguments NO_KPOINTS to suppress writing KPOINTS file,
       NO_POTCAR to suppress writing POTCAR file, NO_POSCAR to
       suppress writing POSCAR file, and NO_INCAR to suppress INCAR
       file completely.
    """
    functional: str | None = None
    images: list[Self] | None = None
    no_kpoints: bool = False
    no_potcar: bool = False
    no_poscar: bool = False
    no_incar: bool = False
    __structure: Structure | None = None

    CONFIG = {'INCAR': {}, 'POTCAR_FUNCTIONAL': "PBE_64"}

    @property
    def structure(self):
        """Get set's structure.
        """
        if self.images is not None:
            return self.images[0].structure
        return self.__structure

    @structure.setter
    def structure(self, new_structure):
        if new_structure is self.structure:
            return
        if self.images is not None and\
           new_structure is not None and\
           new_structure is not self.structure:
            raise AttributeError("Cannot set structure for NEB inputset.")
        self.__structure = new_structure

    @property
    def kpoints(self) -> Kpoints | None:
        """The KPOINTS file."""
        if self.no_kpoints:
            return None

        kpoints = super().kpoints

        kpts = kpoints.kpts
        if kpoints.num_kpts == 0 and len(kpts) != 1\
           and not kpts[0] == [1, 1, 1]:
            n_atoms = len(self.structure)
            n_kpoints = math.prod(kpts[0]) * n_atoms
            # 5-10k kpoints/atom is a reasonable number
            # Note that the number is always approximate wrt the
            # target kpoint density because of discretization
            if n_kpoints < 5000:
                warnings.warn(
                    "KPOINTS density is lower than 5000."
                    f"({kpts})"
                    "\nI hope that you know what you are doing.",
                    BadInputSetWarning,
                )
            if n_kpoints > 15000:
                warnings.warn(
                    "KPOINTS density is higher than 15000."
                    f"({kpts})"
                    "\nI hope that you know what you are doing.",
                    BadInputSetWarning,
                )

        return kpoints

    @property
    def incar_updates(self) -> dict:
        """Updates to the INCAR config according to funcational."""
        incar_updates = {}
        if isinstance(self.functional, str):
            self.functional = self.functional.lower()
        if self.functional:
            default_params = Incar.get_recipe("functional", "__defaults")
            incar_updates.update(default_params)
            params = Incar.get_recipe("functional", self.functional)
            incar_updates.update(params)
        return incar_updates

    @property
    def incar(self) -> Incar:
        """The INCAR.  Also, automatically derive SYSTEM name."""
        if self.no_incar:
            return None
        incar = super().incar

        # Empty incar.  Do nothing.
        if incar is None or list(incar) == []:
            return incar

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

        if incar['ENCUT'] < 500.0:
            warnings.warn(
                "ENCUT parameter in lower than default 500."
                f" ({incar['ENCUT']} < 500eV)"
                "\nI hope that you know what you are doing.",
                BadInputSetWarning,
            )
        # Volume/shape relaxation is requested.  Demand increased ENCUT.
        # 550eV recommended for _volume/shape_ relaxation During
        # volume/shape relaxation, initial automatic k-point grid
        # calculated for original volume becomes slightly less accurate
        # unless we increase ENCUT
        elif 'ISIF' in incar:
            if incar['ENCUT'] < 550.0 and\
               incar['ISIF'] != Incar.ISIF_FIX_SHAPE_VOL:
                warnings.warn(
                    "ENCUT parameter is too low for volume/shape relaxation."
                    f" ({incar['ENCUT']} < 550eV)"
                    "\nI hope that you know what you are doing.",
                    BadInputSetWarning,
                )
            elif (incar['ENCUT'] > 500.0 and
                  incar['ISIF'] == Incar.ISIF_FIX_SHAPE_VOL):
                warnings.warn(
                    "ENCUT parameter is too high for position relaxation."
                    f" ({incar['ENCUT']} > 500eV)"
                    "\nI hope that you know what you are doing.",
                    BadInputSetWarning,
                )

        NCORE = incar['NCORE'] if 'NCORE' in incar else None  # pylint:disable invalid-name
        if 'NPAR' in incar:
            # https://www.vasp.at/wiki/index.php/KPAR
            KPAR = incar['KPAR'] if 'KPAR' in incar else 1
            NCORE = incar['NPAR'] * KPAR
        if NCORE == 1:
            warnings.warn(
                "NCORE = 1 is only useful for up to 8 cores. "
                "See https://www.vasp.at/wiki/index.php/NCORE",
                BadInputSetWarning,
            )
        if NCORE is not None and NCORE > 2 and\
           NCORE * 25 > len(self.structure):
            warnings.warn(
                "NCORE/NPAR parameter in the input set is too large"
                f" ({NCORE} (NCORE) * 25 > {len(self.structure)} atoms)"
                "\n See https://www.vasp.at/wiki/index.php/NCORE",
                BadInputSetWarning,
            )
        return incar

    @property
    def poscar(self) -> Poscar:
        """Check structure and return POSCAR."""

        assert self.structure.is_valid()

        # When using selective dynamics, detect bogus fully fixed atoms.
        # for site in self.structure:
        #     if 'selective_dynamics' in site.properties and\
        #        not np.any(site.properties['selective_dynamics']):
        #         # [False, False, False]
        #         warnings.warn(
        #             "Bogus selective dynamics settings: site is fixed:"
        #             f"\n{site}:{site.properties}",
        #             BadInputSetWarning,
        #         )

        return super().poscar

    @property
    def potcar_symbols(self) -> list[str] | None:
        """List of POTCAR symbols.
        Auto-fill missing potentials for elements using VASP
        recommendations."""
        if self.poscar is None:
            return None
        # Setup default POTCAR.  If an element is missing from
        # POTCAR_RECOMMENED, assume that the potential name is the
        # same with element name.
        elements = self.poscar.site_symbols
        for element in elements:
            if 'POTCAR' not in self._config_dict:
                self._config_dict['POTCAR'] = {}
            if element not in self._config_dict['POTCAR']:
                self._config_dict['POTCAR'][element] = \
                    POTCAR_RECOMMENDED[element]\
                    if element in POTCAR_RECOMMENDED else element
        return super().potcar_symbols

    @property
    def potcar(self) -> Potcar | None:
        """The input set's POTCAR."""
        if self.no_potcar:
            return None
        return super().potcar

    def write_input(self, output_dir, **kwargs) -> None:
        """Write a set of VASP input to OUTPUT_DIR."""
        super().write_input(output_dir, **kwargs)
        # Write inputset info
        log_file = os.path.join(output_dir, "IMDVaspInputSet.log")
        with open(log_file, "w", encoding='utf-8') as f:
            for field in fields(self.__class__):
                if field.name not in ['images']:
                    field_value = getattr(self, field.name)
                    f.write(f"{field.name}: {field_value}\n")
        if self.images is None and self.structure is not None:
            write_selective_dynamics_summary_maybe(
                self.structure,
                os.path.join(output_dir, "selective_dynamics.cif")
            )
        # Maybe remove empty INCAR written
        if self.incar is None or list(self.incar) == []:
            f = os.path.join(output_dir, "INCAR")
            if os.path.isfile(f):
                os.remove(f)
        # Maybe remove POSCAR written
        if self.no_poscar:
            f = os.path.join(output_dir, "POSCAR")
            if os.path.isfile(f):
                os.remove(f)
        # NEB input
        if self.images is not None:
            # Write images
            for d, image in zip(neb_dirs(output_dir), self.images):
                image.write_input(d, **kwargs)
                logger.debug(
                    "Writing trajectory file %s",
                    os.path.join(output_dir, 'NEB_trajectory.cif'))
                # Store NEB path snapshot
                trajectory = merge_structures(
                    [img.structure for img in self.images])
                trajectory.to_file(
                    os.path.join(output_dir, 'NEB_trajectory.cif'))
                # Visualize information about fixed/not fixed sites, if any
                write_selective_dynamics_summary_maybe(
                    trajectory,
                    os.path.join(output_dir, 'NEB_fixed_sites.cif')
                )


@dataclass
class IMDDerivedInputSet(IMDVaspInputSet):
    """Inputset derived from an existing Vasp output or input directory.
    Accepts mandatory argument DIRECTORY.
    Optional argument FORCE_PREV_INCAR_FILE (default False), when set
    to True will discard INCAR parameters from vasprun.xml (if any) if
    there is no actual INCAR present in DIRECTORY.
    Optional argument FORCE_PREV_KPOINTS_FILE (default False) does the
    same for KPOINTS.
    """
    directory: str | None = None
    images = None
    force_prev_incar_file: bool = False
    force_prev_kpoints_file: bool = False

    @property
    def incar(self):
        """Return None when previous dir has no INCAR.
        """
        if self.prev_incar is None:
            return None
        return super().incar

    @property
    def kpoints(self):
        """Return None when previous dir has no KPOINTS.
        """
        if self.prev_kpoints is None:
            return None
        return super().kpoints

    @property
    def kpoints_updates(self):
        """Call kpoints_updates from VaspInputSet, but prefer
        prev_kpoints unconditionally.
        """

        if self.prev_kpoints and isinstance(self.prev_kpoints, Kpoints):
            return self.prev_kpoints
        return super().kpoints_updates

    def __post_init__(self) -> None:

        if nebp(self.directory):
            self.images = []
            logger.debug(
                "Found NEB input in %s",
                self.directory
            )
            for subdir in neb_dirs(self.directory):
                # Re-use user-specified class parameters
                # overriding directory
                kwargs = {
                    'directory': subdir,
                    'images': None,
                    # vasprun.xml in NEB directories will combine
                    # local INCAR and parent INCAR, which we do not
                    # want to mix here.
                    'force_prev_incar_file': True,
                    'force_prev_kpoints_file': True,
                }
                params = {k: kwargs.get(k, getattr(self, k))
                          for k in self.__dict__}
                self.images.append(IMDDerivedInputSet(**params))

        # Directory settings take precedence.
        self.inherit_incar = True
        try:
            logger.debug(
                "Reading previous VASP output from %s", self.directory)
            self.override_from_prev_calc(prev_calc_dir=self.directory)
        except ValueError:
            logger.debug("No VASP output found.  Reading input instead")
            # No VASP output found.  Try to ingest VASP input.
            if os.path.isfile(os.path.join(self.directory, "POSCAR")) and\
               self.images is None:
                poscar = Poscar.from_file(
                    os.path.join(self.directory, "POSCAR"),
                    # https://github.com/materialsproject/pymatgen/issues/4140
                    # We do not care about consistency between POSCAR and
                    # POTCAR here.  POTCAR will be re-generated anyway.
                    check_for_potcar=False,
                )
                self.structure = poscar.structure
            elif self.images is not None:
                pass
            else:
                raise ValueError(
                    f"No VASP input found in {self.directory}"
                )

        super().__post_init__()

        if os.path.isfile(os.path.join(self.directory, "KPOINTS")):
            kpoints = Kpoints.from_file(
                os.path.join(self.directory, "KPOINTS")
            )
            self.prev_kpoints = kpoints
        else:
            if self.force_prev_kpoints_file:
                self.prev_kpoints = None

        incar_path = os.path.join(self.directory, "INCAR")
        if os.path.isfile(incar_path):
            # override_from_prev_calc uses vasprun.xml
            # However, as it turns out vasprun.xml may not have all
            # the incar parameters. For example, it does not store NCORE.
            # Force using the actual INCAR file.
            incar = Incar.from_file(incar_path)
            self.prev_incar = incar
        elif self.force_prev_incar_file:
            self.prev_incar = None
        else:
            parent_dir = os.path.dirname(os.path.abspath(self.directory))
            parent_incar_path = os.path.join(parent_dir, "INCAR")
            if nebp(parent_dir) and os.path.isfile(parent_incar_path):
                incar = Incar.from_file(parent_incar_path)
                self.prev_incar = incar

        # self.override_from_prev_calc does not inherit POTCAR.  Force it.
        if (potcars := sorted(glob(str(Path(self.directory) / "POTCAR*")))) and\
           self.poscar is not None:
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
                  # 500eV is the default.
                  # Note: 550eV recommended for _volume/shape_ relaxation
                  # During volume/shape relaxation, initial automatic
                  # k-point grid calculated for original volume
                  # becomes slightly less accurate unless we increase ENCUT
                  'ENCUT': 500.0,  # energy cutoff
                  # Smearing, defaults suggested in
                  # https://www.vasp.at/wiki/index.php/ISMEAR
                  'ISMEAR': 0,
                  'SIGMA': 0.04,
                  # By default, do not write WAVECAR and CHGCAR - save space
                  'LWAVE': False,
                  'LCHARG': False,
              },
              'KPOINTS': {'grid_density': 10000},
              'POTCAR_FUNCTIONAL': 'PBE_64',
              'POTCAR': POTCAR_RECOMMENDED}

    def __post_init__(self) -> None:

        if self.structure is not None:
            if len(self.structure) < self.CONFIG['INCAR']['NCORE']:
                self.CONFIG['INCAR']['NCORE'] = max(
                    2,
                    min(
                        # https://www.vasp.at/wiki/index.php/NCORE
                        # suggests NCORE = 4 for 100 atoms
                        # NCORE = 12-16 for 400 atoms
                        int(len(self.structure)/25),
                        # Never go beyond 16 as it may not fit number
                        # of CPUs in a given node
                        16
                    )
                )

        super().__post_init__()


class IMDNEBVaspInputSetWarning(UserWarning):
    """Warning from IMDNEBVaspInputSet."""


@dataclass
class IMDNEBVaspInputSet(IMDDerivedInputSet):
    """Input set for NEB calculations.
    Accepts two mandatory arguments directory and target_directory for
    the VASP outputs containing the initial and final structures.  We
    demand VASP outputs as the structures have to be well-converged
    for accurate NEB calculations.

    Optional argument FIX_CUTOFF prohibits relaxation of atoms that
    are further away from atoms that move during NEB than FIX_CUTOFF
    angstrom.

    Optional argument METHOD controls interpolation method.
    Possible values:
    - 'IDPP' (default): Use image dependent pair potential method from ASE.
      Method description: Søren Smidstrup, Andreas Pedersen, Kurt
      Stokbro and Hannes Jónsson Chem. Phys. 140, 214106 (2014)
    - 'linear': Use linear interpolation.

    When METHOD is 'linear', optional argument FRAC_TOL controls
    proximity threshold for generated images.  When a distance between
    some atoms in a generated image is less than sum of their radiuses
    times FRAC_TOL, such image is avoided.  Default: 0.5
    """
    target_directory: str | None = None
    fix_cutoff: float | None = None
    frac_tol: float = 0.5
    method: str = 'IDPP'

    # According to Henkelman et al JCP 2000 (doi: 10.1063/1.1329672),
    # the typical number of images is 4-20.  We take smaller number as
    # the default here.  We also use odd number by default as barrier
    # top often lays in the middle and odd number of images has higher
    # chance to be at the top.
    # POTIM is reduced as NEB tends to generate paths passing close to
    # other atoms, causing problems with convergence
    CONFIG = {
        'INCAR': {
            # https://www.vasp.at/wiki/index.php/SPRING
            # says that IBROIN=2 "*usually* fails to converge"
            "IBRION": 1,
            "IMAGES": 5,
            "SPRING": -5,
            "POTIM": 0.25},
        'POTCAR_FUNCTIONAL': "PBE_64"
    }

    @property
    def incar(self) -> Incar:
        """The INCAR.  Also, check POTIM and IMAGES."""
        incar = super().incar

        if incar['POTIM'] > 0.25:
            warnings.warn(
                f"POTIM={incar['POTIM']} parameter is higher than"
                " the default 0.25 for NEB.\n"
                "I hope that you know what you are doing",
                BadInputSetWarning,
            )
        if incar['IMAGES'] == 0:
            warnings.warn(
                "IMAGES=0 makes no sense for NEB",
                BadInputSetWarning,
            )

        if incar['IBRION'] != 1:
            warnings.warn(
                f"IBRION({incar['IBRION']}) ≠ 1.  Forcing IBRION=1\n"
                "See https://www.vasp.at/wiki/index.php/SPRING",
                IMDNEBVaspInputSetWarning
            )
            incar['IBRION'] = 1
        return incar

    def __post_init__(self) -> None:
        # Do not write top-level POSCAR
        self.no_poscar = True

        beg_run = Vasprun(os.path.join(self.directory, 'vasprun.xml'))
        try:
            end_run = Vasprun(os.path.join(
                self.target_directory, 'vasprun.xml'))
        except FileNotFoundError:
            end_run = None
            warnings.warn(
                f"Failed to read Vasprun from {self.target_directory}.  "
                "Falling back to reading POSCAR."
            )
        if end_run is not None:
            assert beg_run.converged and end_run.converged

        super().__post_init__()

        if end_run is None:
            poscar = Poscar.from_file(
                os.path.join(self.target_directory, "POSCAR"))
            self.target_structure = poscar.structure
        else:
            self.target_structure = end_run.final_structure

        if end_run is not None:
            # Make sure that INCAR parameters for start_dir and end_dir
            # are consistent.
            source_incar = beg_run.parameters
            target_incar = end_run.parameters
            diff = source_incar.diff(target_incar)
            if len(diff['Different']) > 0:
                warnings.warn(
                    f"INCARs in {self.directory} and {self.target_directory}"
                    f" are inconsistent: {diff['Different']}",
                    BadInputSetWarning
                )

        self.update_images()

    def write_input(self, output_dir, **kwargs) -> None:
        """Write a set of VASP input to OUTPUT_DIR."""
        super().write_input(output_dir, **kwargs)
        # Save information about the initial/final image inputs.
        log_file = os.path.join(output_dir, "NEB-inputs.txt")
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("NEB path computed between:\n")
            f.write(f"00: {self.directory}\n")
            f.write(f"{len(self.images)-1:02d}: {self.target_directory}\n")

    def update_images(self, beg=None, end=None):
        """Update self.images to be interpolation of BEG..END.
        BEG and END are structures to interpolate between.
        If not provided, BEG is taken from self.structure and
        END is taken from self.target_structure.
        """
        if beg is None:
            beg = self.structure
        if end is None:
            end = self.target_structure
        else:
            self.target_structure = end
        frac_tol = 0 if self.method == 'IDPP' else self.frac_tol
        try:
            str_images = structure_interpolate2(
                beg, end,
                nimages=self.incar["IMAGES"]+1,
                frac_tol=frac_tol, autosort_tol=0.5)
        except ValueError:
            # Auto-sorting sites failed.
            # Fall back to 1-to-1 site matching
            warnings.warn(
                "Automatic match failed.  Assuming 1-to-1 site mapping during structure interpolation",
                BadInputSetWarning
            )
            str_images = structure_interpolate2(
                beg, end,
                nimages=self.incar["IMAGES"]+1,
                frac_tol=frac_tol, autosort_tol=0)

        if self.method != 'IDPP':
            for image in str_images:
                assert structure_is_valid2(image, self.frac_tol)
        self._fix_atoms_maybe(str_images)  # modify by side effect

        if self.method == 'IDPP':
            adaptor = IMDGAseAtomsAdaptor()
            images_ase = [adaptor.get_atoms(i) for i in str_images]
            # idpp_interpolate(images_ase, fmax=0.001)
            # mic=True is important as periodic boundary conditions are not
            # considered between images otherwise
            idpp_interpolate(images_ase, mic=True, traj=None)
            str_images = [adaptor.get_structure(s) for s in images_ase]

        # Setup NEB image VASP inputsets
        self.images = []
        for image in str_images:
            inputset = IMDVaspInputSet(
                no_kpoints=True, no_potcar=True,
                # Avoid pymatgen automatically adding parameters
                no_incar=True)
            # FIXME: We cannot pass structure via structure= parameter because
            # pymatgen has a bug with _structure vs. structure
            # parameters being completely messed up
            # Need to report a bug.  This is a clear example of why
            # there is a problem in the code (see TODO item there)
            inputset.structure = image
            self.images.append(inputset)

    def _fix_atoms_maybe(self, images):
        """Fix atoms further away than self.fix_cutoff from moving atoms.
        Fixing is done by setting selective dynamics for each image in
        IMAGES.  Modify IMAGES in place.
        """
        if self.fix_cutoff is None:
            return None
        moved_idxs = []
        first = images[0]
        last = images[-1]
        for idx in range(len(images[0])):
            dist = first[idx].distance(last[idx])
            if dist >= 0.5:
                moved_idxs.append(idx)
        idxs_to_fix = []
        for idx in moved_idxs:
            for image in images:
                for idx2, site in enumerate(image):
                    dist = image[idx].distance(site)
                    if dist > self.fix_cutoff:
                        if idx2 not in idxs_to_fix:
                            idxs_to_fix.append(idx2)
        for image in images:
            for idx, site in enumerate(image):
                if 'selective_dynamics' not in site.properties:
                    site.properties['selective_dynamics'] =\
                        [True, True, True]
                if idx in idxs_to_fix:
                    site.properties['selective_dynamics'] =\
                        [False, False, False]
        return None


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
