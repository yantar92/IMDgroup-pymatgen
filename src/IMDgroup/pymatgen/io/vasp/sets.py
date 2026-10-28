# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-pymatgen package
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""This module implements useful VASP input sets to be used for the group research."""

from __future__ import annotations

import os
import math
import warnings
import logging
import copy
from xml.etree.ElementTree import ParseError
from pathlib import Path
from dataclasses import dataclass, fields, field
import numpy as np
from pymatgen.core import Species, DummySpecies, Structure
from pymatgen.io.vasp.sets import VaspInputSet, BadInputSetWarning
from pymatgen.io.vasp.inputs import (
    BadPoscarWarning,
    Kpoints,
    PmgVaspPspDirError,
    Poscar,
    Potcar,
)
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.vasp.setups \
    import setups_defaults as ase_potential_defaults
from ase.mep import idpp_interpolate
from IMDgroup.pymatgen.core.structure import\
    merge_structures, structure_interpolate2, structure_is_valid2
from IMDgroup.pymatgen.io.vasp.incar_toml import (
    read_incar_toml,
    write_incar_toml,
)
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

# ase uses pairs of 'Si': '_suffix'.  Convert them into 'Si': 'Si_suffix'
POTCAR_RECOMMENDED = dict(
    (name, name + suffix)
    for name, suffix in ase_potential_defaults['recommended'].items())
# Fix https://gitlab.com/ase/ase/-/work_items/657
POTCAR_RECOMMENDED['W'] = 'W_sv'

# Group-standard default POTCAR functional (PBE_64 release).
POTCAR_FUNCTIONAL_DEFAULT = 'PBE_64'

# Group-standard default for KPOINTS.grid_density.
KPOINTS_GRID_DENSITY = 10000

# Reference ENCUT is derived from the largest POTCAR ENMAX.  The base
# reference is the VASP default ENCUT (the largest ENMAX).  Shape/volume
# relaxation uses a 30% margin on top because the automatic k-point grid,
# calculated for the original cell volume, becomes slightly less accurate
# as the cell changes shape and volume.
ENCUT_RELAX_FACTOR = 1.3

# Fallback reference ENCUT values used when no POTCAR is available.
ENCUT_DEFAULT = 500.0
ENCUT_RELAX_DEFAULT = 550.0

__author__ = "Ihor Radchenko <yantar92@posteo.net>"
logger = logging.getLogger(__name__)


def write_selective_dynamics_summary_maybe(structure, fname):
    """Visualize site constraints and write a CIF file if non-trivial.

    The CIF uses species substitution for visual cues:
    Fe = fully fixed, Co = partially fixed, Ni = not fixed,
    X = no selective dynamics assigned.

    A site with no ``selective_dynamics`` (missing key or ``None``) is
    marked X rather than silently treated as fixed or free, so that an
    inconsistent assignment is highlighted.

    Args:
        structure: Structure with optional ``selective_dynamics``
            site properties.
        fname: Output filename for the CIF.

    Returns:
        bool: True if the file was written (non-trivial constraints
        were found), False otherwise.
    """
    has_fixed = False
    structure = structure.copy()
    for site in structure:
        site.label = None
        sd = site.properties.get('selective_dynamics', None)
        if sd is not None and np.array_equal(sd, [False, False, False]):
            has_fixed = True
            site.species = Species('Fe')  # fixed
        elif sd is not None and False in sd:
            has_fixed = True
            site.species = Species('Co')  # partially fixed
        elif sd is None:
            site.species = DummySpecies('X')  # no selective dynamics assigned
        else:
            site.species = Species('Ni')  # not fixed
    if has_fixed:
        logger.debug(
            "Writing selective dynamics visualization to %s",
            fname
        )
        structure.to_file(fname)
        return True
    return False


@dataclass
class IMDVaspInputSet(VaspInputSet):
    """IMDGroup variant of VaspInputSet.

    Key additions over pymatgen's VaspInputSet:

    1. ``functional`` argument for specifying the exchange-correlation
       functional (see ``functionals.yaml``).
    2. Automatic SYSTEM name generation from formula, lattice type,
       and space group.
    3. Structure and input validation (warnings for KPOINTS density,
       low ENCUT, conflicting NCORE/NPAR).
    4. Default POTCAR_FUNCTIONAL ``PBE_64``.
    5. Visualization of non-trivial selective dynamics as a CIF file.
    6. ``images`` argument for NEB input sets (writes 00, 01, ...
       subdirectories).
    7. ``no_kpoints``, ``no_potcar``, ``no_poscar``, ``no_incar``
       flags to suppress writing individual files.
    8. Writes an ``INCAR.toml`` recording non-default metadata
       (POTCAR mapping, k-point density, functional) so that ``gorun``
       and other tools can reproduce the written inputs.
    """
    functional: str | None = None
    images: list[IMDVaspInputSet] | None = None
    name: str | None = None
    no_kpoints: bool = False
    no_potcar: bool = False
    no_poscar: bool = False
    no_incar: bool = False
    # __structure: Structure | None = None

    CONFIG = {'INCAR': {}, 'POTCAR_FUNCTIONAL': POTCAR_FUNCTIONAL_DEFAULT}

    # @property
    # def structure(self):
    #     """Get set's structure.
    #     """
    #     if self.images is not None:
    #         structure = self.images[0].structure
    #     else:
    #         structure = self.__structure
    #     # Group species in the structure to avoid duplicate species in POSCAR
    #     # https://github.com/materialsproject/pymatgen/issues/1633
    #     if structure is not None:
    #         # Note that get_sorted_structure uses built-in sort, which
    #         # is stable. So, we will preserve species order for the
    #         # same species.
    #         structure = structure.get_sorted_structure(
    #             key=lambda site: site.species.average_electroneg)
    #     return structure

    # @structure.setter
    # def structure(self, new_structure: None | Structure):
    #     if new_structure is self.structure:
    #         return
    #     if self.images is not None and\
    #        new_structure is not None and\
    #        new_structure != self.structure:
    #         raise AttributeError("Cannot set structure for NEB inputset.")
    #     self.__structure = new_structure

    @property
    def kpoints(self) -> Kpoints | None:
        """KPOINTS for the input set.

        When ``no_kpoints`` is True, returns None.  Otherwise warns
        if the KPOINTS density is below 5000 or above 15000
        k-points/atom.
        """
        assert self.structure is not None
        if self.no_kpoints:
            return None

        kpoints = super().kpoints
        assert kpoints is not None

        kpts = kpoints.kpts
        if kpoints.num_kpts == 0 and len(kpts) == 1\
           and not kpts[0] == (1, 1, 1):
            n_atoms = len(self.structure)
            n_kpoints = math.prod(kpts[0]) * n_atoms
            # 5-10k kpoints/atom is a reasonable number
            # Note that the number is always approximate wrt the
            # target kpoint density because of discretization
            if n_kpoints < 5000:
                warnings.warn(
                    f"KPOINTS density is lower than 5000: {n_kpoints} "
                    f"(from a {kpts[0]} mesh with {n_atoms} atoms).\n"
                    "I hope that you know what you are doing.",
                    BadInputSetWarning,
                )
            if n_kpoints > 15000:
                warnings.warn(
                    f"KPOINTS density is higher than 15000: {n_kpoints} "
                    f"(from a {kpts[0]} mesh with {n_atoms} atoms).\n"
                    "I hope that you know what you are doing.",
                    BadInputSetWarning,
                )

        return kpoints

    @property
    def incar_updates(self) -> dict:
        """INCAR updates derived from the functional choice."""
        incar_updates = {}
        if isinstance(self.functional, str):
            self.functional = self.functional.lower()
        if self.functional:
            default_params = Incar.get_recipe("functional", "__defaults")
            incar_updates.update(default_params)
            params = Incar.get_recipe("functional", self.functional)
            incar_updates.update(params)
        return incar_updates

    def _encut_reference(self) -> tuple[float, float]:
        """Return ``(default, relax)`` reference ENCUT values.

        When a POTCAR can be resolved, the base reference is the VASP
        default ENCUT (the largest POTCAR ``ENMAX``) and the relaxation
        reference adds a 30% margin; otherwise the group defaults
        (``ENCUT_DEFAULT``/``ENCUT_RELAX_DEFAULT``) are used.
        """
        try:
            potcar = self.potcar
        except (FileNotFoundError, ValueError, KeyError):
            potcar = None
        if potcar:
            enmax = max(single.enmax for single in potcar)
            return enmax, ENCUT_RELAX_FACTOR * enmax
        return ENCUT_DEFAULT, ENCUT_RELAX_DEFAULT

    @property
    def incar(self) -> Incar | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """INCAR for the input set.

        Automatically derives a SYSTEM name from formula, lattice type,
        and space group.  Warns about low ENCUT settings and when both
        NCORE and NPAR are set.
        """
        if self.no_incar:
            return None
        incar = super().incar
        # Use IMDG version of Incar class
        incar = Incar(incar)

        assert self.structure is not None
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

        default_encut, relax_encut = self._encut_reference()

        if incar['ENCUT'] < default_encut:
            warnings.warn(
                f"ENCUT parameter is lower than the reference "
                f"{default_encut:.1f} eV ({incar['ENCUT']:.1f} eV)."
                "\nI hope that you know what you are doing.",
                BadInputSetWarning,
            )
        # Volume/shape relaxation is requested.  Demand increased ENCUT.
        # During volume/shape relaxation, initial automatic k-point grid
        # calculated for original volume becomes slightly less accurate
        # unless we increase ENCUT
        elif 'ISIF' in incar:
            if incar['ENCUT'] < relax_encut and\
               incar['ISIF'] != Incar.ISIF_FIX_SHAPE_VOL:
                warnings.warn(
                    "ENCUT parameter is too low for volume/shape relaxation."
                    f" ({incar['ENCUT']:.1f} < {relax_encut:.1f} eV)"
                    "\nI hope that you know what you are doing.",
                    BadInputSetWarning,
                )
            # elif (incar['ENCUT'] > 500.0 and
            #       incar['ISIF'] == Incar.ISIF_FIX_SHAPE_VOL):
            #     warnings.warn(
            #         "ENCUT parameter is too high for position relaxation."
            #         f" ({incar['ENCUT']} > 500eV)"
            #         "\nI hope that you know what you are doing.",
            #         BadInputSetWarning,
            #     )

        if 'NCORE' in incar and 'NPAR' in incar:
            warnings.warn(
                f"Both NCORE({incar['NCORE']}) "
                f"and NPAR({incar['NPAR']}) are set. "
                "NCORE will be ignored.  "
                "See https://www.vasp.at/wiki/index.php/NCORE",
                BadInputSetWarning,
            )
        return incar

    @property
    def poscar(self) -> Poscar:
        """POSCAR for the input set.

        Validates the structure before generating the POSCAR.
        """

        assert self.structure is not None
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
    def potcar_symbols(self) -> list[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """List of POTCAR symbols.

        Auto-fills missing element potentials using ASE-recommended
        defaults.
        """
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
    def potcar(self) -> Potcar | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """POTCAR for the input set.

        When ``no_potcar`` is True, returns None.
        """
        if self.no_potcar:
            return None
        return super().potcar

    def _potcar_toml_mapping(self) -> dict[str, str]:
        """Resolved element -> full POTCAR name mapping.

        Mirrors the written POTCAR file: one entry per species in the
        POSCAR, keyed by element symbol with the resolved potential
        name as the value.
        """
        potcar_symbols = self.potcar_symbols
        return dict(zip(self.poscar.site_symbols, potcar_symbols))

    def _kpoints_grid_density(self) -> float | None:
        """Return the grid density used to generate KPOINTS, or None.

        Returns None when KPOINTS were not generated from a density
        recipe (an explicit grid or an inherited KPOINTS file has no
        single density value to record).
        """
        user = self.user_kpoints_settings
        if isinstance(user, dict) and 'grid_density' in user:
            return user['grid_density']
        if isinstance(user, Kpoints):
            return None
        updates = self.kpoints_updates
        if isinstance(updates, dict) and 'grid_density' in updates:
            return updates['grid_density']
        if isinstance(updates, Kpoints):
            return None
        kconfig = self._config_dict.get('KPOINTS', {})
        if isinstance(kconfig, dict) and 'grid_density' in kconfig:
            return kconfig['grid_density']
        return None

    def _write_incar_toml_metadata(self, output_dir: Path) -> None:
        """Write ``INCAR.toml`` recording non-default constructor metadata.

        Records the resolved POTCAR mapping, the k-point grid density
        recipe, and the exchange-correlation functional so that ``gorun``
        and other tools can reproduce the written POTCAR/KPOINTS
        consistently.  The file is merged with any existing
        ``INCAR.toml``: foreign content is preserved, and only values
        that differ from the group defaults are written.
        """
        path = Path(output_dir) / 'INCAR.toml'
        raw_existing = read_incar_toml(path)

        potcar_mapping = self._potcar_toml_mapping()

        values: dict[str, object] = {
            'POTCAR': dict(potcar_mapping),
        }
        if (density := self._kpoints_grid_density()) is not None:
            values['KPOINTS'] = {'grid_density': density}
        functional: dict[str, object] = {
            'potcar_functional': self.potcar_functional,
        }
        if self.functional is not None:
            functional['functional'] = self.functional
        values['FUNCTIONAL'] = functional

        # The [POTCAR] table is fully owned by pymatgen: its defaults
        # cover every element seen now or previously, so stale or removed
        # entries are dropped and only non-default values are written.
        raw_potcar = raw_existing.get('POTCAR', {})
        potcar_defaults = dict(POTCAR_RECOMMENDED)
        for element in set(potcar_mapping) | set(
                raw_potcar if isinstance(raw_potcar, dict) else {}):
            potcar_defaults.setdefault(element, element)

        defaults: dict[str, object] = {
            'POTCAR': potcar_defaults,
            'KPOINTS': {'grid_density': KPOINTS_GRID_DENSITY},
            'FUNCTIONAL': {'potcar_functional': POTCAR_FUNCTIONAL_DEFAULT},
        }

        write_incar_toml(
            values,
            raw_existing=raw_existing,
            defaults=defaults,
            path=path,
            bootstrap=False,
        )

    def write_input(self, output_dir, *args, **kwargs) -> None:
        """Write VASP input files to a directory.

        In addition to standard pymatgen behaviour, writes an
        ``IMDVaspInputSet.log`` file, an ``INCAR.toml`` recording
        non-default metadata (POTCAR mapping, k-point density,
        functional), and, for NEB runs, the image subdirectories and a
        trajectory CIF.

        Args:
            output_dir: Target directory for the input files.

        Any extra positional and keyword arguments are forwarded to
        ``VaspInputSet.write_input``.

        When ``PMG_VASP_PSP_DIR`` is unconfigured, ``write_input``
        degrades to ``potcar_spec=True`` instead of failing: it emits a
        warning and writes ``POTCAR.spec`` (the symbol list) rather than
        POTCAR content, leaving ``INCAR.toml`` with the POTCAR mapping so
        that ``gorun`` can regenerate the POTCAR on a cluster where the
        pseudopotential library is available.  A configured-but-missing
        ``PMG_VASP_PSP_DIR`` raises instead: the user intended to
        configure it, so proceeding silently would hide a setup mistake.
        """
        try:
            super().write_input(output_dir, *args, **kwargs)
        except PmgVaspPspDirError:
            # PMG_VASP_PSP_DIR is not configured at all.  Degrade to
            # POTCAR.spec so create/derive still work without a local
            # pseudopotential library; gorun regenerates the POTCAR on
            # the cluster where the library is available.
            warnings.warn(
                "PMG_VASP_PSP_DIR is not set; writing POTCAR.spec instead "
                "of POTCAR content.  Configure it with: "
                "pmg config --add PMG_VASP_PSP_DIR /path/to/potentials",
                UserWarning,
                stacklevel=2,
            )
            kwargs = {**kwargs, "potcar_spec": True}
            super().write_input(output_dir, *args, **kwargs)
        except FileNotFoundError as exc:
            # pymatgen reports a configured-but-missing PSP dir as
            # "PMG_VASP_PSP_DIR='...' does not exist."  The user intended
            # to configure it, so fail loudly instead of proceeding.
            # A missing POTCAR file inside a valid dir says "You do not
            # have the right POTCAR..." instead; that one is already
            # actionable as-is.
            if "PMG_VASP_PSP_DIR" in str(exc) and "does not exist" in str(exc):
                raise FileNotFoundError(
                    f"{exc} Update it with: "
                    "pmg config --add PMG_VASP_PSP_DIR /path/to/potentials"
                ) from exc
            raise
        output_dir = Path(output_dir)
        # Write inputset info
        log_file = output_dir / "IMDVaspInputSet.log"
        with open(log_file, "w", encoding='utf-8') as f:
            for field in fields(self.__class__):
                if field.name not in ['images']:
                    field_value = getattr(self, field.name)
                    f.write(f"{field.name}: {field_value}\n")
        # Record non-default metadata in INCAR.toml so that gorun can
        # reproduce the POTCAR/KPOINTS when regenerating input files.
        # ``no_potcar`` means "no POTCAR is written here" (for example,
        # NEB image subdirectories whose parent owns the POTCAR), so the
        # INCAR.toml metadata is skipped too: writing a per-image mapping
        # would duplicate, and could conflict with, the parent's mapping.
        if not self.no_potcar:
            self._write_incar_toml_metadata(output_dir)
        if self.images is None and self.structure is not None:
            write_selective_dynamics_summary_maybe(
                self.structure,
                output_dir / "selective_dynamics.cif"
            )
        # Maybe remove empty INCAR written
        if self.incar is None or not list(self.incar):
            (output_dir / "INCAR").unlink(missing_ok=True)
        # Maybe remove POSCAR written
        if self.no_poscar:
            (output_dir / "POSCAR").unlink(missing_ok=True)
        # NEB input
        if self.images is not None:
            # Write images
            if incar := self.incar:
                image_dirs = incar.image_dir_names()
            else:
                image_dirs = None
            if not image_dirs:
                warnings.warn(
                    "NEB images were provided but IMAGES is not set in the "
                    "INCAR, so no image subdirectories will be written.",
                    BadInputSetWarning,
                )
                image_dirs = []
            for d, image in zip(image_dirs, self.images):
                image.write_input(output_dir / d, *args, **kwargs)
                logger.debug(
                    "Writing trajectory file %s",
                    output_dir / 'NEB_trajectory.cif')
                # Store NEB path snapshot
            image_structures = []
            for img in self.images:
                s = img.structure
                if s is None:
                    raise ValueError(
                        "Cannot build NEB trajectory: an image has no structure")
                image_structures.append(s)
            trajectory = merge_structures(image_structures)
            trajectory.to_file(str(output_dir / 'NEB_trajectory.cif'))
            # Visualize information about fixed/not fixed sites, if any
            write_selective_dynamics_summary_maybe(
                trajectory,
                output_dir / 'NEB_fixed_sites.cif'
            )


@dataclass
class IMDDerivedInputSet(IMDVaspInputSet):
    """Input set derived from an existing VASP output or input directory.

    Unlike plain ``IMDVaspInputSet``, this class inherits settings
    (INCAR, KPOINTS, POTCAR, structure) from a previous calculation.

    Key additions:

    - ``directory`` (mandatory): Source directory with VASP output/input.
    - ``force_prev_incar_file``: When True, discard INCAR settings from
      ``vasprun.xml`` if no actual ``INCAR`` file is present.
    - ``force_prev_kpoints_file``: Same for KPOINTS.
    - ``inherit_prev_incarpy``: When True, copy ``INCAR.py`` from source.
    - ``INCAR.[0-9]*`` files are always copied (used by gorun workflows).
    """
    directory: str | IMDGVaspDir = field(kw_only=True)
    force_prev_incar_file: bool = False
    force_prev_kpoints_file: bool = False
    inherit_prev_incarpy: bool = False

    @property
    def incar(self):
        """INCAR for the derived input set.

        Returns None when ``force_prev_incar_file`` is True and the
        previous directory has no INCAR file.
        """
        if self.prev_incar is None and self.force_prev_incar_file:
            return None
        return super().incar

    @property
    def kpoints(self):
        """KPOINTS for the derived input set.

        Returns None when ``force_prev_kpoints_file`` is True and the
        previous directory has no KPOINTS file, or when the previous
        INCAR uses ``KSPACING``.
        """
        if self.prev_kpoints is None and self.force_prev_kpoints_file:
            return None
        assert not isinstance(self.prev_incar, str)
        if self.prev_incar and self.prev_incar.get('KSPACING') is not None:
            return None
        return super().kpoints

    @property
    def kpoints_updates(self):  # type: ignore[reportIncompatibleMethodOverride]
        """KPOINTS updates, preferring prev_kpoints unconditionally."""
        # FIXME: Upstream ``VaspInputSet.kpoints_updates`` is annotated
        # to return ``dict``, but its ``kpoints`` getter explicitly
        # handles a ``Kpoints`` return (see the
        # ``isinstance(kpoints_updates, Kpoints)`` branch).  The stale
        # annotation makes Pyright flag this override as incompatible.
        # Need to report a bug: the annotation should be
        # ``dict | Kpoints``.
        if self.prev_kpoints and isinstance(self.prev_kpoints, Kpoints):
            return self.prev_kpoints
        return super().kpoints_updates

    def __post_init__(self) -> None:
        # FIXME: We should make heaver use of caching IMDGVaspDir

        if isinstance(self.directory, IMDGVaspDir):
            self._vaspdir = self.directory
            self.directory = self._vaspdir.path
        else:
            self._vaspdir = IMDGVaspDir(self.directory)

        if self._vaspdir.nebp:
            self.images = []
            logger.debug(
                "Found NEB input in %s",
                self.directory
            )
            neb_dirs = self._vaspdir.neb_dirs()
            assert neb_dirs is not None
            for subdir in neb_dirs:
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
                # Re-use the user-specified constructor arguments for the
                # image sub-sets, overriding the directory/image/force flags.
                params = {
                    f.name: kwargs.get(f.name, getattr(self, f.name))
                    for f in fields(self.__class__)
                    if f.init and f.name != 'structure'
                }
                self.images.append(type(self)(**params))

        # Directory settings take precedence.
        self.inherit_incar = True
        try:
            logger.debug(
                "Reading previous VASP output from %s", self.directory)
            for extra_incar in Path(self.directory).glob('INCAR.[0-9]*'):
                logger.info("Copying additional %s", extra_incar)
                self.files_to_transfer[str(extra_incar.name)] = extra_incar
            incarpy = Path(self.directory) / "INCAR.py"
            if self.inherit_prev_incarpy and incarpy.is_file():
                logger.info("Copying additional INCAR.py")
                self.files_to_transfer[str(incarpy.name)] = incarpy
            incartoml = Path(self.directory) / "INCAR.toml"
            if incartoml.is_file():
                logger.info("Copying additional INCAR.toml")
                self.files_to_transfer[str(incartoml.name)] = incartoml
            self.override_from_prev_calc(prev_calc_dir=self.directory)
        except (ValueError, ParseError) as exc:
            if os.path.isfile(os.path.join(self.directory, "CONTCAR")):
                structure_file = os.path.join(self.directory, "CONTCAR")
            else:
                logger.debug("No VASP output found.  Reading input instead")
                structure_file = os.path.join(self.directory, "POSCAR")
            # No VASP output found.  Try to ingest VASP input.
            if os.path.isfile(structure_file) and\
               self.images is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", BadPoscarWarning)
                    try:
                        poscar = Poscar.from_file(
                            structure_file,
                            # https://github.com/materialsproject/pymatgen/issues/4140
                            # We do not care about consistency between POSCAR and
                            # POTCAR here.  POTCAR will be re-generated anyway.
                            check_for_potcar=False,
                        )
                    except BadPoscarWarning:
                        # POSCAR does not contain element info
                        # Try reading from POTCAR after all
                        poscar = Poscar.from_file(
                            structure_file,
                            check_for_potcar=True,
                        )
                self.structure = poscar.structure
            elif self.images is not None:
                pass
            else:
                raise ValueError(
                    f"No VASP input found in {self.directory}"
                ) from exc

        super().__post_init__()

        if os.path.isfile(os.path.join(self.directory, "KPOINTS")):
            kpoints = Kpoints.from_file(
                os.path.join(self.directory, "KPOINTS")
            )
            self.prev_kpoints = kpoints
        elif self.force_prev_kpoints_file:
            self.prev_kpoints = None
        else:
            parent_dir = os.path.dirname(os.path.abspath(self.directory))
            parent_kpoints_path = os.path.join(parent_dir, "KPOINTS")
            if IMDGVaspDir(parent_dir).nebp and os.path.isfile(parent_kpoints_path):
                kpoints = Kpoints.from_file(parent_kpoints_path)
                self.prev_kpoints = kpoints

        incar_path = os.path.join(self.directory, "INCAR")
        if os.path.isfile(incar_path):
            # override_from_prev_calc uses vasprun.xml
            # However, as it turns out vasprun.xml may not have all
            # the incar parameters. For example, it does not store NCORE.
            # Force using the actual INCAR file.
            incar = Incar.from_file(incar_path)
            # FIXME: pymatgen's type definition is too narrow.
            self.prev_incar = incar   # type:ignore[assignment]
        elif self.force_prev_incar_file:
            self.prev_incar = None
        else:
            parent_dir = os.path.dirname(os.path.abspath(self.directory))
            parent_incar_path = os.path.join(parent_dir, "INCAR")
            if IMDGVaspDir(parent_dir).nebp and os.path.isfile(parent_incar_path):
                incar = Incar.from_file(parent_incar_path)
                # FIXME: pymatgen's type definition is too narrow.
                self.prev_incar = incar   # type:ignore[assignment]

        # self.override_from_prev_calc does not inherit POTCAR.  Force it.
        # POTCAR precedence when deriving:
        #   1. a POTCAR file in the source dir wins (the user may have
        #      swapped in a custom potential);
        #   2. otherwise INCAR.toml [POTCAR] non-default entries are used;
        #   3. otherwise the POTCAR_RECOMMENDED defaults apply (auto-filled
        #      in potcar_symbols).
        # Match only the canonical POTCAR file (POTCAR or POTCAR.gz).
        potcar_path = None
        for name in ("POTCAR", "POTCAR.gz"):
            candidate = Path(self.directory) / name
            if candidate.is_file():
                potcar_path = candidate
                break
        if potcar_path is not None and self.poscar is not None:
            # Override defaults with POTCAR data
            # We still want to transfer the file explicitly to
            # make sure that any non-standard POTCARS are not
            # going to be broken
            potcar = Potcar.from_file(str(potcar_path))
            potcar_dict = {}
            for el, symbol in \
                    zip(self.poscar.site_symbols,
                        potcar.symbols):
                potcar_dict[el] = symbol
            self._config_dict['POTCAR'] = potcar_dict
        else:
            # No POTCAR file: inherit the non-default POTCAR mapping from
            # the source INCAR.toml so a no-POTCAR (potcar_spec) derived
            # dir keeps propagating a custom mapping down the chain.
            # NOTE: this duplicates the [POTCAR] parsing in
            # IMDgroup.gorun.cleanVASP.potcar_symbols_map (and the pair
            # validation in imdg_derive._validate_potcar_pair); consider
            # consolidating read+validate into one helper later.
            toml = read_incar_toml(Path(self.directory) / "INCAR.toml")
            potcar_table = toml.get("POTCAR")
            if isinstance(potcar_table, dict):
                self._config_dict.setdefault("POTCAR", {}).update(potcar_table)


@dataclass
class IMDStandardVaspInputSet(IMDVaspInputSet):
    """Standard input set for IMDGroup.

    Uses VASP-recommended potentials from ASE by default.  Potentials
    do not need to be specified explicitly.
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
                  # https://www.vasp.at/wiki/index.php/NCORE has some
                  # recommendations, but they are not really universal.
                  # In particular, too small values may severely
                  # degrade CPU utilization on supercomputers.
                  # So, we use larger value as the default.
                  'NCORE': 16,
                  'NELMIN': 6,
              },
              'KPOINTS': {'grid_density': KPOINTS_GRID_DENSITY},
              'POTCAR_FUNCTIONAL': POTCAR_FUNCTIONAL_DEFAULT,
              'POTCAR': POTCAR_RECOMMENDED}


class IMDNEBVaspInputSetWarning(UserWarning):
    """Warning emitted by IMDNEBVaspInputSet."""


@dataclass
class IMDNEBVaspInputSet(IMDDerivedInputSet):
    """Input set for NEB (Nudged Elastic Band) calculations.

    Requires two directories: the source (``directory``) and the
    target (``target_directory``) containing well-converged VASP
    outputs for the initial and final structures.

    References:
        IDPP: S. Smidstrup et al., J. Chem. Phys. 140, 214106 (2014).
    """
    target_directory: str | IMDGVaspDir = field(kw_only=True)
    fix_cutoff: float | None = None
    frac_tol: float = 0.5
    method: str = 'IDPP'

    # According to Henkelman et al JCP 2000 (doi: 10.1063/1.1329672),
    # the typical number of images is 4-20.  We take smaller number as
    # the default here.  We also use odd number by default as barrier
    # top often lays in the middle and odd number of images has higher
    # chance to be at the top.
    CONFIG = {
        'INCAR': {
            # https://www.vasp.at/wiki/index.php/SPRING
            # says that IBROIN=2 "*usually* fails to converge"
            "IBRION": 1,
            "IMAGES": 5,
            "SPRING": -5},
        'POTCAR_FUNCTIONAL': POTCAR_FUNCTIONAL_DEFAULT
    }

    @property
    def incar(self) -> Incar | None:
        """INCAR for the NEB run.

        Warns when IMAGES=0 or IBRION != 1, and forces IBRION=1
        (required for NEB in VASP).
        """
        incar = super().incar

        if incar and incar['IMAGES'] == 0:
            warnings.warn(
                "IMAGES=0 makes no sense for NEB",
                BadInputSetWarning,
            )

        if incar and incar['IBRION'] != 1:
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

        super().__post_init__()

        beg_run = Vasprun(os.path.join(self.directory, 'vasprun.xml'))
        try:
            end_run = Vasprun(os.path.join(
                self.target_directory, 'vasprun.xml'))
        except FileNotFoundError:
            end_run = None
            warnings.warn(
                f"Failed to read Vasprun from {os.path.relpath(self.target_directory)}.  "
                "Falling back to reading POSCAR."
            )
        if end_run is not None:
            assert beg_run.converged and end_run.converged

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
                    f"INCARs in {self.directory} and {os.path.relpath(self.target_directory)}"
                    f" are inconsistent: {diff['Different']}",
                    BadInputSetWarning
                )

                self.update_images()

    def write_input(self, output_dir, *args, **kwargs) -> None:
        """Write NEB input files to a directory.

        In addition to standard behaviour, writes a ``NEB-inputs.txt``
        file recording the initial and final image source directories.

        Args:
            output_dir: Target directory for the input files.

        Any extra positional and keyword arguments are forwarded to
        ``VaspInputSet.write_input``.
        """
        assert self.images is not None
        super().write_input(output_dir, *args, **kwargs)
        # Save information about the initial/final image inputs.
        log_file = os.path.join(output_dir, "NEB-inputs.txt")
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("NEB path computed between:\n")
            f.write(f"00: {self.directory}\n")
            f.write(f"{len(self.images) - 1:02d}: {self.target_directory}\n")

    def update_images(self, beg=None, end=None, **kwargs):
        """Update NEB images by interpolating between start and end structures.

        Args:
            beg: Starting structure.  Defaults to ``self.structure``.
            end: Ending structure.  Defaults to ``self.target_structure``.
            **kwargs: Forwarded to
                :func:`IMDgroup.pymatgen.core.structure.structure_interpolate2`.
        """
        if beg is None:
            beg = self.structure
        if end is None:
            end = self.target_structure
        else:
            self.target_structure = end
        frac_tol = 0 if self.method == 'IDPP' else self.frac_tol
        assert isinstance(beg, Structure)
        assert isinstance(end, Structure)
        assert self.incar and "IMAGES" in self.incar

        str_images = structure_interpolate2(
            beg, end,
            nimages=self.incar["IMAGES"] + 1,
            frac_tol=frac_tol,
            **kwargs)

        if self.method != 'IDPP':
            for image in str_images:
                assert structure_is_valid2(image, self.frac_tol)
        self._fix_atoms_maybe(str_images)  # modify by side effect

        if self.method == 'IDPP':
            adaptor = AseAtomsAdaptor()
            images_ase = [adaptor.get_atoms(i) for i in str_images]
            # idpp_interpolate(images_ase, fmax=0.001)
            # mic=True is important as periodic boundary conditions are not
            # considered between images otherwise
            # Note: traj=None is handled OK despite type mismatch (no type definition in ASE)
            idpp_interpolate(images_ase, mic=True, traj=None)  # pyright: ignore[reportArgumentType] 
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
