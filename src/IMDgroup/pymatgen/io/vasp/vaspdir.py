"""This module implements abstraction over Vasp input/output directory.
"""
import collections
import typing
import hashlib
import os
import warnings
import logging
from pathlib import Path
from monty.json import MSONable
from diskcache import Cache
from alive_progress import alive_it
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar as pmgPoscar
from pymatgen.io.vasp.inputs import Kpoints as pmgKpoints
from pymatgen.io.vasp.inputs import Potcar as pmgPotcar
from pymatgen.io.vasp.outputs import Oszicar as pmgOszicar
from pymatgen.io.vasp.outputs import Chgcar as pmgChgcar
from pymatgen.io.vasp.outputs import Waveder as pmgWaveder
from pymatgen.io.vasp.outputs import Locpot as pmgLocpot
from pymatgen.io.vasp.outputs import Procar as pmgProcar
from pymatgen.io.vasp.outputs import Elfcar as pmgElfcar
from pymatgen.io.vasp.outputs import WSWQ as pmgWSWQ
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.io.vasp.outputs import Vasprun, Outcar

logger = logging.getLogger(__name__)


# Rewriting the original VaspDir/PMGDir class to add caching, dumping,
# and other goodies.
class IMDGVaspDir(collections.abc.Mapping, MSONable):
    """
    User-friendly class to access all files in a VASP calculation directory as pymatgen objects in a dict.
    Note that the files are lazily parsed to minimize initialization costs since not all files will be needed by all
    users.

    Example:

    ```
    d = VaspDir(".")
    print(d["INCAR"]["NELM"])
    print(d["vasprun.xml"].parameters)
    ```

    The information may be updated if files on disk change:
    d.refresh()

    The class also exposes useful properties that may require parsing
    multiple files in the directory: final_energy, initial_structure,
    structure, total_magnetization, converged_ionic,
    converged_electronic, converged.

    Some new properties distinct from what is usually available from pymatgen:
    - nebp (whether directory is a NEB VASP input)
    - neb_dirs (list of NEB dirs, if any)
    """

    FILE_MAPPINGS: typing.ClassVar = {
        "INCAR": Incar,
        "POSCAR": pmgPoscar,
        "CONTCAR": pmgPoscar,
        "KPOINTS": pmgKpoints,
        "POTCAR": pmgPotcar,
        "vasprun": Vasprun,
        "OUTCAR": Outcar,
        # FIXME: Need to modify parent clases to make them dumpable
        "OSZICAR": pmgOszicar,
        "CHGCAR": pmgChgcar,
        # "WAVECAR": pmgWavecar,
        "WAVEDER": pmgWaveder,
        "LOCPOT": pmgLocpot,
        # "XDATCAR": pmgXdatcar,
        # "EIGENVAL": pmgEigenval,
        "PROCAR": pmgProcar,
        "ELFCAR": pmgElfcar,
        # "DYNMAT": pmgDynmat,
        "WSWQ": pmgWSWQ,
    }

    def reset(self):
        """
        Reset all loaded files and recheck the directory for files.
        """
        path = Path(self.path)
        self.files = [str(f) for f in path.iterdir() if f.is_file()]
        self._neb_vaspdirs = None
        self._parsed_files = {}
        self.__cache = None

    def _dump_to_cache(self):
        """Dump parsed data to cache.
        """
        self._cache.set(
            self.path,
            {
                'hash': self._get_hash(),
                'parsed_files': self._parsed_files
            }
        )

    def refresh(self):
        """Make sure that cache is up to date with disk.
        """
        cache_val = self._cache.get(self.path)
        assert cache_val is None or isinstance(cache_val, dict)
        if cache_val and cache_val['hash'] == self._get_hash():
            self._parsed_files = cache_val['parsed_files']
        else:
            self._dump_to_cache()

    def __init__(self, dirname: str | Path):
        """
        Args:
            dirname: The directory containing the VASP calculation.
        """
        self.path = str(Path(dirname).resolve())
        self.__cache = None  # Pacify linter.  Same is done in reset().
        self.reset()
        self.refresh()

    @property
    def _cache(self) -> Cache:
        """Disk cache associated with current Vasp directory.
        """
        if self.__cache is not None:
            return self.__cache
        self.__cache = Cache(self._get_cache_dir())
        return self.__cache

    @staticmethod
    def read_vaspdirs(
            rootpath: Path | str, path_filter=None
    ) -> dict[str, 'IMDGVaspDir']:
        """Read vasp directories recursively from ROOTPATH.
        Args:
          rootdir (str | Path): Root directory.
          path_filter(function(PathLike): Function returning True for
          paths that should be read.
        Return a dict {'path': IMDGVaspDir object}
        """
        rootpath = Path(rootpath)
        valid_paths = {}
        for parent, _, files in alive_it(
                rootpath.walk(),
                title=f"Scanning {rootpath} for VASP directories"):
            for vaspfile in ['OUTCAR', 'vasprun.xml', 'POSCAR', 'OSZICAR']:
                if vaspfile in files and (
                        path_filter is None or
                        path_filter(parent)):
                    valid_paths[str(parent)] = IMDGVaspDir(parent)
                    break
        return valid_paths

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(self.files)

    def __getitem__(self, item):
        if item in self._parsed_files:
            return self._parsed_files[item]
        path = Path(self.path)
        for k, cls_ in self.FILE_MAPPINGS.items():
            if k in item and (path / item).exists():
                try:
                    try:
                        self._parsed_files[item] = cls_.from_file(path / item)
                    except AttributeError:
                        self._parsed_files[item] = cls_(path / item)
                except Exception:
                    # Parsing failed
                    logger.debug("Failed to read %s", (path / item))
                    self._parsed_files[item] = None
                self._dump_to_cache()
                return self._parsed_files[item]
        if (path / item).exists():
            raise RuntimeError(
                f"Unable to parse {item}. "
                f"Supported files are {list(self.FILE_MAPPINGS.keys())}.")

        return None
        # raise ValueError(
        #     f"{item} not found in {self.path}. "
        #     f"List of files are {self.files}.")

    @staticmethod
    def _get_file_hash(filename: Path | str) -> str:
        """Get hash of FILENAME.
        The hash is simply modification time."""
        f = Path(filename)
        return str(f.stat().st_mtime) + f.name

    def _get_hash(self) -> str:
        """Get hash of all files in path."""
        hashes = map(self._get_file_hash, self.files)
        combined_hash =\
            hashlib.md5("".join(hashes).encode('utf-8')).hexdigest()
        return str(combined_hash)

    @staticmethod
    def _get_cache_dir() -> Path:
        cache_dir = os.getenv("XDG_CACHE_HOME")\
            or os.path.expanduser("~/.cache")
        return Path(cache_dir) / "imdgVASPDIRcache"

    @property
    def final_energy(self) -> float:
        """Final energy computed in current Vasp outputs.
        """
        import numpy as np
        if run := self['vasprun.xml']:
            return run.final_energy
        if outcar := self['OUTCAR']:
            final_energy = outcar.final_energy
            if not isinstance(final_energy, float):
                warnings.warn(
                    f"Problems reading final energy (={final_energy}) from {self.path}/OUTCAR."
                )
                final_energy = np.nan
            return final_energy
        return np.nan

    @property
    def final_energy_reliable(self) -> str | float:
        """Like final_energy, but check if energy can be trusted.
        Return self.final_energy when energy value appears reliable.
        Return "unrealiable" if energy is unreliable due to the nature
        of VASP calculation (volume relax does not provide reliable energy)
        Return "unconverged" when system is not converged.
        """
        if not self.converged:
            return "unconvegred"
        incar = self['INCAR']
        assert incar is not None
        if incar.get('IBRION') in Incar.IBRION_IONIC_RELAX_values and\
           incar.get('ISIF') != Incar.ISIF_FIX_SHAPE_VOL:
            return "unreliable"
        return self.final_energy

    @property
    def initial_structure(self) -> Structure:
        """Get initial structure.
        """
        if poscar := self['POSCAR']:
            return poscar.structure
        raise FileNotFoundError("No POSCAR available")

    @property
    def structure(self) -> Structure:
        """Get the last known structure.
        """
        if contcar := self['CONTCAR']:
            return contcar.structure
        if run := self['vasprun.xml']:
            return run.final_structure
        raise FileNotFoundError("No vasprun.xml/CONTCAR available")

    @property
    def total_magnetization(self) -> float | None:
        """Get total magnetization.
        """
        if oszicar := self['OSZICAR']:
            return oszicar.ionic_steps[-1].get('mag', None)\
                if len(oszicar.ionic_steps) > 0 else None
        return None

    @property
    def converged_ionic(self) -> bool:
        """Return whether run converged ionically.
        """
        if run := self['vasprun.xml']:
            return run.converged_ionic
        if outcar := self['OUTCAR']:
            if 'converged_ionic' not in outcar.data:
                outcar.read_pattern(
                    {'converged_ionic':
                     r'(reached required accuracy - stopping structural energy minimisation|writing wavefunctions)'},
                    reverse=True,
                    terminate_on_match=True
                )
            return outcar.data['converged_ionic']
        return False

    @property
    def converged_electronic(self) -> bool:
        """Return whether run converged electronically.
        """
        if run := self['vasprun.xml']:
            return run.converged_electronic
        if outcar := self['OUTCAR']:
            if 'converged_electronic' not in outcar.data:
                outcar.read_pattern(
                    {'converged_electronic':
                     r'aborting loop (EDIFF was not reached \(unconverged\)|because EDIFF is reached)'},
                    reverse=True,
                    terminate_on_match=True
                )
            converged_electronic = outcar.data['converged_electronic']
            if converged_electronic:
                return converged_electronic[0][0] == 'because EDIFF is reached'
            return False
        return False

    @property
    def converged(self) -> bool:
        """Return whether vasp run converged.
        """
        if self.nebp:
            neb_dirs = self.neb_dirs(include_ends=False)
            assert neb_dirs is not None
            for image_dir in neb_dirs:
                if not image_dir.converged:
                    return False
            return True
        return self.converged_electronic and self.converged_ionic

    @property
    def nebp(self) -> bool:
        """Return True when it is a NEB-like run.
        """
        if incar := self['INCAR']:
            return 'IMAGES' in incar
        return False

    def neb_dirs(self, include_ends=True) -> list['IMDGVaspDir'] | None:
        """Return a list of NEB dirs.
        When optional argument INCLUDE_ENDS is False, do not include the
        first and the last image.
        """
        if self.nebp:
            if self._neb_vaspdirs is None:
                dir_names = self['INCAR'].image_dir_names()
                self._neb_vaspdirs = [
                    IMDGVaspDir(Path(self.path) / d) for d in dir_names
                ]
            return self._neb_vaspdirs if include_ends\
                else self._neb_vaspdirs[1:-1]
        return None
