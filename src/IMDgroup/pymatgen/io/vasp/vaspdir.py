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


"""This module implements abstraction over Vasp input/output directory.
"""
import collections
import typing
import hashlib
import os
import warnings
import logging
import itertools
import tempfile
import pickle
from pathlib import Path
from monty.json import MSONable
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
        self._prev_vaspdirs = None
        self._parsed_files = None

    def __init__(self, dirname: str | Path) -> None:
        """
        Args:
            dirname: The directory containing the VASP calculation.
        """
        self.path = str(Path(dirname).absolute())
        # This was slower: self.path = str(Path(dirname).resolve())
        self._parsed_files = None  # Pacify linter.  Same is done in reset().
        self._prev_vaspdirs = None  # Pacify linter.  Same is done in reset().
        self._neb_vaspdirs = None  # Pacify linter.  Same is done in reset().
        self.reset()

    @staticmethod
    def _get_cache_dir() -> Path:
        cache_dir = os.getenv("XDG_CACHE_HOME")\
            or os.path.expanduser("~/.cache")
        return Path(cache_dir) / "imdgVASPDIRcache"

    def _get_cache_path(self) -> Path:
        """Get path for this directory's cache file"""
        cache_dir = self._get_cache_dir()
        # Create unique filename using path hash
        path_hash = hashlib.md5(self.path.encode('utf-8')).hexdigest()
        # I want the path to be in the usual form of cache_dir/XX/YYYYYYY.pkl
        cache_subdir = path_hash[:2]
        cache_dir = cache_dir / cache_subdir
        return cache_dir / f"{path_hash[2:]}.pkl"

    def _load_from_cache(self) -> dict | None:
        """Load cached data from file if valid"""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
        return None

    def _save_to_cache(self, data: dict) -> bool:
        """Atomically save data to cache file"""
        cache_path = self._get_cache_path()
        cache_dir = cache_path.parent
        tmp_path = None
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Write to temp file first
            with tempfile.NamedTemporaryFile(
                mode='wb',
                dir=cache_dir,
                delete=False
            ) as tmp:
                # Use highest protocol for efficiency
                pickle.dump(data, tmp, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_path = tmp.name
            # Atomic rename
            os.rename(tmp_path, cache_path)
            return True
        except Exception as e:
            logger.warning("Cache save failed: %s", e)
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()
            return False

    def _dump_to_cache(self) -> bool:
        """Dump parsed data to cache file atomically"""
        return self._save_to_cache({
            'hash': self._get_hash(),
            'parsed_files': self._parsed_files
        })

    def refresh(self):
        """Refresh from cache if valid"""
        if cache_data := self._load_from_cache():
            if cache_data.get('hash') == self._get_hash():
                self._parsed_files = cache_data['parsed_files']
                logger.debug("Loaded cache for %s", os.path.relpath(self.path))
                return
        self._parsed_files = {}
        logger.debug("No valid cache for %s", os.path.relpath(self.path))

    @staticmethod
    def read_vaspdirs(
            rootpath: Path | str | list[Path | str], path_filter=None
    ) -> dict[str, 'IMDGVaspDir']:
        """Read vasp directories recursively from ROOTPATH.
        Args:
          rootdir (str | Path | list[str|Path]): Root directory.
          path_filter(function(PathLike): Function returning True for
          paths that should be read.
        Return a dict {'path': IMDGVaspDir object}
        """
        if isinstance(rootpath, list):
            rootpath = [Path(p) for p in rootpath]
        else:
            rootpath = [Path(rootpath)]
        valid_paths = {}
        paths_str = str(map(str, rootpath))
        max_len = 40
        if len(paths_str) > max_len:
            paths_str = paths_str[:max_len] + "…"
        for parent, _, files in alive_it(
                itertools.chain.from_iterable([p.walk() for p in rootpath]),
                title=f"Scanning {list(map(str, rootpath))} for VASP directories"):
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
        if self._parsed_files is None:
            self.refresh()
        if item in self._parsed_files:
            return self._parsed_files[item]
        path = Path(self.path)
        for k, cls_ in self.FILE_MAPPINGS.items():
            if k in item and (path / item).exists():
                try:
                    # Standard parsing for all files
                    try:
                        obj = cls_.from_file(path / item)
                    except AttributeError:
                        obj = cls_(path / item)
                    self._parsed_files[item] = obj
                    self._dump_to_cache()
                    return obj
                except Exception as e:
                    logger.debug("Failed to read %s: %s", path/item, e)
                    self._parsed_files[item] = None
                    self._dump_to_cache()
                    return None
        if (path / item).exists():
            raise RuntimeError(
                f"Unable to parse {item}. "
                f"Supported files are {list(self.FILE_MAPPINGS.keys())}.")
        return None

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
                    f"Problems reading final energy (={final_energy}) from"
                    f" {os.path.relpath(self.path)}/OUTCAR."
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
        If prev_vaspdirs exist, use initial structure from
        the oldest of those subdirectories.
        """
        if prevs := self.prev_dirs():
            return prevs[0].initial_structure
        if poscar := self['POSCAR']:
            return poscar.structure
        if run := self['vasprun.xml']:
            return run.initial_structure
        raise FileNotFoundError("No vasprun.xml/POSCAR available")

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

    def _mtime_1(self) -> float:
        """Return modification time of self.
        Ignore previous runs.  Use NEB dirs if any.
        """
        if self.nebp:
            neb_dirs = self.neb_dirs()
            assert neb_dirs is not None
            times = [d.mtime() for d in neb_dirs]
            return max(times)
        path = Path(self.path)
        outcar = path / "OUTCAR"
        if outcar.is_file():
            return outcar.stat().st_mtime
        # ATAT moves away OUTCAR and other files but not vasprun.xml
        vasprun = path / "vasprun.xml"
        if vasprun.is_file():
            return vasprun.stat().st_mtime
        incar = path / "INCAR"
        if incar.is_file():
            return incar.stat().st_mtime
        return path.stat().st_mtime

    def mtime(self) -> float:
        """Return modification time of VASP run.
        The modification time is the modification time of the
        directory itself or the latest modification time of
        prev_dirs and/or neb_dirs.
        """
        mtimes = [self._mtime_1()]
        prev_dirs = self.prev_dirs()
        if prev_dirs:
            for prev in prev_dirs:
                mtimes.append(prev.mtime())
        return max(mtimes)

    def prev_dirs(self) -> list['IMDGVaspDir'] | None:
        """Return a list of previous VASP runs, ordered by directory name.
        Previous VASP runs are assumed to stay in gorun_* folders.
        """
        if self._prev_vaspdirs is None:
            path = Path(self.path)
            self._prev_vaspdirs = sorted(
                [IMDGVaspDir(d) for d in path.glob('gorun_*/') if d.is_dir()],
                key=lambda d: d.path)
            if len(self._prev_vaspdirs) > 0:
                logger.info(
                    "%s contains previous Vasp runs",
                    os.path.relpath(self.path)
                )
        return self._prev_vaspdirs
