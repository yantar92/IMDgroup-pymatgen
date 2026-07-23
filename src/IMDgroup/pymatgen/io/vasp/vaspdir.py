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


"""This module implements abstraction over Vasp input/output directory."""
import collections
import typing
import hashlib
import os
import re
import warnings
import logging
import itertools
import pickle
import signal
import sys
import threading
import atexit
from pathlib import Path
import lmdb
import numpy as np
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
from IMDgroup.pymatgen.core.structure import structure_distance

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Raised when a VASP directory read operation times out."""


def timeout_handler(signum, frame):
    """Signal handler that raises TimeoutException.

    Args:
        signum: Signal number.
        frame: Current stack frame.

    Raises:
        TimeoutException: Always raised.
    """
    raise TimeoutException


HAS_SIGALRM = sys.platform != 'win32'
if HAS_SIGALRM:
    signal.signal(signal.SIGALRM, timeout_handler)


# Rewriting the original VaspDir/PMGDir class to add caching, dumping,
# and other goodies.
class IMDGVaspDir(collections.abc.Mapping, MSONable):
    """Dictionary-like access to all files in a VASP calculation directory.

    Files are lazily parsed to minimise initialisation cost.  Example::

        d = IMDGVaspDir(".")
        print(d["INCAR"]["NELM"])
        print(d["vasprun.xml"].parameters)

    Call ``refresh()`` to re-read the directory after files change.

    Cached parsing results are stored in LMDB to speed up repeated
    access in HPC workflows.

    Properties that require parsing multiple files:

    - ``final_energy``, ``final_energy_reliable``
    - ``initial_structure``, ``structure``
    - ``total_magnetization``
    - ``converged``, ``converged_ionic``, ``converged_electronic``,
      ``converged_sequence``, ``converged_manual``
    - ``nebp`` (whether directory is a NEB calculation)
    - ``neb_dirs`` (list of NEB subdirectories, if any)
    - ``mtime`` (latest modification time across all files)
    - ``prev_dirs`` (chain of previous gorun_* runs)
    """

    TIMEOUT = 60 * 2  # 2 minutes
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
        """Reset all loaded files and re-scan the directory.

        Clears cached parsed files, previous-run references, and NEB
        subdirectory references.
        """
        path = Path(self.path)
        self.files = sorted(f for f in path.iterdir() if f.is_file())
        self._neb_vaspdirs = None
        self._prev_vaspdirs = None
        self._parsed_files = None

    _pending_writes: typing.ClassVar[dict[str, dict]] = {}
    _pending_lock: typing.ClassVar[threading.Lock] = threading.Lock()
    _flush_registered: typing.ClassVar[bool] = False
    _lmdb_env: typing.ClassVar[typing.Any] = None
    _lmdb_db: typing.ClassVar[typing.Any] = None
    # -- LMDB metadata ("meta") for eviction ----------------------------------
    #
    # LMDB uses a fixed-size mmap (1TB sparse).  When the map fills,
    # we need to free space by evicting the oldest cache entries.
    #
    # To know which entries are oldest and how much space they occupy,
    # every cached value is paired with a *metadata* entry.  The metadata
    # key is the original key prefixed with _META_PREFIX and the value is
    # a pickled dict:
    #
    #     {"size":  <serialized bytes of the cached data>,
    #      "ctime": <monotonic timestamp when the entry was first written>}
    #
    # The metadata is much faster to load, so we can scan for old entries
    # without having to load them in full.
    #
    # _lmdb_get_meta_all scans both explicit __meta__ entries and
    # bare data entries (legacy entries that predate metadata tracking
    # get ctime=0, so they are evicted first).
    #
    # _lmdb_evict uses ctime to sort oldest-first, deleting both the
    # data key and its __meta__ companion until enough bytes are freed.
    _META_PREFIX: typing.ClassVar[bytes] = b"__meta__"

    @classmethod
    def _init_lmdb(cls) -> None:
        if cls._lmdb_env is not None:
            return
        cache_dir = cls._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / "cache.lmdb"
        # 1TB map size; LMDB uses sparse mmap so only written pages
        # consume actual disk.  Eviction logic in _lmdb_set_many
        # reclaims pages when the map fills.
        try:
            cls._lmdb_env = lmdb.open(
                str(db_path),
                map_size=2**40,  # 1TB
                max_dbs=1,
                lock=True,
                subdir=False,
            )
            cls._lmdb_db = cls._lmdb_env.open_db(b"vaspdir")
        except Exception as e:
            logger.warning("LMDB initialization failed; caching disabled: %s", e)
            cls._lmdb_env = None
            cls._lmdb_db = None

    @classmethod
    def _lmdb_get(cls, key: str) -> dict | None:
        cls._init_lmdb()
        if cls._lmdb_env is None:
            return None
        try:
            with cls._lmdb_env.begin(db=cls._lmdb_db, write=False) as txn:
                val = txn.get(key.encode())
                if val is None:
                    return None
                return pickle.loads(val)
        except Exception as e:
            logger.warning("LMDB get failed: %s", e)
            return None

    @classmethod
    def _lmdb_set(cls, key: str, data: dict) -> bool:
        """Write a single entry with metadata tracking.

        On MDB_MAP_FULL, evicts oldest entries and retries once.
        """
        return cls._lmdb_set_many({key: data})

    @classmethod
    def _lmdb_get_meta_all(cls, txn) -> dict[str, dict]:
        """Return all metadata as {key: meta_dict}.

        Must be called inside an active LMDB transaction.

        Scans the entire database in two passes:

        1. Explicit __meta__ entries (new-style): deserialised from
           pickle.  These carry accurate ``size`` and ``ctime``.

        2. Bare data entries (legacy, no __meta__ companion): their
           ``size`` is estimated as the raw value length on disk and
           ``ctime`` defaults to 0.  Setting ctime=0 makes them the
           oldest entries, so they are evicted first -- this is
           intentional: legacy entries are rebuilt with proper metadata
           on next access.

        The metadata dict enables _lmdb_evict to sort entries by age
        and track freed bytes during eviction.
        """
        result = {}
        cursor = txn.cursor()
        for raw_key, raw_val in cursor:
            if raw_key.startswith(cls._META_PREFIX):
                # New-style: explicit metadata for a known cache key.
                try:
                    key = raw_key[len(cls._META_PREFIX):].decode()
                    result[key] = pickle.loads(raw_val)
                except Exception:
                    pass
            else:
                # Legacy: bare data entry without companion metadata.
                try:
                    key = raw_key.decode()
                except UnicodeDecodeError:
                    continue
                if key not in result:
                    # Synthetic metadata: estimate size from raw bytes,
                    # assign ctime=0 so legacy entries are evicted before
                    # any new-style entry.
                    result[key] = {'size': len(raw_val), 'ctime': 0}
        return result

    @classmethod
    def _lmdb_evict(cls, txn, needed_bytes: int) -> int:
        """Evict oldest entries to free at least *needed_bytes*.

        Must be called inside an active write transaction.
        Returns the total bytes freed.
        """
        meta_all = cls._lmdb_get_meta_all(txn)
        if not meta_all:
            logger.warning("LMDB eviction requested but no metadata found")
            return 0

        # Evict oldest-first (by creation time)
        sorted_keys = sorted(
            meta_all.keys(),
            key=lambda k: meta_all[k].get('ctime', 0),
        )

        freed = 0
        evicted = 0
        for key in sorted_keys:
            if freed >= needed_bytes:
                break
            size = meta_all[key].get('size', 0)
            txn.delete(key.encode())
            txn.delete(cls._META_PREFIX + key.encode())
            freed += size
            evicted += 1

        if evicted:
            logger.info(
                "LMDB evicted %d entries, freed %.2f MB",
                evicted, freed / 1024**2,
            )
        return freed

    @classmethod
    def _lmdb_set_many(cls, items: dict[str, dict]) -> bool:
        """Write entries with metadata, evicting on MDB_MAP_FULL.

        Pre-serializes values to avoid double pickling.  On map-full
        errors, evicts the oldest entries (by creation time) and
        retries once before giving up.
        """
        cls._init_lmdb()
        if cls._lmdb_env is None:
            return False
        import time
        now = time.time()

        # Pre-serialize so we can measure sizes and avoid double pickling.
        # Each entry gets a companion __meta__ record with its size and
        # creation time.  The size is measured from the serialized bytes
        # so it reflects what LMDB actually stores, not the Python object.
        serialized: dict[str, bytes] = {}
        meta_serialized: dict[str, bytes] = {}
        for key, data in items.items():
            serialized[key] = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            meta = {'size': len(serialized[key]), 'ctime': now}
            meta_serialized[key] = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)

        new_bytes = sum(len(v) + len(meta_serialized[k]) for k, v in serialized.items())

        for attempt in range(2):
            try:
                with cls._lmdb_env.begin(db=cls._lmdb_db, write=True) as txn:
                    for key in items:
                        # Write the actual cached data under the original key.
                        txn.put(key.encode(), serialized[key])
                        # Write its companion metadata under __meta__<key>.
                        # This pair is what _lmdb_evict deletes together
                        # when freeing space.
                        txn.put(
                            cls._META_PREFIX + key.encode(),
                            meta_serialized[key],
                        )
                return True
            except Exception as e:
                if 'MDB_MAP_FULL' not in str(e):
                    logger.warning("LMDB set_many failed: %s", e)
                    return False
                if attempt == 0:
                    logger.warning(
                        "LMDB map full, evicting before retry "
                        "(need ~%.1f MB)", new_bytes / 1024**2,
                    )
                    try:
                        with cls._lmdb_env.begin(
                            db=cls._lmdb_db, write=True,
                        ) as txn:
                            # Free 2x what we need as headroom
                            cls._lmdb_evict(txn, new_bytes * 2)
                    except Exception as evict_e:
                        logger.error("LMDB eviction failed: %s", evict_e)
                        return False
                else:
                    logger.error(
                        "LMDB set_many failed after eviction: %s", e,
                    )
                    return False
        return False

    def __init__(self, dirname: str | Path) -> None:
        """Initialise from a directory path.

        Args:
            dirname: Path to the VASP calculation directory.
        """
        self.path = str(Path(dirname).absolute())
        # This was slower: self.path = str(Path(dirname).resolve())
        self._cache_key = hashlib.md5(self.path.encode('utf-8')).hexdigest()
        self._parsed_files = None  # Pacify linter.  Same is done in reset().
        self._prev_vaspdirs = None  # Pacify linter.  Same is done in reset().
        self._neb_vaspdirs = None  # Pacify linter.  Same is done in reset().
        self.reset()

    # Implementation note: We cannot use SQLite to store cache
    # because SQLite is not reliable on LUSTRE file system
    # in HPC.  LMDB is used instead.
    @staticmethod
    def _get_cache_dir() -> Path:
        cache_dir = os.getenv("XDG_CACHE_HOME")\
            or os.path.expanduser("~/.cache")
        return Path(cache_dir) / "imdgVASPDIRcache"

    def _load_from_cache(self) -> dict | None:
        """Load cached data from LMDB if valid.

        Returns:
            Cached data dict on success, ``None`` if no entry exists
            or the LMDB environment is unavailable.
        """
        return self._lmdb_get(self._cache_key)

    @classmethod
    def _add_pending_write(cls, key: str, data: dict) -> None:
        """Add a cache entry to the pending write buffer."""
        with cls._pending_lock:
            cls._pending_writes[key] = data
            if not cls._flush_registered:
                atexit.register(cls.flush_cache)
                cls._flush_registered = True

    @classmethod
    def flush_cache(cls) -> None:
        """Write all pending cache entries to LMDB."""
        with cls._pending_lock:
            if not cls._pending_writes:
                return
            writes = cls._pending_writes.copy()
            cls._pending_writes.clear()
        # Write outside lock to avoid holding lock during I/O
        cls._lmdb_set_many(writes)
        logger.debug("Flushed %d cache entries to LMDB", len(writes))

    def _dump_to_cache(self) -> bool:
        """Dump parsed data to cache."""
        data = {
            'hash': self._get_hash(),
            'parsed_files': self._parsed_files
        }
        self._add_pending_write(self._cache_key, data)
        return True

    def refresh(self):
        """Reload cached data from disk or re-parse if files changed."""
        key = self._cache_key
        current_hash = self._get_hash()
        # First check pending writes (in-process, not yet flushed to LMDB)
        with self._pending_lock:
            pending = self._pending_writes.get(key)
        if pending is not None:
            if pending.get('hash') == current_hash:
                self._parsed_files = pending['parsed_files']
                logger.debug(
                    "Loaded pending cache for %s",
                    os.path.relpath(self.path))
                return
            # Stale pending entry, remove it
            logger.debug(
                "Pending cache hash mismatch for %s (pending=%s, current=%s)",
                os.path.relpath(self.path),
                pending.get('hash'), current_hash)
            with self._pending_lock:
                self._pending_writes.pop(key, None)
        # Fall back to disk cache (LMDB)
        cache_data = self._load_from_cache()
        if cache_data is not None:
            cached_hash = cache_data.get('hash')
            if cached_hash == current_hash:
                self._parsed_files = cache_data['parsed_files']
                logger.debug(
                    "Loaded disk cache for %s",
                    os.path.relpath(self.path))
                return
            logger.debug(
                "Disk cache hash mismatch for %s (cached=%s, current=%s)",
                os.path.relpath(self.path),
                cached_hash, current_hash)
        else:
            logger.debug(
                "No disk cache entry for %s",
                os.path.relpath(self.path))
        self._parsed_files = {}
        logger.debug("No valid cache for %s", os.path.relpath(self.path))

    @staticmethod
    def read_vaspdirs(
            rootpath: Path | str | list[Path | str], path_filter=None
    ) -> dict[str, 'IMDGVaspDir']:
        """Recursively scan directories for VASP calculations.

        Args:
            rootpath: Root directory or list of directories to scan.
            path_filter: Optional callable that returns True for paths
                to include.

        Returns:
            dict[str, IMDGVaspDir]: Mapping of ``{path: IMDGVaspDir}``.
        """
        if isinstance(rootpath, list):
            rootpath = [Path(p) for p in rootpath]
        else:
            rootpath = [Path(rootpath)]
        valid_paths = {}
        for parent, _, files in alive_it(
                itertools.chain.from_iterable([p.walk() for p in rootpath]),
                title=f"Scanning {list(map(str, rootpath))} for VASP directories"):
            for vaspfile in ['OUTCAR', 'vasprun.xml', 'POSCAR', 'OSZICAR']:
                if vaspfile in files and (
                        path_filter is None or path_filter(parent)):
                    valid_paths[str(parent)] = IMDGVaspDir(parent)
                    break
        IMDGVaspDir.flush_cache()
        return valid_paths

    def __contains__(self, item):
        return item in self.files

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return iter(f.name for f in self.files)

    def __getitem__(self, item):
        if self._parsed_files is None:
            self.refresh()
        assert self._parsed_files is not None
        if item in self._parsed_files:
            return self._parsed_files[item]
        path = Path(self.path)
        for k, cls_ in self.FILE_MAPPINGS.items():
            if k in item and (path / item).exists():
                # Avoid being stuck in IO
                # It can happen with Outcar
                # See https://github.com/materialsproject/pymatgen/issues/4550
                # or just because of IO issues on cluster
                # avoid being stuck and simply signal failure then.
                if HAS_SIGALRM:
                    signal.alarm(self.TIMEOUT)
                try:
                    # Standard parsing for all files
                    try:
                        obj = cls_.from_file(path / item)
                    except AttributeError:
                        obj = cls_(path / item)
                    if HAS_SIGALRM:
                        signal.alarm(0)
                    self._parsed_files[item] = obj
                    self._dump_to_cache()
                    return obj
                # except TimeoutException:
                except Exception as e:
                    logger.debug("Failed to read %s: %s", path / item, e)
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
        return str(f.stat().st_mtime if f.exists() else "") + f.name

    def _get_hash(self) -> str:
        """Get hash of all files in path.

        Hashes are deterministic because ``self.files`` is sorted
        in :meth:`reset`.
        """
        hashes = map(self._get_file_hash, self.files)
        combined_hash =\
            hashlib.md5("".join(hashes).encode('utf-8')).hexdigest()
        return str(combined_hash)

    @property
    def final_energy(self) -> float:
        """Final energy computed in current Vasp outputs.
        """
        def warn_unconverged():
            warnings.warn(
                f"Reading final energy from unconverged run: {os.path.relpath(self.path)}"
            )
        if run := self['vasprun.xml']:
            if not self.converged:
                warn_unconverged()
            return run.final_energy
        if 'vasprun.xml' in self:
            # incomplete  vasprun.xml
            return np.nan
        if outcar := self['OUTCAR']:
            final_energy = outcar.final_energy
            if not isinstance(final_energy, float):
                warnings.warn(
                    f"Problems reading final energy (={final_energy}) from"
                    f" {os.path.relpath(self.path)}/OUTCAR."
                )
                final_energy = np.nan
            elif not self.converged:
                warn_unconverged()
            return final_energy
        return np.nan

    @property
    def final_energy_reliable(self) -> str | float:
        """Like :attr:`final_energy`, but with a reliability check.

        Returns:
            float: The final energy when judged reliable.
            str: ``"unreliable"`` when energy may be inaccurate (e.g.
            volume relaxation).
            str: ``"unconverged"`` when the run has not converged.
        """
        if not self.converged:
            return "unconvegred"
        incar = self['INCAR']
        assert incar is not None
        run = self['vasprun.xml']
        assert run is not None
        n_steps = len(run.ionic_steps)
        if incar.get('IBRION') in Incar.IBRION_IONIC_RELAX_values and\
           incar.get('ISIF') not in [
               Incar.ISIF_FIX_SHAPE_VOL, Incar.ISIF_FIX_SHAPE_VOL_FAST,
               Incar.ISIF_FIX_SHAPE_VOL_TRACE] and incar.get('NSW', 0) > 0 and n_steps > 1:  # NSW = 0 is SCF
            return "unreliable"
        return self.final_energy

    @property
    def initial_structure(self) -> Structure:
        """Initial structure of the calculation.

        Follows the chain of ``prev_dirs`` to find the earliest
        initial structure if previous runs exist.
        """
        if prevs := self.prev_dirs():
            return prevs[0].initial_structure
        if poscar := self['POSCAR']:
            return poscar.structure
        if run := self['vasprun.xml']:
            return run.initial_structure
        raise FileNotFoundError(f"{self.path}: No vasprun.xml/POSCAR available")

    @property
    def structure(self) -> Structure:
        """Last known structure (CONTCAR if present, else final from vasprun)."""
        if contcar := self['CONTCAR']:
            return contcar.structure
        if run := self['vasprun.xml']:
            return run.final_structure
        raise FileNotFoundError("No vasprun.xml/CONTCAR available")

    @property
    def total_magnetization(self) -> float | None:
        """Total magnetization from OSZICAR, or None if unavailable."""
        if oszicar := self['OSZICAR']:
            return oszicar.ionic_steps[-1].get('mag', None)\
                if len(oszicar.ionic_steps) > 0 else None
        return None

    def check_displacements(self) -> bool:
        """Check whether atomic displacements are below a safe threshold.

        Warns and returns False when the maximum displacement exceeds
        twice the average bond length.

        Returns:
            bool: True if displacements are acceptable.
        """
        max_displacement = 0
        for i, site in enumerate(self.initial_structure):
            displacement = site.distance(self.structure[i])
            max_displacement = max(max_displacement, displacement)
        vol = self.structure.volume
        avg_bond_length = (vol / len(self.structure))**(1 / 3)
        if max_displacement > 2.0 * avg_bond_length:
            warnings.warn(
                f"{os.path.relpath(self.path)}: "
                f"Large atomic displacement {max_displacement}")
            return False
        return True

    def check_framework_symmetry(
            self, framework_elements=None,
            symprec=0.1, max_rms_threshold=0.5) -> bool:
        """Check whether the framework symmetry is preserved.

        Reduces false positives from mobile atoms breaking symmetry.

        Args:
            framework_elements: List of element symbols for the
                framework.  Defaults to the most common element.
            symprec: Symmetry tolerance for space group detection.
            max_rms_threshold: Maximum RMS displacement threshold in
                Angstrom.

        Returns:
            bool: True if framework symmetry is preserved.
        """
        if framework_elements is None:
            from collections import Counter
            elements = Counter([str(site.specie) for site in self.initial_structure])
            framework_elements = [elements.most_common(1)[0][0]]

        # Create framework-only structures
        def filter_framework(structure):
            indices = [i for i, site in enumerate(structure)
                       if str(site.specie) in framework_elements]
            return structure.copy().remove_sites(
                [i for i in range(len(structure)) if i not in indices]
            )

        init_framework = filter_framework(self.initial_structure)
        final_framework = filter_framework(self.structure)

        # Check if framework symmetry changed
        init_sg = init_framework.get_space_group_info(symprec=symprec)
        final_sg = final_framework.get_space_group_info(symprec=symprec)

        if init_sg[0] != final_sg[0]:
            # Calculate RMS displacement of framework atoms
            try:
                rms = structure_distance(
                    init_framework, final_framework,
                    norm=True, match_first=False)
            except Exception:
                rms = float('inf')
            if rms > max_rms_threshold or np.isclose(rms, 0):
                warnings.warn(
                    f"{os.path.relpath(self.path)}: "
                    f"Framework symmetry changed ({init_sg[0]} to {final_sg[0]}) "
                    f"with displacement (RMS={rms:.3f}Å)")
                return False
        return True

    @property
    def converged_ionic(self) -> bool:
        """Whether ionic convergence was reached.

        Also checks framework symmetry and displacements.
        """
        converged_ionic = False
        if run := self['vasprun.xml']:
            converged_ionic = run.converged_ionic
        elif 'vasprun.xml' in self:
            # vasprun.xml present but cannot be parsed
            return False
        elif outcar := self['OUTCAR']:
            if 'converged_ionic' not in outcar.data:
                outcar.read_pattern(
                    {'converged_ionic':
                     r'(reached required accuracy - stopping structural energy minimisation|writing wavefunctions)'},
                    reverse=True,
                    terminate_on_match=True
                )
            converged_ionic = outcar.data['converged_ionic']
        if converged_ionic:
            self.check_framework_symmetry()
            self.check_displacements()
        return converged_ionic

    @property
    def converged_electronic(self) -> bool:
        """Whether electronic convergence was reached."""
        if run := self['vasprun.xml']:
            return run.converged_electronic
        if 'vasprun.xml' in self:
            # vasprun.xml present but cannot be parsed
            return False
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
    def converged_sequence(self) -> bool:
        """Whether the multi-step convergence sequence is complete.

        Returns False when ``INCAR.[0-9]+`` files remain (signalling
        that further convergence steps are pending).
        """
        return not any(re.match(r'INCAR\.[0-9]+', file) for file in self)

    @property
    def converged_manual(self) -> bool:
        """Whether the directory is explicitly marked as converged.

        Returns False when an ``UNCONVERGED`` file is present.
        """
        return not (Path(self.path) / "UNCONVERGED").is_file()

    @property
    def converged(self) -> bool:
        """Overall convergence status.

        Returns True only when the run is electronically and ionically
        converged, the convergence sequence is complete, and no
        ``UNCONVERGED`` marker file is present.  For NEB runs, all
        images must be converged.
        """
        if not self.converged_manual:
            return False
        if self.nebp:
            neb_dirs = self.neb_dirs(include_ends=False)
            assert neb_dirs is not None
            for image_dir in neb_dirs:
                if not image_dir.converged:
                    return False
            return True
        return self.converged_electronic and self.converged_ionic\
            and self.converged_sequence

    @property
    def nebp(self) -> bool:
        """Whether this directory contains a NEB-like calculation.

        Detected by the presence of ``IMAGES`` in the INCAR.
        """
        if incar := self['INCAR']:
            return 'IMAGES' in incar
        return False

    def neb_dirs(self, include_ends=True) -> list['IMDGVaspDir'] | None:
        """List of NEB image subdirectories.

        Args:
            include_ends: When False, exclude the first and last images.

        Returns:
            list[IMDGVaspDir] | None: NEB subdirectories, or None if
            this is not a NEB run.
        """
        if self.nebp:
            if self._neb_vaspdirs is None:
                incar = self['INCAR']
                assert incar is not None
                dir_names = incar.image_dir_names()
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
        """Latest modification time across all relevant files.

        Considers NEB subdirectories and previous-run directories.
        """
        mtimes = [self._mtime_1()]
        prev_dirs = self.prev_dirs()
        if prev_dirs:
            for prev in prev_dirs:
                mtimes.append(prev.mtime())
        return max(mtimes)

    def prev_dirs(self) -> list['IMDGVaspDir'] | None:
        """List of previous VASP runs in the chain.

        Previous runs are assumed to reside in ``gorun_*``
        subdirectories containing a ``POSCAR``.

        Returns:
            list[IMDGVaspDir] | None: Sorted list of previous-run
            directories, or None.
        """
        if self._prev_vaspdirs is None:
            path = Path(self.path)
            self._prev_vaspdirs = sorted(
                [IMDGVaspDir(d) for d in path.glob('gorun_*/') if d.is_dir() and (d / 'POSCAR').is_file()],
                key=lambda d: d.path)
            if len(self._prev_vaspdirs) > 0:
                logger.info(
                    "%s contains previous Vasp runs",
                    os.path.relpath(self.path)
                )
        return self._prev_vaspdirs
