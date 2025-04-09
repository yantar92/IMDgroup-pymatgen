"""Analysis extensions specific to IMD group.
Based on pymatgen's pymatgen.cli.pmg_analyze
"""
import logging
import warnings
import os
import hashlib
from tabulate import tabulate

import numpy as np
from alive_progress import alive_bar
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.io.vasp.inputs import Poscar
from IMDgroup.pymatgen.io.vasp.outputs import Outcar
from IMDgroup.pymatgen.io.vasp.inputs import Incar

SAVE_FILE = "vasp_data_imdg.gz"
logger = logging.getLogger(__name__)


class IMDGBorgQueen (BorgQueen):
    """The Borg Queen controls the drones to assimilate data in an entire
    directory tree. Uses multiprocessing to speed up things considerably. It
    also contains convenience methods to save and load data between sessions.
    """

    class _DroneWithCache:

        def __init__(self, drone, cache, path_filter=None):
            self._drone = drone
            logger.debug("Setting up cache %s", cache.keys())
            self._cache = cache
            self.path_filter = path_filter
            self._ndirs = 0
            self._bar = None

        @staticmethod
        def _get_file_hash(filename):
            """Get hash of FILENAME.
            The hash is simply modification time."""
            return str(os.path.getmtime(filename)) + filename
            # with open(filename, 'rb', buffering=0) as f:
            #     return str(hashlib.file_digest(f, 'sha256').hexdigest())

        def _get_dir_hash(self, path):
            """Get hash of all files in path."""
            files = [os.path.join(path, f) for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))]
            hashes = map(self._get_file_hash, files)
            combined_hash =\
                hashlib.md5("".join(hashes).encode('utf-8')).hexdigest()
            return str(combined_hash)

        def assimilate(self, path):
            """Call _drone.assimulate with caching.
            """
            if self._bar is None and self._ndirs > 0:
                self._barctx = alive_bar(
                    self._ndirs, title='Reading VASP outputs')
                self._bar = self._barctx.__enter__()
            h = self._get_dir_hash(path)
            logger.debug("Assimilating %s [%s]", path, h)
            if self._cache.get(h):
                logger.info("Using cached data for %s", path)
                data = self._cache.get(h)
            else:
                logger.info("Reading data from %s", path)
                data = self._drone.assimilate(path)
            if self._bar is not None:
                self._bar()
            return {h: data}

        def get_valid_paths(self, path):
            """Call drone.get_valid_paths."""
            paths = self._drone.get_valid_paths(path)
            paths = [p for p in paths
                     if self.path_filter is None
                     or self.path_filter(p)]
            self._ndirs += len(paths)
            return paths

    def __init__(
            self, drone,
            rootpath=None,
            number_of_drones=1,
            dump_file=None,
            path_filter=None):
        """
        Args:
            drone (Drone): An implementation of
                pymatgen.apps.borg.hive.AbstractDrone to use for
                assimilation.
            rootpath (str): The root directory to start assimilation. Leave it
                as None if you want to do assimilation later, or is using the
                BorgQueen to load previously assimilated data.
            number_of_drones (int): Number of drones to parallelize over.
                Typical machines today have up to four processors. Note that you
                won't see a 100% improvement with two drones over one, but you
                will definitely see a significant speedup of at least 50% or so.
                If you are running this over a server with far more processors,
                the speedup will be even greater.
            dump_file (PathLike): File containing previously stored data.
            path_filter (function(PathLike)): Function filtering paths
            to be read.  It must return True when path should be included.
        """
        old_data: list = []
        if dump_file and os.path.isfile(dump_file):
            self.load_data(dump_file)
            old_data = self._data
        cache = {}
        for item in old_data:
            for h, val in item.items():
                cache[h] = val
        super().__init__(
            self._DroneWithCache(drone, cache, path_filter),
            rootpath,
            number_of_drones)

    def get_data(self):
        """Get an list of assimilated objects."""
        if getattr(self._drone, '_barctx', None) is not None:
            self._drone._barctx.__exit__(None, None, None)
        return [list(x.values())[0] for x in self._data]


class IMDGVaspToComputedEnrgyDrone(VaspToComputedEntryDrone):
    """Assimilate directories, as VaspToComputedEntryDrone, but
    also put parsed Outcar into result.data['outcar'].
    """
    def assimilate(self, path):
        """Assimilate Vasprun and Outcar from PATH.
        Return ComputedEntry object or None when PATH does not contain
        a valid Vasprun/Outcar.
        """
        computed_entry = super().assimilate(path)

        outcar_path = os.path.join(path, "OUTCAR")
        try:
            outcar = Outcar(outcar_path)
        except Exception as exc:
            if computed_entry is None:
                # Cannot read anything
                logger.debug("Failed to read vasprun.xml/OUTCAR in %s", path)
                return None
            logger.debug("error reading %s: %s", outcar_path, exc)
            outcar = None

        if computed_entry is None:
            assert outcar is not None
            # Vasprun not complete
            # Try to deduce parameters from OUTCAR + CONTCAR instead
            logger.debug("Trying to deduce run parameters from OUTCAR and CONTCAR")
            contcar_path = os.path.join(path, 'CONTCAR')
            if os.path.exists(contcar_path) and os.path.getsize(contcar_path) > 0:
                contcar = Poscar.from_file(os.path.join(path, 'CONTCAR'))
            else:
                return None
            if os.path.exists(os.path.join(path, 'INCAR')):
                incar = Incar.from_file(os.path.join(path, 'INCAR'))
            else:
                incar = None
            if os.path.exists(os.path.join(path, 'POSCAR')):
                initial_structure = Poscar.from_file(os.path.join(path, 'POSCAR')).structure
            else:
                initial_structure = None
            final_energy = outcar.final_energy
            if not isinstance(final_energy, float):
                warnings.warn(
                    f"Problems reading final energy (={final_energy}) from {outcar_path}."
                )
                final_energy = np.nan
            outcar.read_pattern(
                {'converged_ionic':
                 r'(reached required accuracy - stopping structural energy minimisation|writing wavefunctions)'},
                reverse=True,
                terminate_on_match=True
            )
            converged_ionic = True if outcar.data['converged_ionic'] else False
            outcar.read_pattern(
                {'converged_electronic':
                 r'aborting loop (EDIFF was not reached \(unconverged\)|because EDIFF is reached)'},
                reverse=True,
                terminate_on_match=True
            )
            converged_electronic = outcar.data['converged_electronic']
            if converged_electronic and\
               converged_electronic[0][0] == 'because EDIFF is reached':
                converged_electronic = True
            else:
                converged_electronic = False
            computed_entry = ComputedStructureEntry(
                structure=contcar.structure,
                composition=contcar.structure.composition,
                energy=final_energy,
                data={
                    'incar': incar,
                    'initial_structure': initial_structure,
                    'converged': converged_ionic and converged_electronic,
                    'filename': outcar_path,
                    'final_energy': final_energy})

        if outcar is None:
            computed_entry.data['outcar'] = None
        else:
            computed_entry.data['outcar'] = outcar.as_dict()

        return computed_entry


def read_vaspruns(rootdir, path_filter=None):
    """Read all vaspruns in directory (nested).

    Args:
        rootdir (str): Root directory.
        path_filter(function(PathLike): Function returning True for
        paths that should be read.
    Returns: List of Vasprun objects.
    """
    drone = IMDGVaspToComputedEnrgyDrone(
        inc_structure=True,
        data=["filename", "initial_structure", "incar", 'converged'])

    if os.access(rootdir, os.W_OK):
        dump_file = os.path.join(rootdir, SAVE_FILE)
    else:
        dump_file = SAVE_FILE
    queen = IMDGBorgQueen(
        drone,
        rootpath=rootdir,
        dump_file=dump_file,
        path_filter=path_filter)
    queen.save_data(dump_file)

    entries = queen.get_data()
    entries = [e for e in entries if e is not None]

    if len(entries) > 0:
        return entries

    os.unlink(SAVE_FILE)
    return None


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Analyze vasp outputs."""

    parser.add_argument(
        "dir",
        help="""Directory to read (recusrively); defaults to current dir""",
        type=str,
        nargs="?",
        default=".")

    all_fileds = [
        'energy', 'e_per_atom', 'total_mag', '%vol',
        'a', 'b', 'c', '%a', '%b', '%c',
        'alpha', 'beta', 'gamma',
        '%alpha', '%beta', '%gamma'
    ]
    parser.add_argument(
        "--fields",
        help="""List of fields to report
energy: Final energy
e_per_atom: Final energy per atom
%%vol: Volume change before/after the run
a, b, c, alpha, beta, gamma: Lattice parameters
%%a, %%b, %%c, %%alpa, %%beta, %%gamma: Change before/after the run
""",
        nargs="+",
        choices=all_fileds,
        default=all_fileds
        )
    parser.add_argument(
        "--exclude-fields",
        dest='exclude_fields',
        help="""List of fields to NOT report""",
        nargs="+",
        choices=all_fileds,
        default=[]
        )
    parser.add_argument(
        "--group",
        help="""Group by similar INCARs""",
        action="store_true"
        )


def analyze(args):
    """Main routine.
    """
    entries = read_vaspruns(args.dir)
    entries = sorted(entries, key=lambda x: x.data["filename"])
    all_data = {}
    for field, header in [
            ('dir', 'Directory'), ('incar_group', 'INCAR type'),
            ('energy', 'Energy'), ('e_per_atom', 'E/Atom'),
            ('total_mag', 'Magnetization'),
            ('%vol', '%vol'),
            ('a', 'a'), ('b', 'b'), ('c', 'c'),
            ('%a', '%a'), ('%b', '%b'), ('%c', '%c'),
            ('alpha', 'α'), ('beta', 'β'), ('gamma', 'γ'),
            ('%alpha', '%α'), ('%beta', '%β'), ('%gamma', '%γ')]:
        if field == 'dir' or (field == 'incar_group' and args.group) or\
           (field in args.fields and field not in args.exclude_fields):
            all_data[field] = {'header': header, 'data': []}

    file_groups = {}
    if args.group:
        incars = []
        for e in entries:
            incar = e.data["incar"]
            incar['SYSTEM'] = e.data['filename']
            incars.append(incar)
        _, groups = Incar.group_incars(incars)
        if len(groups) > 1:
            for idx, group in enumerate(groups):
                for incar in group:
                    file_groups[incar['SYSTEM'].upper()] = idx

    def _energy_reliable_p(entry):
        """Return True when ENTRY's energy is reliable.
        Energy is not very reliable when using volume relaxation or
        when the calculation is not converged.
        """
        incar = entry.data['incar']
        if incar.get('IBRION') in Incar.IBRION_IONIC_RELAX_values and\
           incar.get('ISIF') != Incar.ISIF_FIX_SHAPE_VOL:
            return False
        if not entry.data.get('converged', False):
            return False
        return True

    for e in entries:
        for field, field_val in all_data.items():
            val = None

            if field == 'dir':
                val = os.path.dirname(e.data['filename'])
                val = val.replace("./", "")
            elif field == 'incar_group' and args.group:
                val = file_groups[e.data['filename'].upper()]\
                    if len(file_groups) > 0 else 0
            elif field == 'energy':
                if _energy_reliable_p(e):
                    val = f"{e.energy:.5f}"
                else:
                    val = "unreliable"
            elif field == 'e_per_atom':
                if _energy_reliable_p(e):
                    val = f"{e.energy_per_atom:.5f}"
                else:
                    val = "unreliable"
            elif field == 'total_mag':
                if e.data['outcar'] is not None:
                    val = e.data['outcar']['total_magnetization']
                if val is None:
                    val = "None"
            elif field == '%vol':
                vol0 = e.data["initial_structure"].volume
                val = e.structure.volume/vol0 - 1
                val = f"{val * 100:.2f}"
            elif field == 'a':
                val = e.structure.lattice.a
            elif field == '%a':
                a0 = e.data['initial_structure'].lattice.a
                val = e.structure.lattice.a/a0 - 1
                val = f"{val * 100:.2f}"
            elif field == 'b':
                val = e.structure.lattice.b
            elif field == '%b':
                b0 = e.data['initial_structure'].lattice.b
                val = e.structure.lattice.b/b0 - 1
                val = f"{val * 100:.2f}"
            elif field == 'c':
                val = e.structure.lattice.c
            elif field == '%c':
                c0 = e.data['initial_structure'].lattice.c
                val = e.structure.lattice.c/c0 - 1
                val = f"{val * 100:.2f}"
            elif field == 'alpha':
                val = e.structure.lattice.alpha
            elif field == '%alpha':
                alpha0 = e.data['initial_structure'].lattice.alpha
                val = e.structure.lattice.alpha/alpha0 - 1
                val = f"{val * 100:.2f}"
            elif field == 'beta':
                val = e.structure.lattice.beta
            elif field == '%beta':
                beta0 = e.data['initial_structure'].lattice.beta
                val = e.structure.lattice.beta/beta0 - 1
                val = f"{val * 100:.2f}"
            elif field == 'gamma':
                val = e.structure.lattice.gamma
            elif field == '%gamma':
                gamma0 = e.data['initial_structure'].lattice.gamma
                val = e.structure.lattice.gamma/gamma0 - 1
                val = f"{val * 100:.2f}"

            field_val['data'].append(val)

    if len(all_data) > 0 and len(entries) > 0:
        data = [[val['data'][idx] for _, val in all_data.items()]
                for idx in range(len(all_data['dir']['data']))]
        headers = [val['header'] for _, val in all_data.items()]
        if args.group:
            group_idx = None
            for idx, field in enumerate(all_data.keys()):
                if field == "incar_group":
                    group_idx = idx
                    break
            data = sorted(data, key=lambda x: x[group_idx])
        print(tabulate(data, headers=headers, tablefmt="orgtbl"))

    return 0
