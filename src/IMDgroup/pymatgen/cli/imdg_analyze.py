"""Analysis extensions specific to IMD group.
Based on pymatgen's pymatgen.cli.pmg_analyze
"""
import logging
import multiprocessing
import os
from tabulate import tabulate

from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen

SAVE_FILE = "vasp_data.gz"
logger = logging.getLogger(__name__)


def _read_vaspruns(rootdir, reanalyze):
    """Read all vaspruns in directory (nested).

    Args:
        rootdir (str): Root directory.
        reanalyze (bool): Whether to ignore saved results and reanalyze
    Returns: List of Vasprun objects.
    """
    drone = VaspToComputedEntryDrone(
        inc_structure=True,
        data=["filename", "initial_structure"])

    n_cpus = multiprocessing.cpu_count()
    logger.info("Detected %d cpus", n_cpus)
    queen = BorgQueen(drone, number_of_drones=n_cpus)
    if os.path.isfile(SAVE_FILE) and not reanalyze:
        msg = (f"Using previously assimilated data from {SAVE_FILE}. "
               "Use -r to force re-analysis.")
        print(msg)
        queen.load_data(SAVE_FILE)
    else:
        if n_cpus > 1:
            queen.parallel_assimilate(rootdir)
        else:
            queen.serial_assimilate(rootdir)
        msg = (f"Analysis results saved to {SAVE_FILE} "
               "for faster subsequent loading.")
        queen.save_data(SAVE_FILE)

    entries = queen.get_data()

    if len(entries) > 0:
        logger.info(msg)
        return entries

    logger.info("No valid VASP run found.")
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
        help="""Directory to read (recusrively)""",
        type=str)

    parser.add_argument(
        "--reanalyze", "-r",
        help="Force re-reading VASP sources",
        action="store_true"
    )

    parser.add_argument(
        "--fields",
        help="""List of fields to report
energy: Final energy
e_per_atom: Final energy per atom
%vol: Volume change before/after the run
a, b, c, alpha, beta, gamma: Lattice parameters
%a, %b, %c, %alpa, %beta, %gamma: Change before/after the run
""",
        nargs="+",
        choices=[
            'energy', 'e_per_atom', '%vol',
            'a', 'b', 'c', '%a', '%b', '%c',
            'alpha', 'beta', 'gamma',
            '%alpha', '%beta', '%gamma'
        ],
        default=[
            'energy', 'e_per_atom', '%vol',
            'a', '%a', 'b', '%b', 'c', '%c',
            'alpha', '%alpha',
            'beta', '%beta',
            'gamma', '%gamma']
        )


def analyze(args):
    """Main routine.
    """
    entries = _read_vaspruns(args.dir, args.reanalyze)
    all_data = {}
    for field, header in [
            ('dir', 'Directory'),
            ('energy', 'Energy'), ('e_per_atom', 'E/Atom'),
            ('a', 'a'), ('b', 'b'), ('c', 'c'),
            ('%a', '%a'), ('%b', '%b'), ('%c', '%c'),
            ('alpha', 'α'), ('beta', 'β'), ('gamma', 'γ'),
            ('%alpha', '%α'), ('%beta', '%β'), ('%gamma', '%γ')]:
        if field == 'dir' or field in args.fields:
            all_data[field] = {'header': header, 'data': []}

    for e in entries:
        for field, field_val in all_data.items():
            val = None

            if field == 'dir':
                val = os.path.dirname(e.data['filename'])
                val = val.replace("./", "")
            elif field == 'energy':
                val = f"{e.energy:.5f}"
            elif field == 'e_per_atom':
                val = f"{e.energy_per_atom:.5f}"
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
        print(tabulate(
            [[val['data'][idx] for _, val in all_data.items()]
             for idx in range(len(all_data['dir']['data']))],
            headers=[val['header'] for _, val in all_data.items()],
            tablefmt="orgtbl"))

    return 0
