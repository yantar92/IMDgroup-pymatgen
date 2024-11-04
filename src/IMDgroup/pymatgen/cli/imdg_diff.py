"""imdg sub-command to compare VASP inputs/outputs
"""
import logging
import os
import shutil
import math
from alive_progress import alive_bar
from termcolor import colored
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
from IMDgroup.pymatgen.io.vasp.outputs import Vasprun
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDGroup.common import groupby_cmp

logger = logging.getLogger(__name__)


def structure_add_args(parser):
    """Setup parser arguments for structure comparison.
    Args:
      parser: subparser
    """
    parser.help = "Compare VASP structures from VASP outputs or from POSCARs"
    parser.set_defaults(func_diff=diff_structures)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--poscar",
        action='store_true',
        help="Force comparing POSCARs"
    )
    group.add_argument(
        "--vasprun",
        action='store_true',
        help="Force comparing VASP outputs"
    )

    parser.add_argument(
        "--copy-to",
        dest="copy_to",
        help="Copy the unique structures to specified directory",
        type=str
    )

    parser.add_argument(
        "--energy-tol",
        dest="energy_tol",
        help="Energy tolerance when comparing structures"
        " (number of digits after dot)",
        default=3,
        type=int
    )


def _read_structures(dir_list, force_poscar=False, force_vasprun=False):
    """Read structures from DIR_LIST.
    When all the directories contain VASP outputs, use final_structure.
    When all the directories do not have VASP outputs, use POSCAR.
    If mixed, throw an error.
    Args:
      dir_list (list[str]): List of directories to read.
      force_poscar (bool):  Force reading POSCAR.
      force_vasprun (bool): Force reading Vasprun.
    Returns: list[Structure]
      List of structures.  Each structure will have its 'source_dir'
      property set to its origin directory.
      When a structure is read from vasprun.xml, it will also have
      'final_energy' property set to the final structure energy.
    """
    def read_vasprun(vaspdir):
        logger.info("Reading vasprun.xml from %s", vaspdir)
        run = Vasprun(os.path.join(vaspdir, "vasprun.xml"))
        res_structure = run.final_structure
        res_structure.properties['source_dir'] = vaspdir
        res_structure.properties['final_energy'] = run.final_energy
        return res_structure

    def read_poscar(vaspdir):
        logger.info("Reading POSCAR from %s", vaspdir)
        poscar = Poscar.from_file(os.path.join(vaspdir, "POSCAR"))
        res_structure = poscar.structure
        res_structure.properties['source_dir'] = vaspdir
        return res_structure

    used_vasp_output = False
    used_poscar_output = False
    structures = []
    with alive_bar(len(dir_list), title='Reading VASP dirs') as abar:
        for vaspdir in dir_list:
            abar.text = f'{vaspdir}'
            if force_poscar:
                structures.append(read_poscar(vaspdir))
            else:
                try:
                    structures.append(read_vasprun(vaspdir))
                    used_vasp_output = True
                except FileNotFoundError:
                    if used_vasp_output or used_poscar_output or force_vasprun:
                        raise
                    structures.append(read_poscar(vaspdir))
                    used_poscar_output = True
            abar()  # pylint: disable=not-callable
    return structures


def _copy_structures_to(structures, directory):
    """Copy directories containing STRUCTURES to DIRECTORY.
    Args:
      structures (list[Structure]):
        List of structures.  Each member of the list must have its
        'source_dir' propety set.
      directory (str):
        Directory to copy source dirs to.
    """
    source_dirs = [s.properties['source_dir'] for s in structures]
    subdirs = []
    print(f"Copying unique structures to '{directory}'...", end='', flush=True)
    for idx, source in enumerate(source_dirs):
        if os.path.isabs(source):
            target_subdir = os.path.basename(source)
        else:
            target_subdir = source
        # Avoid directories with the same name
        if target_subdir in subdirs:
            target_subdir = target_subdir + f".{idx + 1}"
        shutil.copytree(source, os.path.join(directory, target_subdir))
    print(f"\rCopying unique structures to '{directory}'...",
          colored('done', 'green'))


def diff_structures(args):
    """Compare structures.
    """
    structures = _read_structures(sorted(args.dirs), args.poscar, args.vasprun)
    # 2x tigher tolerance compared to the defaults.
    # With defaults, structures different by ~0.1eV could be matched
    # according to tests.
    # Going even lower does not help much though - we have to account
    # actual energy from VASP to distinguish structures.
    matcher = StructureMatcher(ltol=0.1, stol=0.15, angle_tol=2.5)

    def _structures_eq(str1, str2):
        """Compare structures STR1 and STR2 for equality.
        Return True when they are equal.
        """
        str1_energy = str1.properties.get('final_energy', None)
        str2_energy = None
        if str1_energy:
            # Err if energy is available in one structure, but not
            # another.
            str2_energy = str2.properties['final_energy']
        if (str1_energy is None or
            math.isclose(
                str1_energy, str2_energy,
                abs_tol=math.pow(10, -args.energy_tol))):
            if matcher.fit(str1, str2):
                return True
        return False

    def _structure_name(struct):
        """Get directory."""
        return struct.properties['source_dir']

    groups = groupby_cmp(structures, _structures_eq, _structure_name)

    print(colored(
        "List of structures grouped by similarity",
        'magenta', attrs=['bold']))

    for idx, group in enumerate(groups):
        print(colored(f"Group {idx + 1}: ", attrs=['bold']), end='')
        if 'final_energy' in group[0].properties:
            final_energy = round(
                group[0].properties['final_energy'],
                args.energy_tol)
            print(f"Energy={final_energy}eV ", end='')
        else:
            print("Energy=N/A ", end='')
        for s in group:
            print(s.properties['source_dir'], ' ', end='')
        print()

    if args.copy_to is not None:
        _copy_structures_to([group[0] for group in groups], args.copy_to)


def incar_add_args(parser):
    """Setup parser arguments for incar comparison.
    Args:
      parser: subparser
    """
    parser.help = "Compare INCARs from VASP dirs"
    parser.set_defaults(func_diff=diff_incar)


def diff_incar(args):
    """Handle diff commands.

    Args:
        args: Args from command.
    """

    incars = []
    with alive_bar(len(args.dirs), title='Reading INCARs') as abar:
        for vaspdir in args.dirs:
            abar.text = f'{vaspdir}'
            incar = Incar.from_file(os.path.join(vaspdir, "INCAR"))
            # Abuse unused SYSTEM parameter to store directory name.
            incar['SYSTEM'] = vaspdir
            incars.append(incar)
            abar()  # pylint: disable=not-callable

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

    print(colored("Common INCAR parameters", attrs=['bold']))
    print(common_incar.get_str(pretty=True))

    for idx, group in enumerate(groups):
        print(colored(f"Group {idx + 1}: ", attrs=['bold']), end='')
        print(' '.join(_incar_name(incar) for incar in group))
        print(
            colored(f"Group {idx + 1} params: ", attrs=['bold']),
            ' '.join(
                f"{key}:{val}" for key, val in group[0].items()
                if key not in common_incar and key != "SYSTEM"))

    return 0


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Compare VASP inputs/outputs."""

    parser.add_argument(
        "dirs",
        help="VASP directories to compare",
        nargs="+",
        type=str
    )

    subparsers = parser.add_subparsers(required=True)

    parser_structure = subparsers.add_parser("structure")
    structure_add_args(parser_structure)

    parser_incar = subparsers.add_parser("incar")
    incar_add_args(parser_incar)


def diff(args):
    """Main routine.
    """
    args.func_diff(args)
    return 0
