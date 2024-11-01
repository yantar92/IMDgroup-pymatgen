"""imdg sub-command to compare VASP inputs/outputs
"""
import logging
import os
import shutil
import math
from alive_progress import alive_bar
from termcolor import colored
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)


def structure_add_args(parser):
    """Setup parser arguments for structure comparison.
    Args:
      parser: subparser
    """
    parser.help = "Compare VASP structures from VASP outputs or from POSCARs"
    parser.set_defaults(func_diff=structure)

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
    for vaspdir in dir_list:
        with alive_bar(len(dir_list), title='Reading VASP dirs') as bar:
            bar()  # pylint: disable=not-callable
            bar.text = f'{vaspdir}'
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
            bar()  # pylint: disable=not-callable
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


def structure(args):
    """Compare structures.
    """
    structures = _read_structures(sorted(args.dirs), args.poscar, args.vasprun)
    groups = []

    matcher = StructureMatcher()

    def _add_to_groups(structure):
        """Add STRUCTURE to groups.
        If STRUCTURE is not the same with all structure in groups,
        create a separate group.
        Modifies "groups" by side effect.
        """
        in_group = False
        for group in groups:
            for group_structure in group:
                structure_energy = None
                group_energy = None
                if 'final_energy' in structure.properties:
                    structure_energy = structure.properties['final_energy']
                    group_energy = group_structure.properties['final_energy']
                if (structure_energy is None or
                    math.isclose(
                        structure_energy, group_energy,
                        abs_tol=math.pow(10, -args.energy_tol)))\
                   and matcher.fit(structure, group_structure):
                    logger.debug(
                        'Appending %s to existing group',
                        structure.properties['source_dir']
                    )
                    group.append(structure)
                    in_group = True
                    break
            if in_group:
                break
        if not in_group:
            logger.debug(
                'Creating a new group for %s',
                structure.properties['source_dir']
            )
            # pylint: disable=modified-iterating-list
            groups.append([structure])

    for s in structures:
        _add_to_groups(s)

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


def diff(args):
    """Main routine.
    """
    args.func_diff(args)
    return 0
