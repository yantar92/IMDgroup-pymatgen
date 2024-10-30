"""imdg sub-command to compare VASP inputs/outputs
"""
import logging
import os
import shutil
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
    """
    def read_vasprun(vaspdir):
        logger.info("Reading vasprun.xml from %s", vaspdir)
        run = Vasprun(os.path.join(vaspdir, "vasprun.xml"))
        res_structure = run.final_structure
        res_structure.properties['source_dir'] = vaspdir
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
    structures = _read_structures(args.dirs, args.poscar, args.vasprun)
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
                if matcher.fit(structure, group_structure):
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
