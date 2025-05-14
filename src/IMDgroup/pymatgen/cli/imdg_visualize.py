"""Visualization extension specific to IMD group.
"""
import logging
import os
from termcolor import colored
from pymatgen.core import Structure
from IMDgroup.pymatgen.core.structure import merge_structures
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
from IMDgroup.pymatgen.io.vasp.sets import write_selective_dynamics_summary_maybe

logger = logging.getLogger(__name__)


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Visualize vasp outputs."""

    parser.add_argument(
        "dir",
        help="""Directory to read (recusrively); defaults to current dir""",
        type=str,
        nargs="?",
        default=".")

    subparsers = parser.add_subparsers(required=True)

    parser_neb = subparsers.add_parser("neb")
    neb_add_args(parser_neb)

    parser_selective_dynamics = subparsers.add_parser("selective_dynamics")
    selective_dynamics_add_args(parser_selective_dynamics)


def neb_add_args(parser):
    """Setup parser arguments for neb visualization.
    Args:
      parser: subparser
    """
    parser.help = "Visualize NEB outputs"
    parser.set_defaults(func_derive=neb)


def neb(args):
    """Create NEB visualization.
    """
    for wdir, subdirs, _ in os.walk(args.dir):
        subdirs.sort()  # this will make loop go in order
        vaspdir = IMDGVaspDir(wdir)
        if not vaspdir.nebp:
            continue
        if not vaspdir.converged:
            logger.info("Skipping unconverged run at %s", wdir)
            continue
        neb_structures = [
            (imagedir.final_structure or imagedir.structure)
            for imagedir in vaspdir.neb_dirs()]
        trajectory = merge_structures(neb_structures)
        cif_name = 'NEB_trajectory_converged.cif'
        output_cif = os.path.join(wdir, cif_name)
        logger.info("Saving final trajectory to %s", output_cif)
        print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
              + colored("NEB ", "magenta")
              + f"Saved final trajectory to {cif_name}")
        trajectory.to_file(output_cif)
    return 0


def selective_dynamics_add_args(parser):
    """Setup parser arguments for selective dynamics visualization.
    Args:
      parser: subparser
    """
    parser.help = "Visualize selective dynamics for POSCAR files"
    parser.set_defaults(func_derive=selective_dynamics)


def selective_dynamics(args):
    """Visualize selective_dynamics.
    """
    for parent, subdirs, files in os.walk(args.dir):
        subdirs.sort()  # this will make loop go in order
        if 'POSCAR' in files:
            structure = Structure.from_file(os.path.join(parent, 'POSCAR'))
            cif_name = 'selective_dynamics.cif'
            cif_path = os.path.join(parent, cif_name)
            write_selective_dynamics_summary_maybe(structure, cif_path)
            logger.info("Saving illustration to %s", cif_path)
            print(colored(f"{parent.replace("./", "")}: ", attrs=['bold'])
                  + f"Saved selective_dynamics to {cif_name}")


def visualize(args):
    """Main routine.
    """
    return args.func_derive(args)
