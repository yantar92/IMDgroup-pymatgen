"""Visualization extension specific to IMD group.
"""
import logging
import os
from termcolor import colored
from pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.cli.imdg_analyze import read_vaspruns
from IMDgroup.pymatgen.cli.imdg_status import (convergedp, nebp)
from IMDgroup.pymatgen.core.structure import merge_structures

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
    entries = read_vaspruns(
        args.dir,
        # Parent directory is NEB
        path_filter=lambda p: nebp(os.path.dirname(p)))
    entries_dict = {}
    if entries is not None:
        entries_dict = {
            os.path.dirname(e.data['filename']): e
            for e in entries
        }
    else:
        logger.info("No NEB runs found")
        return 0
    for wdir, _, _ in os.walk(args.dir):
        if not nebp(wdir):
            continue
        if not convergedp(wdir, entries_dict):
            logger.info("Skipping unconverged run at %s", wdir)
            continue
        incar = Incar.from_file(os.path.join(wdir, 'INCAR'))
        images = [f"{n:02d}" for n in range(1, 1 + incar['IMAGES'])]
        neb_structures = []
        for image in images:
            neb_structures.append(
                entries_dict[os.path.join(wdir, image)].structure)
        trajectory = merge_structures(neb_structures)
        output_cif = os.path.join(wdir, 'NEB_trajectory_converged.cif')
        logger.info("Saving final trajectory to %s", output_cif)
        print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
              + colored("NEB ", "magenta")
              + f"Saved final trajectory for {wdir}")
        trajectory.to_file(output_cif)
    return 0


def visualize(args):
    """Main routine.
    """
    return args.func_derive(args)
