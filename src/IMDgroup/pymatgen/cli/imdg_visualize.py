"""Visualization extension specific to IMD group.
"""
import logging
import os
import numpy as np
from termcolor import colored
from pymatgen.core import Species, Structure
from pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.cli.imdg_analyze import read_vaspruns
from IMDgroup.pymatgen.cli.imdg_status import (convergedp, nebp)
from IMDgroup.pymatgen.core.structure import merge_structures

logger = logging.getLogger(__name__)


def write_selective_dynamics_summary_maybe(structure, fname):
    """Visualize site constrains in STRUCTURE and write to FNAME.
    Do nothing when STUCTURE does not have non-trivial constraints.
    Return True when FNAME has been produced.
    """
    has_fixed = False
    structure = structure.copy()
    for site in structure:
        if 'selective_dynamics' in site.properties and\
           np.array_equal(site.properties['selective_dynamics'],
                          [False, False, False]):
            has_fixed = True
            site.species = Species('Fe')  # fixed
        elif False in site.properties['selective_dynamics']:
            has_fixed = True
            site.species = Species('Co')  # partially fixed
        else:
            site.species = Species('Ni')  # not fixed
    if has_fixed:
        structure.to_file(fname)
        return True
    return False


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
    for parent, _subdirs, files in os.walk(args.dir):
        if 'POSCAR' in files:
            structure = Structure.from_file(os.path.join(parent, 'POSCAR'))
            cif_name = 'POSCAR.selective_dynamics.cif'
            cif_path = os.path.join(parent, cif_name)
            write_selective_dynamics_summary_maybe(structure, cif_path)
            logger.info("Saving illustration to %s", cif_path)
            print(colored(f"{parent.replace("./", "")}: ", attrs=['bold'])
                  + f"Saved selective_dynamics to {cif_name}")


def visualize(args):
    """Main routine.
    """
    return args.func_derive(args)
