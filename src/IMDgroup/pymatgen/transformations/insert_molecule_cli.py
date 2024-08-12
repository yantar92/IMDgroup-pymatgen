"""Command line interface to insert molecule into a given structure
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import pymatgen.io.vasp.sets as vaspset
# Used by str_to_class
import IMDgroup.pymatgen.io.vasp.sets
from .insert_molecule import InsertMoleculeTransformation

logger = logging.getLogger(__name__)
assert IMDgroup.pymatgen.io.vasp.sets  # silence linters


def get_args():
    """Parse command line args and return arg dictionary."""
    argparser = argparse\
        .ArgumentParser(description="Generate structures for all possible \
        insertions of MOLECULE into STRUCTURE")
    argparser.add_argument(
        "molecule", help="file storing molecule to be inserted")
    argparser.add_argument("structure", help="file storing the host structure")
    argparser.add_argument("output_dir", help="directory to write the result")
    argparser.add_argument(
        "--step", type=float,
        help="search step in angstrems (default: 0.5 ans)")
    argparser.add_argument(
        "--anglestep", type=float,
        help="rotation step in deg (default: no rotation)")
    argparser.add_argument(
        "--randomize_molecule", action="count",
        help="randomize initial molecule orientation", default=None)
    argparser.add_argument(
        "--vasp_input", type=str,
        help="VASPInputSet name (default: IMDRelaxCellulose); \
        https://pymatgen.org/pymatgen.io.vasp.html\
        #module-pymatgen.io.vasp.sets",
        default="IMDRelaxCellulose")
    argparser.add_argument(
        "-v", "--verbosity", action="count",
        help="log verbosity (default: 0)", default=0)
    return argparser.parse_args()


def setup_logger(args):
    """Setup logging according to command line args."""
    log_file =\
        os.path.join(args.output_dir, "pmg-insert-molecule-settings.log")

    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    log_format = '[%(levelname)s] [%(asctime)s] %(message)s'
    date_format = "%b %d %X"

    if args.verbosity >= 2:
        logging.basicConfig(
            format=log_format, datefmt=date_format,
            level=logging.DEBUG, handlers=[file_handler, stdout_handler])
        logging.info("Setting debug level to: DEBUG")
    elif args.verbosity >= 1:
        logging.basicConfig(
            format=log_format, datefmt=date_format,
            level=logging.INFO, handlers=[file_handler, stdout_handler])
        logging.info("Setting debug level to: INFO")
    else:
        logging.basicConfig(
            format=log_format, datefmt=date_format,
            level=logging.INFO, handlers=[file_handler])
        logging.info("Setting debug level to: INFO (writing to file)")


def main():
    """Run script."""

    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args)

    logger.info("Starting up with arguments: %s", args)

    # FIXME: StructureMatcher is the default, but it is way too slow
    # We need more optimizations
    transformer = InsertMoleculeTransformation(
        args.molecule, step=args.step,
        anglestep=None
        if args.anglestep is None
        else np.radians(args.anglestep),
        matcher=None)

    if args.randomize_molecule is not None:
        transformer.rotate_molecule_euler(np.random.rand(3)*2*np.pi)

    structures = transformer.all_inserts(args.structure)

    def str_to_class(classname):
        """Return class named classname or None.
        Search classes in pymatgen.io.vasp.sets and
        pymatgen.io.vasp.IMDgroup.sets.
        """
        try:
            return getattr(sys.modules['pymatgen.io.vasp.sets'],
                           classname)
        except AttributeError:
            try:
                return getattr(sys.modules['IMDgroup.pymatgen.io.vasp.sets'],
                               classname)
            except AttributeError:
                return None

    vaspset.batch_write_input(
        structures,
        vasp_input_set=str_to_class(args.vasp_input),
        output_dir=args.output_dir,
        potcar_spec=True)

    settings_file = os.path.join(
        args.output_dir, "pmg-insert-molecule-settings.yaml")
    with open(settings_file, "w", encoding="utf-8") as file:
        yaml.dump(vars(args), file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
