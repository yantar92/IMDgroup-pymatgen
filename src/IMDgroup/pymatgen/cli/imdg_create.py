"""imdg sub-command to create new VASP inputs from scratch.
"""
import logging
import pymatgen.core as pmg
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from IMDgroup.pymatgen.io.vasp.sets import IMDStandardVaspInputSet

logger = logging.getLogger(__name__)


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Create new VASP inputs from scratch."""

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        help="Input structure file (any file that can be read by pymatgen)",
        type=str
    )
    group.add_argument(
        "--mpid",
        help="Materials Project structure id",
        type=str
    )


def create(args):
    """Main routine.
    """
    if args.mpid is not None:
        structure = pmg.Structure.from_id(args.mpid)
        structure.properties['mpid'] = args.mpid
        logger.info(
            "Downloaded structure from Materials Project db:\n%s",
            structure
        )
    elif args.file is not None:
        structure = pmg.Structure.from_file(args.file)
        logger.info(
            "Loaded structure from %s:\n%s",
            args.file, structure
        )
    else:
        raise AttributeError("This should not happen.")

    # Sometimes, Materials Project does not return standardized structure
    # Force it
    analyzer = SpacegroupAnalyzer(structure)
    standard_structure = analyzer.get_primitive_standard_structure()
    standard_structure.properties = structure.properties

    if standard_structure.lattice != structure.lattice:
        logger.info(
            "Non-standardized structure converted:\n%s",
            standard_structure
        )

    inputset = IMDStandardVaspInputSet(structure)
    inputset.write_input(output_dir=inputset.incar['SYSTEM'])

    return 0
