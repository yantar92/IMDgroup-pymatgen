"""imdg sub-command to create new VASP inputs from existing.
"""
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Create new VASP inputs from existing."""

    parser.add_argument(
        "input_directory",
        default=".",
        help="VASP directory to read system"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-s", "--supercell",
        help="Supercell size (default: 1x1x1)",
        default="1x1x1")


def derive(args):
    """Main routine.
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)

    if args.supercell is not None:
        # supercell: N1xN2xN3 string
        scaling = [int(x) for x in args.supercell.split("x")]
        inputset.structure.make_supercell(scaling)
        output_suffix = args.supercell

    if "SYSTEM" in inputset.incar:
        system_name = inputset.incar["SYSTEM"]
    else:
        system_name = "unknown"

    inputset.write_input(output_dir=f'{system_name}.{output_suffix}')

    return 0
