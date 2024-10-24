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
        "--supercell",
        help="Supercell size (default: 1x1x1)",
        default="1x1x1")
    group.add_argument(
        "--functional",
        help="Functional to be used",
        choices=[
            'PBE', 'PBEsol', 'PBE+D2', 'PBE+TS',
            'vdW-DF', 'vdW-DF2',
            'optB88-vdW', 'optB86b-vdW'],
        type=str)


def derive(args):
    """Main routine.
    """

    if args.supercell is not None:
        inputset = IMDDerivedInputSet(directory=args.input_directory)
        # supercell: N1xN2xN3 string
        scaling = [int(x) for x in args.supercell.split("x")]
        inputset.structure.make_supercell(scaling)
        output_suffix = args.supercell
    elif args.functional is not None:
        inputset = IMDDerivedInputSet(
            directory=args.input_directory,
            functional=args.functional)
    else:
        return 1

    if "SYSTEM" in inputset.incar:
        system_name = inputset.incar["SYSTEM"]
    else:
        system_name = "unknown"

    inputset.write_input(output_dir=f'{system_name}.{output_suffix}')

    return 0
