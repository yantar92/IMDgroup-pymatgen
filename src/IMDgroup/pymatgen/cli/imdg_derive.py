"""imdg sub-command to create new VASP inputs from existing.
"""
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet

ISIF_RELAX_POS = ISIF_FIX_SHAPE_VOL = 2
ISIF_RELAX_POS_SHAPE_VOL = ISIF_FIX_NONE = 3
ISIF_RELAX_POS_SHAPE = ISIF_FIX_VOL = 4
ISIF_RELAX_SHAPE = IFIX_FIX_POS_VOL = 5
ISIF_RELAX_SHAPE_VOL = ISIF_FIX_POS = 6
ISIF_RELAX_VOL = ISIF_FIX_POS_SHAPE = 7
ISIF_RELAX_POS_VOL = ISIF_FIX_SHAPE = 8

IBRION_IONIC_RELAX_CGA = 2


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
        type=str)
    group.add_argument(
        "--functional",
        help="Functional to be used",
        choices=[
            'PBE', 'PBEsol', 'PBE+D2', 'PBE+TS',
            'vdW-DF', 'vdW-DF2',
            'optB88-vdW', 'optB86b-vdW'],
        type=str)
    group.add_argument(
        "--relax",
        help="Relax system",
        choices=[
            "RELAX_POS", "FIX_SHAPE_VOL",
            "RELAX_POS_SHAPE_VOL", "FIX_NONE",
            "RELAX_POS_SHAPE", "FIX_VOL",
            "RELAX_SHAPE", "FIX_POS_VOL",
            "RELAX_SHAPE", "FIX_POS_VOL",
            "RELAX_SHAPE_VOL", "FIX_POS",
            "RELAX_VOL", "FIX_POS_SHAPE",
            "RELAX_POS_VOL", "FIX_SHAPE"
        ]
    )


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
        output_suffix = args.functional
    elif args.relax is not None:

        relax_overrides = {
            "ISTART": 0,
            # Volume relaxation
            # 500 steps because 100 suggested in some online resources
            # may not be enough in complex supercells.
            "NSW": 500,
            "IBRION": IBRION_IONIC_RELAX_CGA,
            'ISIF': globals()["ISIF_" + args.relax],
            'EDIFF': 1e-06,
            'EDIFFG': -0.01
        }

        inputset = IMDDerivedInputSet(
            directory=args.input_directory,
            user_incar_settings=relax_overrides,
            )
        output_suffix = f"relax.{args.relax}"
    else:
        return 1

    if "SYSTEM" in inputset.incar:
        system_name = inputset.incar["SYSTEM"]
    else:
        system_name = "unknown"

    inputset.write_input(output_dir=f'{system_name}.{output_suffix}')

    return 0
