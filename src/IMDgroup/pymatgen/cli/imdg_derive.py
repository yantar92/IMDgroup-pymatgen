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

    subparsers = parser.add_subparsers(required=True)

    parser_incar = subparsers.add_parser("incar")
    parser_incar.set_defaults(func_derive=incar)
    parser_incar.add_argument(
        "parameter",
        action="append",
        help="PARAM:VALUE to be set in the INCAR.",
        type=str)

    parser_supercell = subparsers.add_parser("supercell")
    parser_supercell.set_defaults(func_derive=supercell)
    parser_supercell.add_argument(
        "supercell_size",
        help="Supercell size",
        type=str)

    parser_functional = subparsers.add_parser("functional")
    parser_functional.set_defaults(func_derive=supercell)
    parser_functional.add_argument(
        "functional_type",
        help="Functional to be used",
        choices=[
            'PBE', 'PBEsol', 'PBE+D2', 'PBE+TS',
            'vdW-DF', 'vdW-DF2',
            'optB88-vdW', 'optB86b-vdW'],
        type=str)

    parser_relax = subparsers.add_parser("relax")
    parser_relax.set_defaults(func_derive=relax)
    parser_relax.add_argument(
        "isif",
        help="What to relax",
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


def relax(args):
    """Create relaxation setup.
    Return (inputset, output_dir)
    """
    relax_overrides = {
        "ISTART": 0,
        # Volume relaxation
        # 500 steps because 100 suggested in some online resources
        # may not be enough in complex supercells.
        "NSW": 500,
        "IBRION": IBRION_IONIC_RELAX_CGA,
        'ISIF': globals()["ISIF_" + args.isif],
        'EDIFF': 1e-06,
        'EDIFFG': -0.01
    }

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=relax_overrides,
    )
    output_dir = f"relax.{args.isif}"
    return (inputset, output_dir)


def supercell(args):
    """Create supercell.
    Return (inputset, output_dir)
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)
    # supercell: N1xN2xN3 string
    scaling = [int(x) for x in args.supercell_size.split("x")]
    inputset.structure.make_supercell(scaling)
    output_dir = args.supercell_size
    return (inputset, output_dir)


def functional(args):
    """Create custom functional setup.
    Return (inputset, output_dir)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        functional=args.functional_type)
    output_dir = args.functional_type
    return (inputset, output_dir)


def incar(args):
    """Create custom incar setup.
    Return (inputset, output_dir)
    """
    incar_overrides = {}
    if args.parameter is None:
        warnings.warn("No INCAR settings provided.  Creating a copy of the inputs.")
    else:
        for str_val in args.incar:
            key, val = str_val.split(":")
            incar_overrides[key] = val

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=incar_overrides,
    )
    output_dir = f"{system_name}.{','.join([f'{key}.{val}' for key, val in args.incar.items()])}"
    return (inputset, output_dir)


def derive(args):
    """Main routine.
    """
    inputset, output_dir = args.func_derive(args)

    inputset.write_input(output_dir=output_dir)

    return 0
