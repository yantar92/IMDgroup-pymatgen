"""imdg sub-command to create new VASP inputs from existing.
"""
import warnings
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
import IMDgroup.pymatgen.io.vasp.sets as sets

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
    parser_incar.help = "Modify incar"
    parser_incar.set_defaults(func_derive=incar)
    parser_incar.add_argument(
        "parameters",
        nargs="*",
        help="PARAM:VALUE to be set in the INCAR.",
        type=str)

    parser_supercell = subparsers.add_parser("supercell")
    parser_supercell.help = "Create supercell from input (rescaling k-points)"
    parser_supercell.set_defaults(func_derive=supercell)
    parser_supercell.add_argument(
        "supercell_size",
        help="Supercell size",
        type=str)
    parser_supercell.add_argument(
        "--kpoint-density",
        dest="kpoint_density",
        help="K-point density to be used (default: 10000)",
        type=float,
        default=10000)

    parser_functional = subparsers.add_parser("functional")
    parser_functional.help = "Create input with a given functional"
    parser_functional.set_defaults(func_derive=functional)
    parser_functional.add_argument(
        "functional_type",
        help="Functional to be used",
        choices=[
            'PBE', 'PBEsol', 'PBE+D2', 'PBE+TS',
            'vdW-DF', 'vdW-DF2',
            'optB88-vdW', 'optB86b-vdW'],
        type=str)

    parser_relax = subparsers.add_parser("relax")
    parser_relax.help = "Create relaxation input"
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

    parser_kpoints = subparsers.add_parser("kpoints")
    parser_kpoints.help = "Create input with custom kpoints settings"
    parser_kpoints.set_defaults(func_derive=kpoints)
    parser_kpoints.add_argument(
        "--density",
        help="K-point density to be used",
        type=float,
        default=10000
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
        "IBRION": sets.IBRION_IONIC_RELAX_CGA,
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
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_kpoints_settings={'grid_density': args.kpoint_density})
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
    if args.parameters is None:
        warnings.warn("No INCAR settings provided.  Creating a copy of the inputs.")
    else:
        for str_val in args.parameters:
            key, val = str_val.split(":")
            incar_overrides[key] = val

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=incar_overrides,
    )
    output_dir = ','.join([f'{key}.{val}' for key, val in incar_overrides.items()])
    return (inputset, output_dir)


def kpoints(args):
    """Create custom kpoints setup.
    Return (inputset, output_dir)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_kpoints_settings={'grid_density': args.density},
    )
    output_dir = f"KPOINTS.{args.density}"
    return (inputset, output_dir)


def derive(args):
    """Main routine.
    """
    inputset, output_dir = args.func_derive(args)

    inputset.write_input(output_dir=output_dir)

    return 0
