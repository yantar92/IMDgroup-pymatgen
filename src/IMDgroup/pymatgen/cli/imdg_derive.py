"""imdg sub-command to create new VASP inputs from existing.
"""
import warnings
import argparse
import dataclasses
import numpy as np
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
from IMDgroup.pymatgen.io.vasp import sets


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
    incar_add_args(parser_incar)

    parser_supercell = subparsers.add_parser("supercell")
    supercell_add_args(parser_supercell)

    parser_functional = subparsers.add_parser("functional")
    functional_add_args(parser_functional)

    parser_relax = subparsers.add_parser("relax")
    relax_add_args(parser_relax)

    parser_kpoints = subparsers.add_parser("kpoints")
    kpoints_add_args(parser_kpoints)

    parser_strain = subparsers.add_parser("strain")
    strain_add_args(parser_strain)


def _str_to_bool(value):
    """Convert string value to boolean.
    """
    if value.lower() in ['true', '1', 'yes']:
        return True
    if value.lower() in ['false', '0', 'no']:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def strain_add_args(parser):
    """Setup parser arguments for strain.
    Args:
      parser: subparser
    """
    parser.help = "Created strained input"
    parser.set_defaults(func_derive=strain)
    for name in ["a", "b", "c"]:
        parser.add_argument(
            "--" + name + "min",
            help=f"Min value of {name} lattice parameter, %% of initial",
            type=float,
            default=100
        )
        parser.add_argument(
            "--" + name + "max",
            help=f"Max value of {name} lattice parameter, %% of initial",
            type=float,
            default=100
        )
        parser.add_argument(
            "--" + name + "steps",
            help=f"Number of strain steps along {name} (default: 1)",
            type=int,
            default=1
        )
    parser.add_argument(
        "--selective-dynamics",
        dest="selective_dynamics",
        help="Selective dynamics to be applied to the sites"
        "(example: True, True, False)",
        nargs=3,
        type=_str_to_bool,
        default=None
    )


def strain(args):
    """Create strained input
    Return (inputset, output_dir)
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)

    if args.selective_dynamics is not None:
        for site in inputset.structure:
            site.properties['selective_dynamics'] =\
                args.selective_dynamics

    structure0 = inputset.structure

    strainsa = np.linspace(
        args.amin / 100.0 - 1.0,
        args.amax / 100.0 - 1.0,
        args.asteps
    )
    strainsb = np.linspace(
        args.bmin / 100.0 - 1.0,
        args.bmax / 100.0 - 1.0,
        args.bsteps
    )
    strainsc = np.linspace(
        args.cmin / 100.0 - 1.0,
        args.cmax / 100.0 - 1.0,
        args.csteps
    )

    strains = [[straina, strainb, strainc]
               for straina in strainsa
               for strainb in strainsb
               for strainc in strainsc]

    outputs = []
    for strain in strains:
        inputset_new = dataclasses.replace(inputset)  # copy
        inputset_new.structure =\
            structure0.apply_strain(strain, inplace=False)
        output_dir = (f"strain.a.{strain[0]:f.2}"
                      f".b.{strain[1]:f.2}"
                      f".c.{strain[2]:f.2}")
        outputs.append((inputset_new, output_dir))

    return outputs


def relax_add_args(parser):
    """Setup parser arguments for relax.
    Args:
      parser: subparser
    """
    parser.help = "Create relaxation input"
    parser.set_defaults(func_derive=relax)
    parser.add_argument(
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
        "IBRION": sets.IBRION_IONIC_RELAX_CGA,
        'ISIF': vars(sets)["ISIF_" + args.isif],
        'EDIFF': 1e-06,
        'EDIFFG': -0.01
    }
    # 550eV recommended for _volume/shape_ relaxation During
    # volume/shape relaxation, initial automatic k-point grid
    # calculated for original volume becomes slightly less accurate
    # unless we increase ENCUT
    if args.isif in [sets.ISIF_RELAX_POS_SHAPE, sets.ISIF_RELAX_SHAPE,
                     sets.ISIF_RELAX_SHAPE_VOL, sets.ISIF_RELAX_VOL,
                     sets.ISIF_RELAX_POS_VOL]:
        relax_overrides['ENCUT'] = 550.0
    else:
        relax_overrides['ENCUT'] = 500.0

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=relax_overrides,
    )
    output_dir = f"relax.{args.isif}"
    return (inputset, output_dir)


def supercell_add_args(parser):
    """Setup parser arguments for supercell.
    Args:
      parser: subparser
    """
    parser.help = "Create supercell from input (rescaling k-points)"
    parser.set_defaults(func_derive=supercell)
    parser.add_argument(
        "supercell_size",
        help="Supercell size",
        type=str)
    parser.add_argument(
        "--kpoint-density",
        dest="kpoint_density",
        help="K-point density to be used (default: 10000)",
        type=float,
        default=10000)


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


def functional_add_args(parser):
    """Setup parser arguments for functional.
    Args:
      parser: subparser
    """
    parser.help = "Create input with a given functional"
    parser.set_defaults(func_derive=functional)
    parser.add_argument(
        "functional_type",
        help="Functional to be used",
        choices=[
            'PBE', 'PBEsol', 'PBE+D2', 'PBE+TS',
            'vdW-DF', 'vdW-DF2',
            'optB88-vdW', 'optB86b-vdW'],
        type=str)


def functional(args):
    """Create custom functional setup.
    Return (inputset, output_dir)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        functional=args.functional_type)
    output_dir = args.functional_type
    return (inputset, output_dir)


def incar_add_args(parser):
    """Setup parser arguments for incar.
    Args:
      parser: subparser
    """
    parser.help = "Modify incar"
    parser.set_defaults(func_derive=incar)
    parser.add_argument(
        "parameters",
        nargs="*",
        help="PARAM:VALUE to be set in the INCAR.",
        type=str)


def incar(args):
    """Create custom incar setup.
    Return (inputset, output_dir)
    """
    incar_overrides = {}
    if args.parameters is None:
        warnings.warn(
            "No INCAR settings provided.  Creating a copy of the inputs."
        )
    else:
        for str_val in args.parameters:
            key, val = str_val.split(":")
            incar_overrides[key] = val

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=incar_overrides,
    )
    output_dir = ','.join(
        [f'{key}.{val}' for key, val in incar_overrides.items()])
    return (inputset, output_dir)


def kpoints_add_args(parser):
    """Setup parser arguments for kpoints.
    Args:
      parser: subparser
    """
    parser.help = "Create input with custom kpoints settings"
    parser.set_defaults(func_derive=kpoints)
    parser.add_argument(
        "--density",
        help="K-point density to be used",
        type=float,
        default=10000
    )


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
    value_or_values = args.func_derive(args)
    try:
        inputset, output_dir = value_or_values
        inputset.write_input(output_dir=output_dir)
    except TypeError:
        for inputset, output_dir in value_or_values:
            inputset.write_input(output_dir=output_dir)

    return 0
