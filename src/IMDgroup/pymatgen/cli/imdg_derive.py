"""imdg sub-command to create new VASP inputs from existing.
"""
import os
import re
import warnings
import argparse
import dataclasses
import logging
import numpy as np
from IMDgroup.pymatgen.io.vasp.sets import IMDDerivedInputSet
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.transformations.insert_molecule\
    import InsertMoleculeTransformation

logger = logging.getLogger(__name__)


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

    parser.add_argument(
        "--output",
        help="Directory to write the mutated VASP input"
        "(default: <old-name>.<suffix>)."
    )
    parser.add_argument(
        "--output-prefix",
        dest="output_prefix",
        help="Directory prefix to write the mutated VASP input",
        type=str
    )
    parser.add_argument(
        "--subdir",
        help="Write VASP input into a subdir instead of top level",
        type=str
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

    parser_perturb = subparsers.add_parser("perturb")
    perturb_add_args(parser_perturb)

    parser_scf = subparsers.add_parser("scf")
    scf_add_args(parser_scf)

    parser_insert = subparsers.add_parser("ins")
    insert_add_args(parser_insert)

    parser_delete = subparsers.add_parser("del")
    delete_add_args(parser_delete)


def _str_to_bool(value):
    """Convert string value to boolean.
    """
    if value.lower() in ['true', '1', 'yes']:
        return True
    if value.lower() in ['false', '0', 'no']:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def perturb_add_args(parser):
    """Setup parser arguments for perturb.
    Args:
      parser: subparser
    """
    parser.help = "Created perturbed input"
    parser.set_defaults(func_derive=perturb)

    parser.add_argument(
        "--distance",
        help="Perturbation distance in ans (default: 0.1ans)",
        type=float,
        default=0.1
    )


def perturb(args):
    """Create perturbed input
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)

    inputset.structure.perturb(args.distance)
    output_dir_suffix = f"PERTURB.{args.distance}"

    return (inputset, output_dir_suffix)


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
            help=f"Min value of {name} lattice parameter, "
            "in ans (10.0) or %% of initial (10%%)",
            type=str,
            default="100%"
        )
        parser.add_argument(
            "--" + name + "max",
            help=f"Max value of {name} lattice parameter, "
            "in ans (10.0) %% of initial (10%%)",
            type=str,
            default="100%"
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
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)

    if args.selective_dynamics is not None:
        for site in inputset.structure:
            site.properties['selective_dynamics'] =\
                args.selective_dynamics

    structure0 = inputset.structure

    # 10% -> 0.1; 1.3 -> 1.3/lattice constant
    for name in ["a", "b", "c"]:
        for suffix in ["min", "max"]:
            attr_name = name + suffix
            value = getattr(args, attr_name)
            if "%" in value:
                new_val = float(re.sub("%", "", value)/100.0 - 1.0)
            else:
                new_val = float(value)/getattr(structure0.lattice, name) - 1.0
            logger.debug("%s: %s -> %f", attr_name, value, new_val)
            locals[attr_name] = new_val

    strainsa = np.linspace(amin, amax, args.asteps)
    strainsb = np.linspace(bmin, bmax, args.bsteps)
    strainsc = np.linspace(cmin, cmax, args.csteps)

    strains = [[straina, strainb, strainc]
               for straina in strainsa
               for strainb in strainsb
               for strainc in strainsc]

    outputs = []
    for strn in strains:
        inputset_new = dataclasses.replace(inputset)  # copy
        inputset_new.structure =\
            structure0.apply_strain(strn, inplace=False)
        output_dir_suffix = (
            "strain" +
            (f".a.{strn[0]:.2f}" if strn[0] != 0 else "") +
            (f".b.{strn[1]:.2f}" if strn[1] != 0 else "") +
            f".c.{strn[2]:.2f}")
        outputs.append((inputset_new, output_dir_suffix))

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
    Return (inputset, output_dir_suffix)
    """
    relax_overrides = {
        "ISTART": 0,
        # Volume relaxation
        # 500 steps because 100 suggested in some online resources
        # may not be enough in complex supercells.
        "NSW": 500,
        "IBRION": Incar.IBRION_IONIC_RELAX_CGA,
        'ISIF': vars(Incar)["ISIF_" + args.isif],
        'EDIFF': 1e-06,
        'EDIFFG': -0.01
    }
    # 550eV recommended for _volume/shape_ relaxation During
    # volume/shape relaxation, initial automatic k-point grid
    # calculated for original volume becomes slightly less accurate
    # unless we increase ENCUT
    if relax_overrides['ISIF'] != Incar.ISIF_FIX_SHAPE_VOL:
        logger.info("Shape/volume relaxation.  Setting ENCUT=550.0")
        warnings.warn(
            "Shape/volume relaxation.  Setting ENCUT=550.0"
        )
        relax_overrides['ENCUT'] = 550.0
    else:
        logger.info("Shape and volume are fixed.  Setting ENCUT=500.0")
        warnings.warn(
            "Shape and volume are fixed.  Setting ENCUT=500.0"
        )
        relax_overrides['ENCUT'] = 500.0

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=relax_overrides,
    )
    output_dir_suffix = f"relax.{args.isif}"
    return (inputset, output_dir_suffix)


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
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_kpoints_settings={'grid_density': args.kpoint_density})
    # supercell: N1xN2xN3 string
    scaling = [int(x) for x in args.supercell_size.split("x")]
    inputset.structure.make_supercell(scaling)
    output_dir_suffix = args.supercell_size
    return (inputset, output_dir_suffix)


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
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        functional=args.functional_type)
    output_dir_suffix = args.functional_type
    return (inputset, output_dir_suffix)


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
    Return (inputset, output_dir_suffix)
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
    output_dir_suffix = ','.join(
        [f'{key}.{val}' for key, val in incar_overrides.items()])
    return (inputset, output_dir_suffix)


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
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_kpoints_settings={'grid_density': args.density},
    )
    output_dir_suffix = f"KPOINTS.{args.density}"
    return (inputset, output_dir_suffix)


def scf_add_args(parser):
    """Setup parser arguments for SCF calculation.
    Args:
      parser: subparser
    """
    parser.help = "Create input for SCF calculation"
    parser.set_defaults(func_derive=scf)


def scf(args):
    """Create SCF setup.
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings={'NSW': 0, 'IBRION': -1},
    )
    output_dir_suffix = "SCF"
    return (inputset, output_dir_suffix)


def insert_add_args(parser):
    """Setup parser arguments for inserting an atom/molecule.
    Args:
      parser: subparser
    """
    parser.help = \
        """Create possible structures with a given atom/molecule inserted.
When the original structure sets selective dynamics, the inserted
structure will not be constrained.
"""
    parser.set_defaults(func_derive=insert)

    parser.add_argument(
        "atom",
        help="Atom name to be inserted"
    )
    parser.add_argument(
        "--limit",
        help="Number of structures (negative to randomize search)",
        type=int)
    parser.add_argument(
        "--step",
        help="Scan step, ans",
        type=float)
    parser.add_argument(
        "--threshold",
        help=("Threshold multiplier for atom proximity"
              " (default: 0.75 [x atomic radii sum])"),
        type=float,
        default=0.75)
    parser.add_argument(
        "--no-matcher",
        dest="no_matcher",
        help=("do not compare the candidates by symmetry"
              "(will save generation time)"),
        action="store_true")
    parser.add_argument(
        "--count",
        help="do not write output, just print count",
        action="store_true")


def insert(args):
    """Create setup for inserted molecules/atoms.
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings={'NSW': 0, 'IBRION': -1},
    )

    if args.no_matcher:
        transformer = InsertMoleculeTransformation(
            args.atom,
            step=args.step,
            proximity_threshold=args.threshold,
            selective_dynamics=[True, True, True],
            matcher=None)
    else:
        transformer = InsertMoleculeTransformation(
            args.atom,
            step=args.step,
            proximity_threshold=args.threshold,
            selective_dynamics=[True, True, True])
    structures = transformer.all_inserts(inputset.structure, limit=args.limit)

    results = []
    if args.count:
        print(len(structures))
    else:
        for idx, structure in enumerate(structures):
            suffix = f"ins.{args.atom}.{idx}"
            inputset2 = dataclasses.replace(inputset)
            inputset2.structure = structure
            results.append((inputset2, suffix))

    return results


def delete_add_args(parser):
    """Setup parser arguments for deleting a site.
    Args:
      parser: subparser
    """
    parser.help = "Delete sites/atoms from structure"
    parser.set_defaults(func_derive=delete)
    parser.add_argument(
        "what",
        nargs="+",
        help="Specie names (e.g. Na) to be removed",
        type=str)


def delete(args):
    """Delete a site/sites from structure.
    Return (inputset, output_dir_suffix)
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)
    len_before = len(inputset.structure)
    inputset.structure.remove_species(args.what)
    if len(inputset.structure) == len_before:
        warnings.warn("Nothing was deleted")
    output_dir_suffix = ",".join(args.what)
    return (inputset, output_dir_suffix)


def derive(args):
    """Main routine.
    """
    value_or_values = args.func_derive(args)
    if isinstance(value_or_values, tuple):
        value_or_values = [value_or_values]

    output_dir_prefix = os.path.basename(
        os.path.abspath(args.input_directory))
    if args.output_prefix is not None:
        output_dir_prefix = args.output_prefix

    output_dir = args.output
    if args.output == "":
        raise ValueError("--output cannot be empty")

    for inputset, output_dir_suffix in value_or_values:
        if args.output is None:
            if output_dir_prefix:
                output_dir = output_dir_prefix + "." + output_dir_suffix
            else:
                output_dir = output_dir_suffix
        else:
            output_dir = args.output
        if args.subdir:
            output_dir = os.path.join(output_dir, args.subdir)
        inputset.write_input(output_dir=output_dir)

    return 0
