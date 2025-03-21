"""imdg sub-command to create new VASP inputs from existing.
"""
import os
import re
import warnings
import argparse
import dataclasses
import logging
from multiprocessing import Pool
import numpy as np
import pymatgen.core as pmg
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Structure, PeriodicSite
from IMDgroup.pymatgen.core.structure import\
    get_matched_structure, merge_structures, structure_diff, structure_is_valid2
from IMDgroup.pymatgen.diffusion.neb import get_neb_pairs
from IMDgroup.pymatgen.io.vasp.sets\
    import (IMDDerivedInputSet, IMDNEBVaspInputSet)
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.transformations.insert_molecule\
    import InsertMoleculeTransformation

logger = logging.getLogger(__name__)


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Create new VASP inputs from existing.
Write them to <prefix><output name><subdir>."""

    parser.add_argument(
        "input_directory",
        default=".",
        help="VASP directory to read system"
    )

    parser.add_argument(
        "--overwrite_output",
        help="Whether to overwrite non-empty output directories"
        "(default: True).",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--output",
        help="Directory to write the mutated VASP input"
        "(default: <input name>.<mutated name>)."
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

    parser_insert = subparsers.add_parser("insert")
    insert_add_args(parser_insert)
    parser_insert2 = subparsers.add_parser("ins")
    insert_add_args(parser_insert2)

    parser_delete = subparsers.add_parser("del")
    delete_add_args(parser_delete)

    parser_neb = subparsers.add_parser("neb")
    neb_add_args(parser_neb)

    parser_neb_diffusion = subparsers.add_parser("neb_diffusion")
    neb_diffusion_add_args(parser_neb_diffusion)


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
    Return {'inputsets': [<inputset>]}
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)

    inputset.structure.perturb(args.distance)
    output_dir_suffix = f"PERTURB.{args.distance}"
    inputset.name = output_dir_suffix

    return {'inputsets': [inputset]}


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
    Return {'inputsets': <list of inputsets>}
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
                new_val = float(re.sub("%", "", value))/100.0 - 1.0
            else:
                new_val = float(value)/getattr(structure0.lattice, name) - 1.0
            logger.info("%s: %s -> %f", attr_name, value, new_val)
            setattr(args, attr_name, new_val)

    strainsa = np.linspace(args.amin, args.amax, args.asteps)
    strainsb = np.linspace(args.bmin, args.bmax, args.bsteps)
    strainsc = np.linspace(args.cmin, args.cmax, args.csteps)

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
        inputset_new.name = output_dir_suffix
        outputs.append(inputset_new)

    return {'inputsets': outputs}


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
    parser.add_argument(
        "auto_encut",
        help="Set ENCUT automatically",
        action="store_true"
    )


def relax(args):
    """Create relaxation setup.
    Return {'inputsets': [inputset]}
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
    if args.auto_encut:
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
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


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
    Return {'inputsets': [<inputset>]}
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_kpoints_settings={'grid_density': args.kpoint_density})
    # supercell: N1xN2xN3 string
    scaling = [int(x) for x in args.supercell_size.split("x")]
    inputset.structure.make_supercell(scaling)
    output_dir_suffix = args.supercell_size
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


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
    Return {'inputsets': [inputset]}
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        functional=args.functional_type)
    output_dir_suffix = args.functional_type
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


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
        help="PARAM:VALUE to be set in the INCAR. (VALUE=None to unset)",
        type=str)


def incar(args):
    """Create custom incar setup.
    Return {'inputsets': [inputset]}
    """
    incar_overrides = {}
    if args.parameters is None:
        warnings.warn(
            "No INCAR settings provided.  Creating a copy of the inputs."
        )
    else:
        for str_val in args.parameters:
            key, val = str_val.split(":")
            if val == 'None':
                val = None
            incar_overrides[key] = val

    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings=incar_overrides,
    )
    output_dir_suffix = ','.join(
        [f'{key}.{val}' for key, val in incar_overrides.items()])
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


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
    Return {'inputsets': [inputset]}
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_kpoints_settings={'grid_density': args.density},
    )
    output_dir_suffix = f"KPOINTS.{args.density}"
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


def scf_add_args(parser):
    """Setup parser arguments for SCF calculation.
    Args:
      parser: subparser
    """
    parser.help = "Create input for SCF calculation"
    parser.set_defaults(func_derive=scf)


def scf(args):
    """Create SCF setup.
    Return {'inputsets': [inputset]}
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings={'NSW': 0, 'IBRION': -1},
    )
    output_dir_suffix = "SCF"
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


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
        help="Atom name to be inserted or path to molecule structure"
    )
    parser.add_argument(
        "--limit",
        help="""Number of structures (negative to select subset of structures randomly)""",
        type=int)
    parser.add_argument(
        "--step",
        help="Scan step, ans",
        type=float)
    parser.add_argument(
        "--step_noise",
        help="""Standard deviation of noise added to each point in the scan grid (default: None)
When negative number, use random sampling instead of scanning a grid""",
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
    parser.add_argument(
        "--multithread",
        help="Use multithreading?",
        action="store_true")


def insert(args):
    """Create setup for inserted molecules/atoms.
    Return {'inputsets': [<list of inputsets>]}
    """
    inputset = IMDDerivedInputSet(
        directory=args.input_directory,
        user_incar_settings={'NSW': 0, 'IBRION': -1},
    )

    if args.no_matcher:
        transformer = InsertMoleculeTransformation(
            args.atom,
            step=args.step,
            step_noise=args.spet_noise,
            proximity_threshold=args.threshold,
            selective_dynamics=[True, True, True],
            matcher=None,
            multithread=args.multithread,
        )
    else:
        transformer = InsertMoleculeTransformation(
            args.atom,
            step=args.step,
            step_noise=args.spet_noise,
            proximity_threshold=args.threshold,
            selective_dynamics=[True, True, True],
            multithread=args.multithread,
        )
    structures = transformer.all_inserts(inputset.structure, limit=args.limit)

    results = []
    if args.count:
        print(len(structures))
    else:
        for idx, structure in enumerate(structures):
            suffix = f"ins.{args.atom}.{idx}"
            inputset2 = dataclasses.replace(inputset)
            inputset2.structure = structure
            inputset2.name = suffix
            results.append(inputset2)

    return {'inputsets': results}


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
    Return {'inputsets': [inputset]}
    """
    inputset = IMDDerivedInputSet(directory=args.input_directory)
    len_before = len(inputset.structure)
    inputset.structure.remove_species(args.what)
    if len(inputset.structure) == len_before:
        warnings.warn("Nothing was deleted")
    output_dir_suffix = ",".join(args.what)
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


def neb_add_args(parser):
    """Setup parser arguments for NEB input.
    Args:
      parser: subparser
    """
    parser.help = "Create NEB input between two VASP runs"
    parser.set_defaults(func_derive=neb)
    parser.add_argument(
        "target",
        help="VASP output dir containing the target NEB point",
        type=str)
    parser.add_argument(
        "--nimages",
        help="Number of NEB images (default: 4)",
        type=int,
        default=4)


def neb(args):
    """Create NEB input.
    Return {'inputsets': [inputset]}
    """
    inputset = IMDNEBVaspInputSet(
        directory=args.input_directory,
        target_directory=args.target,
        user_incar_settings={'IMAGES': args.nimages})
    output_dir_suffix = "NEB"
    inputset.name = output_dir_suffix
    return {'inputsets': [inputset]}


def neb_diffusion_add_args(parser):
    """Setup parser arguments for diffusion NEB input.
    Args:
      parser: subparser
    """
    parser.help =\
        "Create NEB input between multiple VASP runs derived from prototype"
    parser.set_defaults(func_derive=neb_diffusion)
    parser.add_argument(
        "--diffusion_points",
        help="VASP output dirs containing the stable, "
        "converged diffusion sites",
        required=True,
        nargs="+",
        type=str)
    parser.add_argument(
        "--nimages",
        help="Number of NEB images (default: 4)",
        type=int,
        default=4)
    parser.add_argument(
        "--cutoff",
        help="Distance cutoff between diffusion points (float or 'auto' to determine automatically).",
        type=str,
        default='auto')
    parser.add_argument(
        "--fix_dist",
        help="Atoms further than fix_dist ans away will not be allowed to move (default: -1; no restrictions)",
        type=float,
        default=-1)
    parser.add_argument(
        "--path_method",
        help="Interpolation method: 'linear' or 'IDPP' (default)",
        type=str,
        default='IDPP')
    parser.add_argument(
        "--frac_tol",
        help="when path_method='linear', NEB images with atoms less than frac_tol * sum of radiuses will not be generated",
        type=float,
        default=0.75)
    parser.add_argument(
        "--use_prototype_matrix",
        help="Before processing, alter diffusion points to use prototype matrix"
        " (this will build diffusion points by inserting added atoms into unchanged prototype structure)",
        action="store_true",
    )
    parser.add_argument(
        "--remove_compound",
        help="Filter out diffusion paths that may be constructed out of shorter paths",
        action="store_true",
    )
    parser.add_argument(
        "--multithread",
        help="Use multithreading?",
        action="store_true")
    parser.add_argument(
        "--write_graph",
        help="Whether to write the full diffusion graph",
        action="store_true")


def __neb_diffusion_get_inputset(idx, beg, end, args, auto_nimages=False):
    if auto_nimages:
        # Dynamically compute NIMAGES for best visualization
        diff = structure_diff(beg, end, tol=0, match_first=False)
        max_disp = np.max(np.linalg.norm(diff, axis=1))
        nimages = max(args.nimages, int(np.ceil(max_disp/0.5)))
        frac_tol = 0
        method = 'linear'
    else:
        nimages = args.nimages
        frac_tol = args.frac_tol
        method = args.path_method
    inputset = IMDNEBVaspInputSet(
        directory=beg.properties['origin_path'],
        target_directory=end.properties['origin_path'],
        fix_cutoff=args.fix_dist if args.fix_dist > 0 else None,
        method=method,
        frac_tol=frac_tol,
        user_incar_settings={'IMAGES': nimages})
    # Disable period boundaries and structure adjustment to force
    # diffusion path as is.
    inputset.update_images(
        beg, end,
        pbc=False, center=False, match_first=False)
    output_dir_suffix = f"NEB.{idx:02}"
    inputset.name = output_dir_suffix
    return inputset


def __neb_diffusion_get_inputsets(pairs, args, auto_nimages=False):
    if args.multithread:
        with Pool() as pool:
            result = pool.starmap(
                __neb_diffusion_get_inputset,
                [(idx, beg, end, args, auto_nimages)
                 for idx, (beg, end) in enumerate(pairs)]
            )
    else:
        result = []
        for idx, (beg, end) in enumerate(pairs):
            inputset = __neb_diffusion_get_inputset(
                idx, beg, end, args, auto_nimages)
            result.append(inputset)
    return result


def _append_valid(
        site: PeriodicSite,
        structure: Structure,
        frac_tol: float):
    """Append SITE to STRUCTURE without invalidating it.
    If SITE can be appended without breaking structure.is_valid(),
    just append it, modifying STRUCTRE by side effect.  If not, try
    moving SITE, so that all distances to existing STRUCTURE sites are
    not too small.
    FRAC_TOL is distance tolerance, in fraction of atomic radii sum.
    """
    structure.append(
        site.species, site.coords,
        coords_are_cartesian=True
    )
    if structure_is_valid2(structure, frac_tol):
        return structure

    warnings.warn(
        "Added atoms clash with prototype.  Trying to adjust"
    )

    while not structure_is_valid2(structure, frac_tol):
        site = structure[-1]
        _, _, neighbor_indices, _ =\
            structure.lattice.get_points_in_sphere(
                frac_points=structure.frac_coords,
                center=site.coords,  # Cartesian
                r=5, zip_results=False
            )
        close_idx = neighbor_indices[0]
        if structure[close_idx] == site:
            close_idx = neighbor_indices[1]
        close = structure[close_idx]
        frac_vec = 0.1 * (site.frac_coords - close.frac_coords)
        structure.translate_sites(
            [len(structure) - 1],
            frac_vec, frac_coords=True
            )
    return structure


def neb_diffusion(args):
    """Create NEB input for all possible diffusion paths.
    Return {'inputsets': <list of inputsets>}
    """

    if args.cutoff != 'auto' and args.cutoff:
        args.cutoff = float(args.cutoff)

    logger.info("Reading prototype from %s", args.input_directory)
    if os.path.isdir(args.input_directory):
        prototype_run = Vasprun(os.path.join(
            args.input_directory, 'vasprun.xml'))
        prototype = prototype_run.final_structure
    else:
        # Try to load structure from file
        prototype = pmg.Structure.from_file(args.input_directory)

    structures = []
    for struct_path in args.diffusion_points:
        logger.info("Reading structure from %s", struct_path)
        structure_run = Vasprun(os.path.join(struct_path, 'vasprun.xml'))
        assert structure_run.converged
        structure = structure_run.final_structure
        structure.properties['origin_path'] = struct_path
        structure.properties['final_energy'] = structure_run.final_energy
        structures.append(structure)

    if args.use_prototype_matrix:
        logger.info("Building new diffusion points as prototype+added atoms")
        for idx, struct in enumerate(structures):
            structures[idx] =\
                get_matched_structure(prototype, struct)
        if len(structures[0]) > len(prototype):
            idxs = list(range(len(prototype), len(structures[0])))
            logger.debug(
                "Found inserted sites: %s",
                [structures[0][idx] for idx in idxs]
            )
            for struct in structures:
                new_sites = [struct[idx] for idx in idxs]
                struct.remove_sites(list(range(len(struct))))
                for site in prototype:
                    struct.append(
                        species=site.species,
                        coords=site.coords,
                        coords_are_cartesian=True
                        )
                for site in new_sites:
                    _append_valid(site, struct, args.frac_tol)

    pairs, unfiltered_pairs = get_neb_pairs(
        structures, prototype, args.cutoff, args.remove_compound,
        multithread=args.multithread, return_unfiltered=True)

    result = __neb_diffusion_get_inputsets(pairs, args)
    if args.write_graph:
        graph_file = "imdg-full-graph.cif"
        logger.info("Writing diffusion graph summary to %s", graph_file)
        with warnings.catch_warnings(action="ignore"):
            graph_inputs = __neb_diffusion_get_inputsets(
                unfiltered_pairs, args, auto_nimages=True)
        graph_images = []
        for inputset in graph_inputs:
            assert inputset.images is not None
            for image in inputset.images:
                graph_images.append(image.structure)
        graph_combined = merge_structures(graph_images, tol=0.1)
        graph_combined.to_file(graph_file)

    return {'inputsets': result}


def derive(args):
    """Main routine.
    """
    data = args.func_derive(args)
    inputsets = data['inputsets']

    output_dir_prefix = os.path.basename(
        os.path.abspath(args.input_directory)) + "."
    if args.output_prefix is not None:
        output_dir_prefix = args.output_prefix

    output_dir = args.output
    if args.output == "":
        raise ValueError("--output cannot be empty")

    for inputset in inputsets:
        if output_dir_prefix:
            output_dir = output_dir_prefix + inputset.name
        else:
            output_dir = inputset.name
        if args.output:
            if len(inputsets) == 1:
                output_dir = args.output
            else:
                output_dir = os.path.join(args.output, output_dir)
        if args.subdir:
            output_dir = os.path.join(output_dir, args.subdir)
        write_input = True
        if os.path.isdir(output_dir) and not os.listdir(output_dir):
            if args.overwrite_output:
                warnings.warn(f"Overwriting non-empty dir: {output_dir}")
            else:
                warnings.warn(f"Skipping non-empty dir: {output_dir}")
                write_input = False
        if write_input:
            logger.info("Writing inputset to %s", output_dir)
            inputset.write_input(output_dir=output_dir)

    return 0
