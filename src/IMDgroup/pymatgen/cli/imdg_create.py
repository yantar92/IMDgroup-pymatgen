# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-pymatgen package
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""imdg sub-command to create new VASP inputs from scratch.
"""
import os
import re
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

    parser.add_argument(
        "what",
        help="""What to create (atom name+cell, mp-XXX, or file)
- Atom name+cell is atom name and cell size (in ans).
  Example: Li 20x20x20
- mp-XXX is material project id of the structure to be fetched
  Example: mp-48
- file is a path to structure file that can be read by pymatgen
  Example: ./structure.cif
""",
        type=str)


def create_from_mpid(mpid):
    """Return structure from materials project id.
    Structure will have property 'mpid' set to its id."""
    structure = pmg.Structure.from_id(mpid)
    structure.properties['mpid'] = mpid
    logger.info(
        "Downloaded structure %s from Materials Project db:\n%s",
        mpid, structure
    )
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

    return structure


def create_from_file(path):
    """Return structure created from file.
    """
    structure = pmg.Structure.from_file(path)
    logger.info(
        "Loaded structure from %s:\n%s",
        path, structure
    )
    return structure


def create_from_atom_name(name, size):
    """Return periodic structure with atom NAME in the middle.
    Structure dimentions are defined in SIZE vector.
    """
    logger.info(
        "Creating %fx%fx%f supercell for %s",
        size[0], size[1], size[2], name)
    molecule = pmg.Molecule([name], [[0, 0, 0]])
    return molecule.get_boxed_structure(size[0], size[1], size[2])


def create(args):
    """Main routine.
    """
    if os.path.isfile(args.what):
        structure = create_from_file(args.what)
        inputset = IMDStandardVaspInputSet(structure=structure)
    elif match := re.match(
            r'([A-Z][a-z]) +([0-9]+)x([0-9]+)x([0-9]+)',
            args.what):
        structure = create_from_atom_name(
            match[1],
            [float(match[2]), float(match[3]), float(match[4])])
        inputset = IMDStandardVaspInputSet(
            structure=structure,
            user_incar_settings={'SYSTEM': f'{match[1]}.boxed'},
            # Single k-point
            user_kpoints_settings={'length': 1})
    else:
        structure = create_from_mpid(args.what)
        inputset = IMDStandardVaspInputSet(structure=structure)
    inputset.write_input(output_dir=inputset.incar['SYSTEM'])

    return 0
