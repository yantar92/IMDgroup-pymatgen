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


"""Analysis extensions specific to IMD group.
Based on pymatgen's pymatgen.cli.pmg_analyze
"""
import logging
from pathlib import Path
from tabulate import tabulate
import pandas as pd
from alive_progress import alive_it
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.core.structure import structure_distance
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

logger = logging.getLogger(__name__)


ALL_FIELDS = {
    'dir': 'Directory', 'incar_group': 'INCAR type',
    'energy': 'Energy', 'e_per_atom': 'E/Atom',
    'total_mag': 'Magnetization',
    '%vol': '%vol', 'displ': 'displacement',
    'a': 'a', 'b': 'b', 'c': 'c',
    '%a': '%a', '%b': '%b', '%c': '%c',
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ',
    '%alpha': '%α', '%beta': '%β', '%gamma': '%γ'
}


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Analyze vasp outputs."""

    parser.add_argument(
        "dir",
        help="""Directory(ies) to read (recusrively); defaults to current dir""",
        type=str,
        nargs="*",
        default=["."])

    all_fields = [
        k for k in ALL_FIELDS
        if k not in ['dir', 'incar_group']]
    parser.add_argument(
        "--fields",
        help="""List of fields to report
energy: Final energy
e_per_atom: Final energy per atom
%%vol: Volume change before/after the run
displ: Total atom displacement / number of displaced atoms
a, b, c, alpha, beta, gamma: Lattice parameters
%%a, %%b, %%c, %%alpa, %%beta, %%gamma: Change before/after the run
""",
        nargs="+",
        choices=all_fields,
        default=all_fields
        )
    parser.add_argument(
        "--exclude-fields",
        dest='exclude_fields',
        help="""List of fields to NOT report""",
        nargs="+",
        choices=all_fields,
        default=[]
        )
    parser.add_argument(
        "--group",
        help="""Group by similar INCARs""",
        action="store_true"
        )


def read_field(field: str, vaspdir: IMDGVaspDir):
    """Read FIELD from VASPDIR.  Return the value read.
    """
    try:
        return _read_field_1(field, vaspdir)
    except FileNotFoundError:
        return "N/A"


def _read_field_1(field: str, vaspdir: IMDGVaspDir):
    """Read FIELD from VASPDIR.  Return the value read.
    May throw FileNotFound error if field cannot be derived
    from files in VASPDIR.
    """
    val = None

    if field == 'dir':
        val = Path(vaspdir.path).relative_to(Path.cwd())
        val = str(val).replace("./", "")
    elif field == 'energy':
        val = vaspdir.final_energy_reliable
        if isinstance(val, float):
            val = f"{vaspdir.final_energy_reliable:.5f}"
    elif field == 'e_per_atom':
        val = vaspdir.final_energy_reliable
        if not isinstance(val, str):
            val = val / len(vaspdir.structure)
            val = f"{val:.5f}"
    elif field == 'total_mag':
        val = vaspdir.total_magnetization
        if val is None:
            val = "None"
    elif field == '%vol':
        vol0 = vaspdir.initial_structure.volume
        val = vaspdir.structure.volume/vol0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'displ':
        try:
            displ = structure_distance(
                vaspdir.initial_structure,
                vaspdir.structure,
                match_first=True,
                norm=True)
        except ValueError:
            # structures are too different
            displ = structure_distance(
                vaspdir.initial_structure,
                vaspdir.structure,
                match_first=False,
                norm=True)
        val = f"{displ:.2f}"
    elif field == 'a':
        val = vaspdir.structure.lattice.a
    elif field == '%a':
        a0 = vaspdir.initial_structure.lattice.a
        val = vaspdir.structure.lattice.a/a0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'b':
        val = vaspdir.structure.lattice.b
    elif field == '%b':
        b0 = vaspdir.initial_structure.lattice.b
        val = vaspdir.structure.lattice.b/b0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'c':
        val = vaspdir.structure.lattice.c
    elif field == '%c':
        c0 = vaspdir.initial_structure.lattice.c
        val = vaspdir.structure.lattice.c/c0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'alpha':
        val = vaspdir.structure.lattice.alpha
    elif field == '%alpha':
        alpha0 = vaspdir.initial_structure.lattice.alpha
        val = vaspdir.structure.lattice.alpha/alpha0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'beta':
        val = vaspdir.structure.lattice.beta
    elif field == '%beta':
        beta0 = vaspdir.initial_structure.lattice.beta
        val = vaspdir.structure.lattice.beta/beta0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'gamma':
        val = vaspdir.structure.lattice.gamma
    elif field == '%gamma':
        gamma0 = vaspdir.initial_structure.lattice.gamma
        val = vaspdir.structure.lattice.gamma/gamma0 - 1
        val = f"{val * 100:.2f}"

    return val


def analyze(args):
    """Main routine.
    """

    vaspdirs = IMDGVaspDir.read_vaspdirs(args.dir)
    all_data = {field: [] for field in ALL_FIELDS}

    file_groups = {}
    if args.group:
        incars = []
        for _, vaspdir in vaspdirs.items():
            incar = vaspdir['INCAR']
            incar['SYSTEM'] = vaspdir.path
            incars.append(incar)
        _, groups = Incar.group_incars(incars)
        if len(groups) > 1:
            for idx, group in enumerate(groups):
                for incar in group:
                    file_groups[incar['SYSTEM'].upper()] = idx

    for _, vaspdir in alive_it(
            vaspdirs.items(), title="Reading VASP outputs"):
        for field in all_data:
            if field == 'incar_group' and args.group:
                val = file_groups[vaspdir.path.upper()]\
                    if len(file_groups) > 0 else 0
            else:
                val = read_field(field, vaspdir)
            all_data[field].append(val)

    if len(all_data) > 0 and len(vaspdirs) > 0:
        df = pd.DataFrame(all_data)
        if args.group:
            df = df.sort_values('incar_group')
        else:
            df = df.sort_values('dir')
        print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

    return 0
