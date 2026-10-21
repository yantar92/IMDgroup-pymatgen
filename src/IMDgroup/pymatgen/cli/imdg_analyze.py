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
Based on pymatgen's pymatgen.cli.pmg_analyze."""
import logging
import re
import sys
from pathlib import Path
from tabulate import tabulate
import numpy as np
import pandas as pd
from alive_progress import alive_it
from rich.console import Console
from rich.table import Table
from pymatgen.core import Structure
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.core.structure import structure_distance
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

logger = logging.getLogger(__name__)


ALL_FIELDS = {
    'dir': 'Directory', 'incar_group': 'INCAR type',
    'energy': 'Energy', 'e_per_atom': 'E/Atom',
    'total_mag': 'Magnetization',
    'max_force': 'Max force',
    '%vol': '%vol', 'displ': 'displacement',
    'space_group': 'Space group',
    'a': 'a', 'b': 'b', 'c': 'c',
    '%a': '%a', '%b': '%b', '%c': '%c',
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ',
    '%alpha': '%α', '%beta': '%β', '%gamma': '%γ'
}

# Fields shown by default in the human-readable table.  Machine formats
# (csv/json/orgtbl) default to every field for full fidelity.
DEFAULT_FIELDS = [
    'energy', 'e_per_atom', 'max_force', '%vol', 'displ', 'space_group',
]


def add_args(parser):
    """Register subcommand arguments.

    Args:
        parser: Sub-parser from argparse.
    """
    parser.add_argument(
        "dir",
        help="""Directory(ies) to read (recusrively); defaults to current dir""",
        type=str,
        nargs="*",
        default=["."])
    parser.add_argument(
        "--exclude",
        help="Dirs matching this Python regexp pattern will be excluded",
        type=str,
    )

    all_fields = [
        k for k in ALL_FIELDS
        if k not in ['dir', 'incar_group']]
    parser.add_argument(
        "--fields",
        help="""List of fields to report.  Use ALL to show every field.
When omitted, the table shows a short default set; machine formats
(--format csv/json/orgtbl) show all fields.
energy: Final energy
e_per_atom: Final energy per atom
max_force: Maximum atomic force in the final step (eV/Å)
%%vol: Volume change before/after the run
displ: Total atom displacement / number of displaced atoms
space_group, a, b, c, alpha, beta, gamma: Lattice parameters
%%a, %%b, %%c, %%alpha, %%beta, %%gamma: Change before/after the run
""",
        nargs="+",
        choices=all_fields + ['ALL'],
        default=None
    )
    parser.add_argument(
        "--format",
        dest="format",
        help="""Output format.  'table' is the human-readable rich table
(default); 'csv', 'json', and 'orgtbl' bypass rich for programmatic use.""",
        choices=["table", "csv", "json", "orgtbl"],
        default="table"
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
        "--nogroup",
        help="""Do not group by similar INCARs""",
        action="store_true"
    )


def read_field(field: str, vaspdir: IMDGVaspDir):
    """Read a single analysis field from a VASP directory.

    Args:
        field: Field name (see ``ALL_FIELDS`` for valid values).
        vaspdir: VASP directory wrapper.

    Returns:
        Field value, or ``"N/A"`` when data is unavailable.
    """
    try:
        return _read_field_1(field, vaspdir)
    except (FileNotFoundError, TypeError, AttributeError):
        return "N/A"


def _require_structure(structure: Structure | None) -> Structure:
    """Return *structure*, raising ``TypeError`` when it is ``None``.

    Args:
        structure: Possibly missing structure.

    Returns:
        Structure: The non-``None`` structure.

    Raises:
        TypeError: If *structure* is ``None``.  This is caught by
            :func:`read_field`.
    """
    if structure is None:
        raise TypeError("structure is None")
    return structure


def _read_field_1(field: str, vaspdir: IMDGVaspDir):
    """Read a single analysis field (internal, may raise).

    Args:
        field: Field name.
        vaspdir: VASP directory wrapper.

    Returns:
        Field value.

    Raises:
        FileNotFoundError: If the required data files are missing.
        TypeError: If the required structure is missing
            (``vaspdir.final_structure`` / ``vaspdir.initial_structure`` is
            ``None``).  Caught by :func:`read_field`.
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
            val = val / len(_require_structure(vaspdir.final_structure))
            val = f"{val:.5f}"
    elif field == 'total_mag':
        val = vaspdir.total_magnetization
        if val is None:
            val = "None"
    elif field == 'max_force':
        run = vaspdir['vasprun.xml']
        if run is None:
            val = "None"
        else:
            forces = run.ionic_steps[-1].get('forces')
            if forces is None:
                val = "None"
            else:
                val = max(np.linalg.norm(force) for force in forces)
                val = f"{val:.4f}"
    elif field == '%vol':
        vol0 = _require_structure(vaspdir.initial_structure).volume
        val = _require_structure(vaspdir.final_structure).volume / vol0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'displ':
        initial = _require_structure(vaspdir.initial_structure)
        final = _require_structure(vaspdir.final_structure)
        try:
            displ = structure_distance(
                initial,
                final,
                match_first=True,
                norm=True)
        except ValueError:
            # structures are too different
            displ = structure_distance(
                initial,
                final,
                match_first=False,
                norm=True)
        val = f"{displ:.2f}"
    elif field == 'space_group':
        structure = _require_structure(vaspdir.final_structure)
        val = structure.get_space_group_info()[0]
    elif field == 'a':
        val = _require_structure(vaspdir.final_structure).lattice.a
    elif field == '%a':
        a0 = _require_structure(vaspdir.initial_structure).lattice.a
        val = _require_structure(vaspdir.final_structure).lattice.a / a0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'b':
        val = _require_structure(vaspdir.final_structure).lattice.b
    elif field == '%b':
        b0 = _require_structure(vaspdir.initial_structure).lattice.b
        val = _require_structure(vaspdir.final_structure).lattice.b / b0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'c':
        val = _require_structure(vaspdir.final_structure).lattice.c
    elif field == '%c':
        c0 = _require_structure(vaspdir.initial_structure).lattice.c
        val = _require_structure(vaspdir.final_structure).lattice.c / c0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'alpha':
        val = _require_structure(vaspdir.final_structure).lattice.alpha
    elif field == '%alpha':
        alpha0 = _require_structure(vaspdir.initial_structure).lattice.alpha
        val = _require_structure(vaspdir.final_structure).lattice.alpha / alpha0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'beta':
        val = _require_structure(vaspdir.final_structure).lattice.beta
    elif field == '%beta':
        beta0 = _require_structure(vaspdir.initial_structure).lattice.beta
        val = _require_structure(vaspdir.final_structure).lattice.beta / beta0 - 1
        val = f"{val * 100:.2f}"
    elif field == 'gamma':
        val = _require_structure(vaspdir.final_structure).lattice.gamma
    elif field == '%gamma':
        gamma0 = _require_structure(vaspdir.initial_structure).lattice.gamma
        val = _require_structure(vaspdir.final_structure).lattice.gamma / gamma0 - 1
        val = f"{val * 100:.2f}"

    return val


def _render_table(df, console=None):
    """Render a pandas DataFrame as a rich table.

    Args:
        df: DataFrame whose columns are field keys (plus ``dir`` and,
            when grouping is enabled, ``incar_group``).
        console: rich Console to print to (defaults to a new one).
    """
    if console is None:
        console = Console()
    table = Table(box=None, show_header=True, header_style="bold",
                  padding=(0, 1))
    for col in df.columns:
        label = ALL_FIELDS.get(col, col)
        if col == 'dir':
            table.add_column(label, overflow="ellipsis", max_width=40)
        elif col == 'incar_group':
            table.add_column(label, justify="right", no_wrap=True)
        else:
            table.add_column(label, justify="right", no_wrap=True)
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)


def analyze(args):
    """Run the analysis subcommand.

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        int: Exit code (0 on success).
    """
    def include_dirp(p):
        p = str(p)
        if args.exclude is not None and re.search(args.exclude, p):
            return False
        return True

    vaspdirs = IMDGVaspDir.read_vaspdirs(args.dir, path_filter=include_dirp)

    all_fields = [k for k in ALL_FIELDS if k not in ('dir', 'incar_group')]
    if args.fields is not None:
        base_fields = all_fields if 'ALL' in args.fields else list(args.fields)
    elif args.format == 'table':
        base_fields = list(DEFAULT_FIELDS)
    else:
        base_fields = all_fields
    selected_fields = [f for f in base_fields if f not in args.exclude_fields]

    all_data = {}
    all_data['dir'] = []
    if not args.nogroup:
        all_data['incar_group'] = []
    for field in selected_fields:
        all_data[field] = []

    file_groups = {}
    if not args.nogroup:
        incars = []
        for _, vaspdir in vaspdirs.items():
            incar = vaspdir['INCAR']
            if incar is None:
                incar = Incar()
            incar['SYSTEM'] = vaspdir.path
            incars.append(incar)
        _, groups = Incar.group_incars(incars)
        if len(groups) > 1:
            for idx, group in enumerate(groups):
                for incar in group:
                    file_groups[incar['SYSTEM'].upper()] = idx

    len_vaspdirs = len(vaspdirs)
    for path, vaspdir in alive_it(
            list(vaspdirs.items()), title="Reading VASP outputs"):
        for field in all_data:
            if field == 'incar_group' and (not args.nogroup):
                val = file_groups[vaspdir.path.upper()]\
                    if len(file_groups) > 0 else 0
            else:
                val = read_field(field, vaspdir)
            all_data[field].append(val)
        # Avoid using too much memory by holding all vaspdir objects
        del vaspdirs[path]

    if len(all_data) > 0 and len_vaspdirs > 0:
        df = pd.DataFrame(all_data)
        if not args.nogroup:
            df = df.sort_values(by=['incar_group', 'dir'])
        else:
            df = df.sort_values('dir')
        if args.format == 'table':
            _render_table(df)
        elif args.format == 'csv':
            df.to_csv(sys.stdout, index=False)
        elif args.format == 'json':
            sys.stdout.write(df.to_json(orient='records', indent=2) + '\n')
        else:  # orgtbl
            print(tabulate(
                df, headers=df.columns.to_list(),
                tablefmt="orgtbl", showindex=False))

    return 0
