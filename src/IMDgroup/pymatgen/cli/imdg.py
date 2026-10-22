#!/usr/bin/env python
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

"""Master CLI for VASP workflows and cluster job submission."""

import argparse
import importlib
import logging
import os
import sys
import warnings
from termcolor import colored
from IMDgroup.common import configure_progress_output, suppress_spglib_warnings

logger = logging.getLogger(__name__)


# Subcommand registry: name -> module, one-line help, and long description.
#
# Subcommand modules are imported lazily in :func:`main`.  Each one
# drags in heavy third-party dependencies (pymatgen, pandas,
# matplotlib, sklearn, gorun/...), and importing all seven at startup
# costs several seconds even when a single subcommand is used.
_SUBCOMMANDS = {
    "create": {
        "module": "IMDgroup.pymatgen.cli.imdg_create",
        "help": "Create new VASP inputs from scratch",
        "description": """\
Create new VASP inputs from scratch.

Accepts a Materials Project ID (e.g. mp-48), a path to a structure
file (CIF, POSCAR, etc.), or an atom name and cell dimensions
(e.g. "Li 20x20x20").  Writes a complete set of VASP input files
to a directory named after the system.""",
    },
    "derive": {
        "module": "IMDgroup.pymatgen.cli.imdg_derive",
        "help": "Derive new VASP inputs from an existing calculation",
        "description": """\
Derive new VASP inputs from an existing calculation.

Applies mutations -- relaxation, strain, supercell scaling, INCAR
changes, k-point modifications, atom insertion/deletion, functional
switching, and NEB path generation -- to a VASP directory and writes
the resulting input files.  Type "imdg derive -h" to see available
subcommands.""",
    },
    "diff": {
        "module": "IMDgroup.pymatgen.cli.imdg_diff",
        "help": "Compare VASP inputs and outputs across directories",
        "description": """\
Compare VASP inputs and outputs across multiple directories.

Supports structure comparison (grouping by symmetry and energy) and
INCAR comparison (grouping by parameter differences).  Type
"imdg diff -h" to see available subcommands.""",
    },
    "analyze": {
        "module": "IMDgroup.pymatgen.cli.imdg_analyze",
        "help": "Tabular summary of VASP outputs across directories",
        "description": """\
Produce a tabular summary of VASP outputs across directories.

Reports energies, lattice parameters, volume changes, atomic
displacements, and space groups.  Results can be grouped by INCAR
similarity and filtered by field selection.""",
    },
    "status": {
        "module": "IMDgroup.pymatgen.cli.imdg_status",
        "help": "Check running status of VASP calculations",
        "description": """\
Check the running status of VASP calculations and identify anomalies.

Scans directories recursively for VASP inputs and reports convergence
state, progress, wall-clock timing, SLURM job status, and warnings.
Supports filtering by problem status, regexp patterns, and warning
types.""",
    },
    "visualize": {
        "module": "IMDgroup.pymatgen.cli.imdg_visualize",
        "help": "Generate plots and visualizations from VASP outputs",
        "description": """\
Generate plots and visualizations from VASP outputs.

Includes NEB trajectory export, ATAT cluster expansion summaries,
formation energy convex hulls, voltage profiles, and selective
dynamics illustrations.  Type "imdg visualize -h" to see available
subcommands.""",
    },
    "run": {
        "module": "IMDgroup.pymatgen.cli.imdg_run",
        "help": "Submit/run VASP, MACE, ATAT, or GPU jobs",
        "description": """\
Submit or run calculations through the gorun job-submission layer.

Forwards all remaining arguments to ``gorun``, so ``imdg run --help``
and ``imdg run <subcommand> --help`` behave exactly like the ``gorun*``
entry points.  Subcommands: vasp, mace-finetune, maps, atat-local,
gpu.""",
    },
}


def _user_error_types() -> tuple[type[BaseException], ...]:
    """Exception types reported as one-line user errors in :func:`main`.

    ``MPRestError`` is imported lazily: importing it drags in the whole
    ``pymatgen.ext.matproj`` module (requests, pandas, ...), which adds
    several seconds to the startup of every subcommand.  Only ``create``
    and ``derive`` can raise it, so it is loaded here on the failure path
    instead of at module import time.
    """
    from pymatgen.ext.matproj import MPRestError
    return (ValueError, OSError, MPRestError)


def _showwarning(message, category, _filename, _lineno, file=None, _line=None):
    """Display warnings with coloured output."""
    _ = (_filename, _lineno, _line)  # suppress info-level "not accessed"
    output = colored(
        f"{category.__name__}: ", "yellow", attrs=['bold']) +\
        f"{message}"
    logger.warning("%s", message)
    print(output, file=file or sys.stderr)


warnings.showwarning = _showwarning


def setup_logger(args):
    """Configure logging based on verbosity flags.

    Args:
        args: Parsed command-line arguments.
    """
    log_file = os.path.join("imdg.log")

    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    log_format = '[%(levelname)s] [%(asctime)s] %(message)s'
    date_format = "%b %d %X"

    if args.verbosity >= 2:
        logging.basicConfig(
            format=log_format, datefmt=date_format,
            level=logging.DEBUG, handlers=[file_handler, stdout_handler])
        logging.info("Setting debug level to: DEBUG")
    elif args.verbosity >= 1:
        logging.basicConfig(
            format=log_format, datefmt=date_format,
            level=logging.INFO, handlers=[file_handler, stdout_handler])
        logging.info("Setting debug level to: INFO")
    else:
        logging.basicConfig(
            format=log_format, datefmt=date_format,
            level=logging.INFO, handlers=[file_handler])
        logging.info("Setting debug level to: INFO (writing to file)")


def main():
    """Entry point for the ``imdg`` command.

    Returns:
        int: Exit code from the selected subcommand.
    """
    configure_progress_output()
    suppress_spglib_warnings()

    # ``imdg run ...`` forwards to gorun's own parser, which is the single
    # source of truth for the run subcommands (vasp, mace-finetune, maps,
    # atat-local, gpu).  Intercepting before argparse ensures ``--help``
    # and every gorun flag behave exactly like the ``gorun*`` entry points.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from IMDgroup.gorun.gorun import main as gorun_main
        return gorun_main(sys.argv[2:], prog="imdg run")

    parser = argparse.ArgumentParser(
        description="""
        imdg is a script to generate, analyze, and run VASP
        calculations.  The script is similar to pymatgen's pmg, but
        implements workflows used in the Inverse Material Design Group,
        including cluster job submission via "imdg run".

        imdg supports several subcommands.  Type "imdg
        sub-command -h" to get help for individual sub-commands.
        """,
        epilog="Author: Ihor Radchenko"
    )
    parser.add_argument(
        "-v", "--verbosity", action="count",
        help="log verbosity as -v or -vv (default: no logging)",
        default=0)

    subparsers = parser.add_subparsers(required=True, dest="command")

    # Register every subcommand with its name/help/description so that
    # ``imdg --help`` and ``imdg -h`` list them all, but do not import the
    # subcommand modules yet.  ``add_help=False`` keeps a subparser's
    # built-in ``-h`` from firing on the probe parse below, before its
    # subcommand-specific flags have been registered.
    subparser_by_name = {}
    for name, spec in _SUBCOMMANDS.items():
        subparser_by_name[name] = subparsers.add_parser(
            name,
            help=spec["help"],
            description=spec["description"],
            add_help=False)

    # Resolve the requested subcommand without importing any subcommand
    # module.  All subcommand-specific flags are unknown on this pass and
    # end up in ``_unknown``, which we ignore here.
    args, _unknown = parser.parse_known_args()
    command = args.command

    # Import only the requested subcommand module and finish wiring its
    # subparser.  The entry function is named after the subcommand
    # (create -> imdg_create.create, status -> imdg_status.status, ...).
    module = importlib.import_module(_SUBCOMMANDS[command]["module"])
    subparser = subparser_by_name[command]
    module.add_args(subparser)
    subparser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS,
        help="show this help message and exit")
    subparser.set_defaults(func=getattr(module, command))

    args = parser.parse_args()

    setup_logger(args)
    logger.info(
        "Running from %s with the following args: %s",
        os.getcwd(),
        args
    )

    try:
        return args.func(args)
    except _user_error_types() as exc:
        # User-facing failures: report a one-line message instead of a
        # traceback.  ``-vv`` re-raises so the full traceback is shown.
        if args.verbosity >= 2:
            raise
        print(colored(f"error: {exc}", "red"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
