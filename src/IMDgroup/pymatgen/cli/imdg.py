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


#!/usr/bin/env python

"""Master script to work with VASP inputs and outputs."""

import argparse
import logging
import os
import sys
import warnings
from termcolor import colored
import IMDgroup.pymatgen.cli.imdg_create
import IMDgroup.pymatgen.cli.imdg_derive
import IMDgroup.pymatgen.cli.imdg_diff
import IMDgroup.pymatgen.cli.imdg_analyze
import IMDgroup.pymatgen.cli.imdg_status
import IMDgroup.pymatgen.cli.imdg_visualize

logger = logging.getLogger(__name__)


def _showwarning(message, category, _filename, _lineno, file=None, _line=None):
    """Print warning in nicer way."""
    output = colored(
        f"{category.__name__}: ", "yellow", attrs=['bold']) +\
        f"{message}"
    logger.warning("%s", message)
    print(output, file=file or sys.stderr)


warnings.showwarning = _showwarning


def setup_logger(args):
    """Setup logging according to command line args."""
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
    """Main routine."""
    parser = argparse.ArgumentParser(
        description="""
        imdg is a script to generate and analyze VASP inputs/outputs.
        The script is similar to pymatgen's pmg, but implements
        workflows used in the Inverse Material Design Group.

        The scripts supports several subcommands.  Type "imdg
        sub-command -h" to get help for individual sub-commands.
        """,
        epilog="Author: Ihor Radchenko"
    )
    parser.add_argument(
        "-v", "--verbosity", action="count",
        help="log verbosity as -v or -vv (default: no logging)",
        default=0)

    subparsers = parser.add_subparsers(required=True)

    parser_create = subparsers.add_parser("create")
    IMDgroup.pymatgen.cli.imdg_create.add_args(parser_create)
    parser_create.set_defaults(func=IMDgroup.pymatgen.cli.imdg_create.create)

    parser_derive = subparsers.add_parser("derive")
    IMDgroup.pymatgen.cli.imdg_derive.add_args(parser_derive)
    parser_derive.set_defaults(func=IMDgroup.pymatgen.cli.imdg_derive.derive)

    parser_diff = subparsers.add_parser("diff")
    IMDgroup.pymatgen.cli.imdg_diff.add_args(parser_diff)
    parser_diff.set_defaults(func=IMDgroup.pymatgen.cli.imdg_diff.diff)

    parser_analyze = subparsers.add_parser("analyze")
    IMDgroup.pymatgen.cli.imdg_analyze.add_args(parser_analyze)
    parser_analyze.set_defaults(
        func=IMDgroup.pymatgen.cli.imdg_analyze.analyze)

    parser_status = subparsers.add_parser("status")
    IMDgroup.pymatgen.cli.imdg_status.add_args(parser_status)
    parser_status.set_defaults(
        func=IMDgroup.pymatgen.cli.imdg_status.status)

    parser_visualize = subparsers.add_parser("visualize")
    IMDgroup.pymatgen.cli.imdg_visualize.add_args(parser_visualize)
    parser_visualize.set_defaults(
        func=IMDgroup.pymatgen.cli.imdg_visualize.visualize)

    args = parser.parse_args()

    setup_logger(args)
    logger.info(
        "Running from %s with the following args: %s",
        os.getcwd(),
        args
    )

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
