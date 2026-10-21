# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
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


"""imdg sub-command to submit/run jobs through the gorun layer."""
from __future__ import annotations

import argparse

from IMDgroup.gorun.gorun import main as gorun_main


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register subcommand arguments.

    The single positional argument captures everything after ``run`` and
    forwards it to gorun.  In practice ``imdg.main`` intercepts ``imdg
    run ...`` before argparse (so ``--help`` and all gorun flags are
    handled by gorun's own parser); this registration exists so that
    ``imdg --help`` lists the subcommand and so that programmatic
    ``parse_args([...])`` calls still dispatch correctly.

    Args:
        parser: Sub-parser from argparse.
    """
    parser.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="arguments passed through to gorun (try: imdg run --help)",
    )


def run(args: argparse.Namespace) -> int:
    """Forward to gorun's entry point.

    Reached only when ``run`` is dispatched through argparse rather than
    the early intercept in ``imdg.main``.

    Args:
        args: Parsed arguments; ``args.argv`` holds the gorun tail.

    Returns:
        int: gorun exit code.
    """
    argv = getattr(args, "argv", None)
    return gorun_main(argv if argv is not None else [], prog="imdg run")
