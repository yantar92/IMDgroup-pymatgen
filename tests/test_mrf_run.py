# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of MatRiFFlow package
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


"""Tests for matrifflow.pymatgen.cli.mrf_run."""

from __future__ import annotations

import argparse

import matrifflow.pymatgen.cli.mrf_run as mrf_run


## add_args


def test_add_args_registers_remainder() -> None:
    """add_args registers a single REMAINDER positional named argv."""
    parser = argparse.ArgumentParser()
    mrf_run.add_args(parser)
    ns = parser.parse_args(['vasp', 'some', 'tail'])
    assert ns.argv == ['vasp', 'some', 'tail']


def test_add_args_empty_tail() -> None:
    """With no trailing tokens, argv defaults to an empty list."""
    parser = argparse.ArgumentParser()
    mrf_run.add_args(parser)
    assert parser.parse_args([]).argv == []


## run


def test_run_forwards_argv_and_prog(monkeypatch) -> None:
    """run() forwards args.argv to gorun_main with a fixed prog."""
    calls: dict = {}

    def fake_main(argv=None, prog=None):
        calls['argv'] = argv
        calls['prog'] = prog
        return 3

    monkeypatch.setattr(mrf_run, 'gorun_main', fake_main)
    assert mrf_run.run(
        argparse.Namespace(argv=['vasp', '--dir', 'x'])) == 3
    assert calls == {'argv': ['vasp', '--dir', 'x'], 'prog': 'mrf run'}


def test_run_defaults_to_empty_argv(monkeypatch) -> None:
    """run() falls back to an empty argv list when the attribute is absent."""
    calls: dict = {}

    def fake_main(argv=None, prog=None):
        calls['argv'] = argv
        calls['prog'] = prog
        return 0

    monkeypatch.setattr(mrf_run, 'gorun_main', fake_main)
    mrf_run.run(argparse.Namespace())
    assert calls == {'argv': [], 'prog': 'mrf run'}
