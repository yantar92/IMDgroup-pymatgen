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


"""Tests for IMDgroup.pymatgen.cli.imdg (master CLI dispatch)."""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import sys

import pytest

import IMDgroup.pymatgen.cli.imdg as imdg


def _patch_create_module(monkeypatch, fake_module) -> None:
    """Patch importlib so only imdg_create resolves to *fake_module*.

    main() imports the requested subcommand module lazily; every other
    import must keep resolving normally (for example
    pymatgen.ext.matproj on the error path).
    """
    real_import = importlib.import_module

    def fake_import(name):
        if name == 'IMDgroup.pymatgen.cli.imdg_create':
            return fake_module
        return real_import(name)

    monkeypatch.setattr(importlib, 'import_module', fake_import)


## _showwarning


def test_showwarning_prints_category_and_message(capsys) -> None:
    """The coloured warning printer includes the category name."""
    imdg._showwarning('boom', UserWarning, 'f.py', 12)
    captured = capsys.readouterr()
    assert 'UserWarning:' in captured.err
    assert 'boom' in captured.err


def test_showwarning_honours_file_argument() -> None:
    """An explicit file redirects the printed warning to that file."""
    buf = io.StringIO()
    imdg._showwarning('msg', RuntimeWarning, 'f.py', 1, file=buf)
    assert 'RuntimeWarning:' in buf.getvalue()
    assert 'msg' in buf.getvalue()


## setup_logger


def test_setup_logger_verbosity_levels(monkeypatch) -> None:
    """Verbosity selects the log level and number of handlers."""
    calls: list[dict] = []
    monkeypatch.setattr(logging, 'basicConfig', lambda **kw: calls.append(kw))
    monkeypatch.setattr(logging, 'FileHandler', lambda *a, **k: object())
    monkeypatch.setattr(logging, 'StreamHandler', lambda *a, **k: object())

    imdg.setup_logger(argparse.Namespace(verbosity=0))
    imdg.setup_logger(argparse.Namespace(verbosity=1))
    imdg.setup_logger(argparse.Namespace(verbosity=2))

    assert calls[0]['level'] == logging.INFO
    assert len(calls[0]['handlers']) == 1  # file only
    assert calls[1]['level'] == logging.INFO
    assert len(calls[1]['handlers']) == 2  # file + stdout
    assert calls[2]['level'] == logging.DEBUG
    assert len(calls[2]['handlers']) == 2


## main


def test_main_run_intercepts_to_gorun(monkeypatch) -> None:
    """``imdg run ...`` is forwarded to gorun before argparse runs."""
    calls: dict = {}
    monkeypatch.setattr(sys, 'argv', ['imdg', 'run', 'vasp', '--dir', 'x'])

    def fake_gorun_main(argv=None, prog=None):
        calls['argv'] = argv
        calls['prog'] = prog
        return 0

    monkeypatch.setattr('IMDgroup.gorun.gorun.main', fake_gorun_main)
    assert imdg.main() == 0
    assert calls == {'argv': ['vasp', '--dir', 'x'], 'prog': 'imdg run'}


def test_main_dispatches_to_subcommand(monkeypatch) -> None:
    """A subcommand module is imported and its func run, returning its code."""
    monkeypatch.setattr(sys, 'argv', ['imdg', 'create'])
    monkeypatch.setattr(imdg, 'setup_logger', lambda args: None)

    class _FakeModule:
        def __init__(self) -> None:
            self.add_args_called = False

        def add_args(self, parser) -> None:
            self.add_args_called = True

        def create(self, args) -> int:
            return 7

    fake = _FakeModule()
    _patch_create_module(monkeypatch, fake)
    assert imdg.main() == 7
    assert fake.add_args_called


def test_main_value_error_is_reported(monkeypatch, capsys) -> None:
    """A ValueError from the subcommand is reported as a one-line error."""
    monkeypatch.setattr(sys, 'argv', ['imdg', 'create'])
    monkeypatch.setattr(imdg, 'setup_logger', lambda args: None)

    class _FakeModule:
        def add_args(self, parser) -> None:
            pass

        def create(self, args) -> int:
            raise ValueError('nope')

    _patch_create_module(monkeypatch, _FakeModule())
    assert imdg.main() == 1
    assert 'nope' in capsys.readouterr().err


def test_main_value_error_reraised_with_verbose(monkeypatch) -> None:
    """With -vv, a user error re-raises so the full traceback is shown."""
    monkeypatch.setattr(sys, 'argv', ['imdg', '-vv', 'create'])
    monkeypatch.setattr(imdg, 'setup_logger', lambda args: None)

    class _FakeModule:
        def add_args(self, parser) -> None:
            pass

        def create(self, args) -> int:
            raise ValueError('nope')

    _patch_create_module(monkeypatch, _FakeModule())
    with pytest.raises(ValueError, match='nope'):
        imdg.main()


def test_main_no_subcommand_exits(monkeypatch) -> None:
    """Running with no subcommand exits via argparse."""
    monkeypatch.setattr(sys, 'argv', ['imdg'])
    with pytest.raises(SystemExit) as exc:
        imdg.main()
    assert exc.value.code == 2
