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


"""Tests for matrifflow.gorun.gorun_maps (backward-compat wrapper)."""

from __future__ import annotations

import sys

import matrifflow.gorun.gorun as gorun
import matrifflow.gorun.gorun_maps as gorun_maps


def test_main_prepends_maps_subcommand(monkeypatch) -> None:
    """Explicit argv is forwarded as ``gorun maps <argv>``."""
    calls: list[list[str]] = []
    monkeypatch.setattr(gorun, "main", lambda argv: calls.append(argv) or 0)
    assert gorun_maps.main(["-d", "--verbose"]) == 0
    assert calls == [["maps", "-d", "--verbose"]]


def test_main_no_args_uses_sys_argv(monkeypatch) -> None:
    """Without argv, the process argv is read and forwarded."""
    calls: list[list[str]] = []
    monkeypatch.setattr(gorun, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["gorun-maps", "foo"])
    assert gorun_maps.main() == 0
    assert calls == [["maps", "foo"]]


def test_main_propagates_return_code(monkeypatch) -> None:
    """The gorun exit code is returned unchanged."""
    monkeypatch.setattr(gorun, "main", lambda argv: 3)
    assert gorun_maps.main([]) == 3
