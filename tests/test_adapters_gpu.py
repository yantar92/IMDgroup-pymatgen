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


"""Tests for IMDgroup.gorun.adapters.gpu (phase 1: _wrap_command only)."""

from __future__ import annotations

from IMDgroup.gorun.adapters.gpu import _wrap_command


def test_wrap_command_exact_output() -> None:
    """A simple command is wrapped in the heredoc template verbatim."""
    assert _wrap_command("echo hi") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "echo hi\n"
        "EOF"
    )


def test_wrap_command_multiline_command() -> None:
    """A multi-line command is placed between the setup line and EOF."""
    assert _wrap_command("line1\nline2") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "line1\nline2\n"
        "EOF"
    )


def test_wrap_command_empty_command() -> None:
    """An empty command still emits the wrapper and setup lines."""
    assert _wrap_command("") == (
        "${GORUN_WRAPPER:-bash} <<'EOF'\n"
        "${GORUN_INNER_SETUP:-}\n"
        "\n"
        "EOF"
    )
