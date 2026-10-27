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


"""Tests for IMDgroup.gorun.vasp_calculator."""

from __future__ import annotations

import pytest

from IMDgroup.gorun.vasp_calculator import IMDASEVasp


def test_build_pp_list_empty() -> None:
    """_build_pp_list returns an empty list (no VASP_PP_PATH resolution)."""
    calc = IMDASEVasp(directory=".", txt="-")
    assert calc._build_pp_list(atoms=None) == []


def test_write_potcar_is_noop(tmp_path) -> None:
    """write_potcar does not create a POTCAR file."""
    calc = IMDASEVasp(directory=str(tmp_path), txt="-")
    calc.write_potcar(directory=str(tmp_path))
    assert not (tmp_path / "POTCAR").exists()


def test_default_nelect_from_ppp_raises() -> None:
    """default_nelect_from_ppp raises NotImplementedError with a pointer."""
    calc = IMDASEVasp(directory=".", txt="-")
    with pytest.raises(NotImplementedError, match="NELECT"):
        calc.default_nelect_from_ppp()
