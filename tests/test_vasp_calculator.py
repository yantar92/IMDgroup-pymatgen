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
