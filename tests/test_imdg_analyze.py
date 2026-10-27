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


"""Tests for IMDgroup.pymatgen.cli.imdg_analyze."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from rich.console import Console

from IMDgroup.pymatgen.cli.imdg_analyze import (
    _read_field_1,
    _render_table,
    _require_structure,
    read_field,
)


class _VaspdirStub:
    """Minimal IMDGVaspDir stand-in for the fields analyze reads.

    ``path``, ``final_energy_reliable``, and ``final_structure`` are the
    only attributes the tested helpers touch; everything else is left
    absent, matching a deliberately narrow surface.
    """

    def __init__(
            self,
            *,
            path: str = '.',
            final_energy_reliable: float | str | None = None,
            final_structure: Structure | None = None) -> None:
        self.path = path
        self.final_energy_reliable = final_energy_reliable
        self.final_structure = final_structure


class _BoomVaspdir:
    """A vaspdir whose final_energy_reliable property raises *exc*."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def final_energy_reliable(self):
        raise self._exc


## _require_structure


def _single_atom_structure() -> Structure:
    """A one-atom Li structure used as a real Structure fixture."""
    return Structure(
        lattice=Lattice.cubic(4.0),
        species=['Li'],
        coords=[[0.0, 0.0, 0.0]],
    )


def test_require_structure_none_raises() -> None:
    """A missing structure raises TypeError (caught by read_field)."""
    with pytest.raises(TypeError):
        _require_structure(None)


def test_require_structure_returns_structure() -> None:
    """A non-None Structure is returned unchanged."""
    structure = _single_atom_structure()
    assert _require_structure(structure) is structure


## read_field


@pytest.mark.parametrize(
    'exc',
    [FileNotFoundError(), TypeError(), AttributeError()],
)
def test_read_field_swallows_data_errors(exc) -> None:
    """Data-access failures are reported as N/A, not propagated."""
    assert read_field('energy', _BoomVaspdir(exc)) == 'N/A'


def test_read_field_missing_attribute_is_n_a() -> None:
    """A vaspdir lacking the field attribute degrades to N/A."""
    assert read_field('energy', object()) == 'N/A'


def test_read_field_missing_structure_is_n_a() -> None:
    """A field needing a structure reports N/A when the structure is None."""
    assert read_field('space_group', _VaspdirStub()) == 'N/A'


## _read_field_1


def test_read_field_dir_is_relative(monkeypatch, tmp_path: Path) -> None:
    """The dir field reports the path relative to the working directory."""
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / 'sub'
    sub.mkdir()
    assert _read_field_1('dir', _VaspdirStub(path=str(sub))) == 'sub'


def test_read_field_energy_formats_float() -> None:
    """A float energy is formatted to five decimals."""
    assert _read_field_1(
        'energy', _VaspdirStub(final_energy_reliable=-123.456789)
    ) == '-123.45679'


def test_read_field_energy_string_passthrough() -> None:
    """A non-float energy (N/A marker) is returned verbatim."""
    assert _read_field_1(
        'energy', _VaspdirStub(final_energy_reliable='N/A')) == 'N/A'


## _render_table


def test_render_table_smoke() -> None:
    """A DataFrame renders with mapped headers and cell values."""
    df = pd.DataFrame({'dir': ['a'], 'energy': ['-1.23']})
    console = Console(record=True, width=200)
    _render_table(df, console=console)
    text = console.export_text()
    assert 'Directory' in text  # ALL_FIELDS['dir']
    assert 'Energy' in text      # ALL_FIELDS['energy']
    assert 'a' in text
    assert '-1.23' in text
