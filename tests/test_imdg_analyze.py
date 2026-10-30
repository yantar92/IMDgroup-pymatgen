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

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from rich.console import Console

import IMDgroup.pymatgen.cli.imdg_analyze as imdg_analyze
from IMDgroup.common import configure_progress_output
from IMDgroup.pymatgen.cli.imdg_analyze import (
    _read_field_1,
    _render_table,
    _require_structure,
    add_args,
    analyze,
    read_field,
)
from IMDgroup.pymatgen.io.vasp.inputs import Incar


@pytest.fixture(autouse=True)
def _redirect_progress():
    """Keep alive_progress output off stdout so format assertions hold."""
    configure_progress_output()


class _VaspdirStub:
    """Minimal IMDGVaspDir stand-in for the fields analyze reads.

    Each attribute corresponds to one thing ``_read_field_1`` can touch;
    anything left absent degrades to ``N/A`` via :func:`read_field`.
    """

    def __init__(
            self,
            *,
            path: str = '.',
            final_energy_reliable: float | str | None = None,
            final_structure: Structure | None = None,
            initial_structure: Structure | None = None,
            total_magnetization: float | None = None,
            vasprun: object | None = None) -> None:
        self.path = path
        self.final_energy_reliable = final_energy_reliable
        self.final_structure = final_structure
        self.initial_structure = initial_structure
        self.total_magnetization = total_magnetization
        self._vasprun = vasprun

    def __getitem__(self, item):
        if item == 'vasprun.xml':
            return self._vasprun
        return None


class _FakeVasprun:
    """Stand-in for a parsed vasprun.xml with one ionic step."""

    def __init__(self, forces) -> None:
        self.ionic_steps = [{'forces': forces}]


class _AnalyzeVaspdir:
    """Fake IMDGVaspDir for exercising :func:`analyze` control flow.

    Supports just enough of the IMDGVaspDir surface for the fields
    ``analyze`` reads: ``path``, ``__getitem__('INCAR')``, and a
    ``final_energy_reliable`` property.  Every other field degrades to
    ``N/A`` through :func:`read_field`.
    """

    def __init__(self, path, *, energy=None, incar=None) -> None:
        self.path = str(path)
        self._energy = energy
        self._incar = incar

    def __getitem__(self, item):
        if item == 'INCAR':
            return self._incar
        return None

    @property
    def final_energy_reliable(self):
        return self._energy


def _make_fake_read_vaspdirs(vaspdirs):
    """Return a fake ``IMDGVaspDir.read_vaspdirs`` applying *path_filter*."""
    def fake_read_vaspdirs(dirs, path_filter=None):
        result = {}
        for vd in vaspdirs:
            if path_filter is None or path_filter(Path(vd.path)):
                result[vd.path] = vd
        return result
    return fake_read_vaspdirs


def _make_args(**kwargs):
    """Build an argparse-like namespace for :func:`analyze`."""
    defaults = dict(
        dir=['.'],
        exclude=None,
        fields=None,
        format='table',
        exclude_fields=[],
        nogroup=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


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


def _carbon_lithium_structure() -> Structure:
    """A carbon framework with one intercalated Li atom."""
    return Structure(
        lattice=Lattice.cubic(4.0),
        species=['C', 'C', 'Li'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
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


def test_read_field_framework_space_group() -> None:
    """Framework space group ignores the intercalated species."""
    structure = _carbon_lithium_structure()
    framework = Structure(
        lattice=Lattice.cubic(4.0),
        species=['C', 'C'],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    expected = framework.get_space_group_info()[0]
    assert _read_field_1(
        'framework_space_group',
        _VaspdirStub(final_structure=structure),
    ) == expected


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


def test_render_table_default_console(capsys) -> None:
    """Without an explicit console, output goes to stdout."""
    df = pd.DataFrame({'dir': ['a']})
    _render_table(df)
    out = capsys.readouterr().out
    assert 'Directory' in out
    assert 'a' in out


def test_render_table_incar_group_column() -> None:
    """The incar_group column maps to the INCAR-type header."""
    df = pd.DataFrame({'dir': ['a'], 'incar_group': [0]})
    console = Console(record=True, width=200)
    _render_table(df, console=console)
    text = console.export_text()
    assert 'INCAR type' in text
    assert '0' in text


## add_args


def test_add_args_parses_defaults() -> None:
    """add_args registers the subcommand with its documented defaults."""
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args(['some/dir'])
    assert args.dir == ['some/dir']
    assert args.exclude is None
    assert args.fields is None
    assert args.format == 'table'
    assert args.exclude_fields == []
    assert args.nogroup is False


def test_add_args_parses_full() -> None:
    """add_args accepts every optional flag together."""
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args([
        'a', 'b', '--exclude', 'tmp', '--fields', 'energy', 'e_per_atom',
        '--format', 'json', '--exclude-fields', 'energy', '--nogroup'])
    assert args.dir == ['a', 'b']
    assert args.exclude == 'tmp'
    assert args.fields == ['energy', 'e_per_atom']
    assert args.format == 'json'
    assert args.exclude_fields == ['energy']
    assert args.nogroup is True


## _read_field_1 -- remaining fields


def test_read_field_e_per_atom() -> None:
    """Energy per atom divides by the final structure atom count."""
    structure = _carbon_lithium_structure()  # 3 atoms
    vd = _VaspdirStub(
        final_energy_reliable=-12.3, final_structure=structure)
    assert _read_field_1('e_per_atom', vd) == f"{-12.3 / 3:.5f}"


def test_read_field_e_per_atom_string_passthrough() -> None:
    """A string energy (N/A marker) is returned verbatim."""
    assert _read_field_1(
        'e_per_atom', _VaspdirStub(final_energy_reliable='N/A')) == 'N/A'


def test_read_field_total_mag_none() -> None:
    """A missing magnetization is reported as the string None."""
    assert _read_field_1('total_mag', _VaspdirStub()) == 'None'


def test_read_field_total_mag_value() -> None:
    """A numeric magnetization is returned unchanged."""
    assert _read_field_1(
        'total_mag', _VaspdirStub(total_magnetization=3.0)) == 3.0


def test_read_field_max_force_no_vasprun() -> None:
    """Without vasprun.xml the max force is reported as None."""
    assert _read_field_1('max_force', _VaspdirStub()) == 'None'


def test_read_field_max_force_no_forces() -> None:
    """A vasprun without forces in the last step is reported as None."""
    vd = _VaspdirStub(vasprun=_FakeVasprun(None))
    assert _read_field_1('max_force', vd) == 'None'


def test_read_field_max_force() -> None:
    """Max force is the largest force-vector norm in the last step."""
    forces = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    vd = _VaspdirStub(vasprun=_FakeVasprun(forces))
    assert _read_field_1('max_force', vd) == '4.0000'


def test_read_field_vol_change() -> None:
    """%vol is the relative volume change formatted to two decimals."""
    init = _single_atom_structure()  # cubic 4.0 -> volume 64
    final = Structure(
        lattice=Lattice.cubic(4.4),
        species=['Li'], coords=[[0.0, 0.0, 0.0]],
    )
    vd = _VaspdirStub(initial_structure=init, final_structure=final)
    expected = f"{(final.volume / init.volume - 1) * 100:.2f}"
    assert _read_field_1('%vol', vd) == expected


def test_read_field_displacement() -> None:
    """Zero displacement between identical structures formats as 0.00."""
    init = _single_atom_structure()
    vd = _VaspdirStub(initial_structure=init, final_structure=init.copy())
    assert _read_field_1('displ', vd) == '0.00'


def test_read_field_displacement_fallback() -> None:
    """When match_first fails (species differ), the raw path is used."""
    init = Structure(
        lattice=Lattice.cubic(4.0),
        species=['C', 'C'], coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    final = Structure(
        lattice=Lattice.cubic(4.0),
        species=['Li', 'Li'], coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    vd = _VaspdirStub(initial_structure=init, final_structure=final)
    assert _read_field_1('displ', vd) == '0.00'


def test_read_field_space_group() -> None:
    """Space group comes from the final structure."""
    structure = _single_atom_structure()
    expected = structure.get_space_group_info()[0]
    assert _read_field_1(
        'space_group', _VaspdirStub(final_structure=structure)) == expected


def test_read_field_unknown_field_returns_none() -> None:
    """An unrecognized field falls through every elif and returns None."""
    assert _read_field_1('nonexistent', _VaspdirStub()) is None


def _lattice_structure(a, b, c, alpha, beta, gamma) -> Structure:
    """A one-atom structure with explicit lattice parameters."""
    return Structure(
        lattice=Lattice.from_parameters(
            a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma),
        species=['Li'],
        coords=[[0.25, 0.25, 0.25]],
    )


def test_read_field_lattice_parameters() -> None:
    """Raw and percent lattice parameters read from final vs initial."""
    init = _lattice_structure(4.0, 5.0, 6.0, 90.0, 90.0, 90.0)
    final = _lattice_structure(4.4, 5.5, 6.6, 99.0, 100.0, 101.0)
    vd = _VaspdirStub(initial_structure=init, final_structure=final)

    for field, attr in [
        ('a', 'a'), ('b', 'b'), ('c', 'c'),
        ('alpha', 'alpha'), ('beta', 'beta'), ('gamma', 'gamma'),
    ]:
        assert _read_field_1(field, vd) == getattr(final.lattice, attr)

    for field, attr in [
        ('%a', 'a'), ('%b', 'b'), ('%c', 'c'),
        ('%alpha', 'alpha'), ('%beta', 'beta'), ('%gamma', 'gamma'),
    ]:
        i = getattr(init.lattice, attr)
        f = getattr(final.lattice, attr)
        expected = f"{(f / i - 1) * 100:.2f}"
        assert _read_field_1(field, vd) == expected


## analyze


def _patch_read_vaspdirs(monkeypatch, vaspdirs):
    monkeypatch.setattr(
        imdg_analyze, 'IMDGVaspDir',
        SimpleNamespace(read_vaspdirs=_make_fake_read_vaspdirs(vaspdirs)))


def test_analyze_groups_by_incar(monkeypatch, tmp_path, capsys) -> None:
    """Two INCAR groups split into distinct incar_group values."""
    monkeypatch.chdir(tmp_path)
    dirs = [
        _AnalyzeVaspdir(tmp_path / 'a', energy=-1.0,
                        incar=Incar({'ENCUT': 400})),
        _AnalyzeVaspdir(tmp_path / 'b', energy=-2.0,
                        incar=Incar({'ENCUT': 500})),
    ]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(fields=['energy'], format='table')
    assert analyze(args) == 0
    out = capsys.readouterr().out
    assert 'Energy' in out
    assert '-1.00000' in out
    assert '-2.00000' in out


def test_analyze_single_group_incar_group_zero(
        monkeypatch, tmp_path, capsys) -> None:
    """A single INCAR group collapses every row to incar_group 0."""
    monkeypatch.chdir(tmp_path)
    dirs = [
        _AnalyzeVaspdir(tmp_path / 'a', energy=-1.0),
        _AnalyzeVaspdir(tmp_path / 'b', energy=-2.0),
    ]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(fields=['energy'], format='json')
    assert analyze(args) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r['dir'] for r in records] == ['a', 'b']
    assert all(r['incar_group'] == 0 for r in records)


def test_analyze_nogroup(monkeypatch, tmp_path, capsys) -> None:
    """--nogroup drops the incar_group column and sorts by dir only."""
    monkeypatch.chdir(tmp_path)
    dirs = [
        _AnalyzeVaspdir(tmp_path / 'b', energy=-2.0),
        _AnalyzeVaspdir(tmp_path / 'a', energy=-1.0),
    ]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(fields=['energy'], format='json', nogroup=True)
    assert analyze(args) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r['dir'] for r in records] == ['a', 'b']
    assert 'incar_group' not in records[0]


def test_analyze_default_fields_table(monkeypatch, tmp_path, capsys) -> None:
    """A table with no --fields reports the short DEFAULT_FIELDS set."""
    monkeypatch.chdir(tmp_path)
    dirs = [_AnalyzeVaspdir(tmp_path / 'a', energy=-1.0)]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(format='table')
    assert analyze(args) == 0
    out = capsys.readouterr().out
    assert 'Energy' in out  # energy is in DEFAULT_FIELDS
    assert 'N/A' in out     # missing-structure fields degrade


def test_analyze_default_fields_machine_format(
        monkeypatch, tmp_path, capsys) -> None:
    """A machine format with no --fields reports every field."""
    monkeypatch.chdir(tmp_path)
    dirs = [_AnalyzeVaspdir(tmp_path / 'a', energy=-1.0)]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(format='csv')
    assert analyze(args) == 0
    df = pd.read_csv(io.StringIO(capsys.readouterr().out))
    for field in imdg_analyze.ALL_FIELDS:
        if field in ('dir', 'incar_group'):
            continue
        assert field in df.columns


def test_analyze_fields_all(monkeypatch, tmp_path, capsys) -> None:
    """--fields ALL selects the full field set."""
    monkeypatch.chdir(tmp_path)
    dirs = [_AnalyzeVaspdir(tmp_path / 'a', energy=-1.0)]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(fields=['ALL'], format='json')
    assert analyze(args) == 0
    records = json.loads(capsys.readouterr().out)
    for field in imdg_analyze.ALL_FIELDS:
        if field in ('dir', 'incar_group'):
            continue
        assert field in records[0]


def test_analyze_exclude_fields(monkeypatch, tmp_path, capsys) -> None:
    """--exclude-fields removes a field from the report."""
    monkeypatch.chdir(tmp_path)
    dirs = [_AnalyzeVaspdir(tmp_path / 'a', energy=-1.0)]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(
        fields=['energy', 'max_force'],
        exclude_fields=['max_force'], format='csv')
    assert analyze(args) == 0
    df = pd.read_csv(io.StringIO(capsys.readouterr().out))
    assert 'energy' in df.columns
    assert 'max_force' not in df.columns


@pytest.mark.parametrize('fmt', ['csv', 'json', 'orgtbl'])
def test_analyze_machine_formats(monkeypatch, tmp_path, capsys, fmt) -> None:
    """csv/json/orgtbl each bypass the rich table renderer."""
    monkeypatch.chdir(tmp_path)
    dirs = [_AnalyzeVaspdir(tmp_path / 'a', energy=-1.0)]
    _patch_read_vaspdirs(monkeypatch, dirs)
    args = _make_args(fields=['energy'], format=fmt)
    assert analyze(args) == 0
    out = capsys.readouterr().out
    if fmt == 'csv':
        df = pd.read_csv(io.StringIO(out), dtype=str)
        assert df['energy'].tolist() == ['-1.00000']
    elif fmt == 'json':
        records = json.loads(out)
        assert records[0]['energy'] == '-1.00000'
    else:  # orgtbl
        assert '|' in out
        assert 'energy' in out


def test_analyze_exclude_pattern(monkeypatch, tmp_path, capsys) -> None:
    """--exclude filters directories before any field is read."""
    monkeypatch.chdir(tmp_path)
    keep = _AnalyzeVaspdir(tmp_path / 'keep', energy=-1.0)
    skip = _AnalyzeVaspdir(tmp_path / 'skip_me', energy=-2.0)
    _patch_read_vaspdirs(monkeypatch, [keep, skip])
    args = _make_args(fields=['energy'], format='json', exclude='skip_me')
    assert analyze(args) == 0
    records = json.loads(capsys.readouterr().out)
    assert [r['dir'] for r in records] == ['keep']


def test_analyze_empty(monkeypatch, tmp_path, capsys) -> None:
    """No VASP directories produce no output and exit 0."""
    monkeypatch.chdir(tmp_path)
    _patch_read_vaspdirs(monkeypatch, [])
    args = _make_args(fields=['energy'], format='table')
    assert analyze(args) == 0
    assert capsys.readouterr().out == ''
