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


"""Tests for IMDgroup.pymatgen.cli.imdg_create."""

from __future__ import annotations

import argparse

import pymatgen.core as pmg
import pytest
from pymatgen.ext.matproj import MPRestError
from pymatgen.util.testing import MatSciTest

import IMDgroup.pymatgen.cli.imdg_create as imdg_create


POSCAR = """Si
1.0
5.431 0.0 0.0
0.0 5.431 0.0
0.0 0.0 5.431
Si
1
Direct
0.0 0.0 0.0
"""


def _write_poscar(path) -> None:
    """Write a single-atom Si POSCAR to *path*."""
    path.write_text(POSCAR)


## add_args


def test_add_args_registers_positional_and_output() -> None:
    """The subparser registers the positional ``what`` and ``--output``."""
    parser = argparse.ArgumentParser()
    imdg_create.add_args(parser)
    args = parser.parse_args(["mp-1234", "--output", "out"])
    assert args.what == "mp-1234"
    assert args.output == "out"


## create_from_file


def test_create_from_file_reads_structure(tmp_path) -> None:
    """A POSCAR on disk is parsed into a Structure."""
    path = tmp_path / 'POSCAR'
    _write_poscar(path)
    structure = imdg_create.create_from_file(str(path))
    assert structure[0].species_string == 'Si'
    assert len(structure) == 1


## create_from_atom_name


def test_create_from_atom_name_boxes_single_atom() -> None:
    """A single atom is placed in the centre of the requested box."""
    structure = imdg_create.create_from_atom_name('Li', [5.0, 6.0, 7.0])
    assert len(structure) == 1
    assert structure[0].species_string == 'Li'
    assert structure.lattice.a == pytest.approx(5.0)
    assert structure.lattice.b == pytest.approx(6.0)
    assert structure.lattice.c == pytest.approx(7.0)
    assert structure[0].frac_coords == pytest.approx([0.5, 0.5, 0.5])


## create_from_mpid


def test_create_from_mpid_missing_api_key(monkeypatch) -> None:
    """An invalid-length PMG_MAPI_KEY is rejected before any network call."""
    monkeypatch.setattr(pmg, 'SETTINGS', {'PMG_MAPI_KEY': 'short'})
    with pytest.raises(ValueError, match='API key'):
        imdg_create.create_from_mpid('mp-1234')


def test_create_from_mpid_401_maps_to_value_error(monkeypatch) -> None:
    """An HTTP 401 rejection is reported as a user-facing ValueError."""
    monkeypatch.setattr(pmg, 'SETTINGS', {'PMG_MAPI_KEY': 'x' * 32})

    def fake_from_id(mpid):
        raise MPRestError('status code 401 rejected')

    monkeypatch.setattr(pmg.Structure, 'from_id', fake_from_id)
    with pytest.raises(ValueError, match='API key was rejected'):
        imdg_create.create_from_mpid('mp-1234')


def test_create_from_mpid_non_401_reraised(monkeypatch) -> None:
    """A non-401 MPRestError is re-raised unchanged."""
    monkeypatch.setattr(pmg, 'SETTINGS', {'PMG_MAPI_KEY': 'x' * 32})

    def fake_from_id(mpid):
        raise MPRestError('some other API failure')

    monkeypatch.setattr(pmg.Structure, 'from_id', fake_from_id)
    with pytest.raises(MPRestError, match='some other API failure'):
        imdg_create.create_from_mpid('mp-1234')


def test_create_from_mpid_standardises_non_primitive(monkeypatch) -> None:
    """A non-primitive fetched structure is reduced to its standard cell."""
    monkeypatch.setattr(pmg, 'SETTINGS', {'PMG_MAPI_KEY': 'x' * 32})
    primitive = MatSciTest().get_structure('CsCl')

    def fake_from_id(mpid):
        # A supercell is non-primitive, so standardisation changes the lattice.
        return primitive * (2, 1, 1)

    monkeypatch.setattr(pmg.Structure, 'from_id', fake_from_id)
    structure = imdg_create.create_from_mpid('mp-1234')
    assert structure.properties['mpid'] == 'mp-1234'
    assert len(structure) == len(primitive)


def test_create_from_mpid_happy_path(monkeypatch) -> None:
    """A fetched structure is standardised and tagged with its mpid."""
    monkeypatch.setattr(pmg, 'SETTINGS', {'PMG_MAPI_KEY': 'x' * 32})

    def fake_from_id(mpid):
        return MatSciTest().get_structure('CsCl')

    monkeypatch.setattr(pmg.Structure, 'from_id', fake_from_id)
    structure = imdg_create.create_from_mpid('mp-1234')
    assert structure.properties['mpid'] == 'mp-1234'
    assert structure.composition.reduced_formula == 'CsCl'


## create


def _patch_inputset(monkeypatch):
    """Replace IMDStandardVaspInputSet with a recording fake.

    Returns the list of created fake instances.
    """
    records = []

    class _FakeInputSet:
        def __init__(self, structure=None, **kwargs) -> None:
            self.structure = structure
            self.kwargs = kwargs
            self.incar = {'SYSTEM': 'TestSystem'}
            self.written = None
            records.append(self)

        def write_input(self, output_dir=None) -> None:
            self.written = output_dir

    monkeypatch.setattr(imdg_create, 'IMDStandardVaspInputSet', _FakeInputSet)
    return records


def test_create_dispatch_file(monkeypatch, tmp_path) -> None:
    """A file path routes through create_from_file."""
    path = tmp_path / 'POSCAR'
    _write_poscar(path)
    records = _patch_inputset(monkeypatch)
    monkeypatch.setattr(
        imdg_create, 'create_from_file', lambda p: 'FROM_FILE')

    args = argparse.Namespace(what=str(path), output=None)
    assert imdg_create.create(args) == 0

    assert len(records) == 1
    assert records[0].structure == 'FROM_FILE'
    assert records[0].written == 'TestSystem'


def test_create_dispatch_atom_name(monkeypatch) -> None:
    """An atom+cell string routes through create_from_atom_name."""
    records = _patch_inputset(monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        imdg_create,
        'create_from_atom_name',
        lambda name, size: calls.append((name, size)) or 'FROM_ATOM',
    )

    args = argparse.Namespace(what='Li 2x3x4', output=None)
    assert imdg_create.create(args) == 0

    assert calls == [('Li', [2.0, 3.0, 4.0])]
    assert len(records) == 1
    assert records[0].structure == 'FROM_ATOM'
    assert records[0].kwargs['user_incar_settings'] == {
        'SYSTEM': 'Li.boxed', 'NCORE': 1}
    assert records[0].kwargs['user_kpoints_settings'] == {'length': 1}


def test_create_dispatch_mpid(monkeypatch) -> None:
    """Anything else is treated as a Materials Project id."""
    records = _patch_inputset(monkeypatch)
    calls: list = []
    monkeypatch.setattr(
        imdg_create,
        'create_from_mpid',
        lambda mpid: calls.append(mpid) or 'FROM_MPID',
    )

    args = argparse.Namespace(what='mp-1234', output=None)
    assert imdg_create.create(args) == 0

    assert calls == ['mp-1234']
    assert len(records) == 1
    assert records[0].structure == 'FROM_MPID'


def test_create_empty_output_raises(monkeypatch) -> None:
    """--output "" is rejected with a clear error."""
    _patch_inputset(monkeypatch)
    monkeypatch.setattr(
        imdg_create, 'create_from_mpid', lambda mpid: 'FROM_MPID')

    args = argparse.Namespace(what='mp-1234', output='')
    with pytest.raises(ValueError, match='--output cannot be empty'):
        imdg_create.create(args)
