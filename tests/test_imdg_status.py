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


"""Tests for IMDgroup.pymatgen.cli.imdg_status."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar

import IMDgroup.pymatgen.cli.imdg_status as imdg_status
from IMDgroup.pymatgen.cli.imdg_status import (
    _RunRow,
    _collect_warnings,
    _details_text,
    _get_neb_summary,
    _strip_markup,
    format_duration,
    format_timing_stats,
    print_seconds,
    slurm_runningp,
    vasp_output_time,
)
from IMDgroup.pymatgen.io.vasp.diagnostics import VaspWarnings, VaspWarningRecord
from IMDgroup.pymatgen.io.vasp.outputs import MeanStd, RunTimingStats
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir


def _write_structure(dirpath: Path, x: float, filename: str = "POSCAR") -> None:
    """Write a single-atom Li structure file at fractional coordinate ``x``."""
    dirpath.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        lattice=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Li"],
        coords=[[x, 0.0, 0.0]],
    )
    Poscar(structure).write_file(dirpath / filename)


def test_get_neb_summary_end_images_without_contcar(tmp_path) -> None:
    """NEB summary falls back to POSCAR for fixed end images.

    End images (00 and NN) are fixed endpoints that never produce a
    CONTCAR/vasprun.xml.  ``_get_neb_summary`` must not raise when
    only their POSCAR is present.
    """
    (tmp_path / "INCAR").write_text("IMAGES = 1\n")
    _write_structure(tmp_path / "00", x=0.0)
    _write_structure(tmp_path / "01", x=0.1)
    _write_structure(tmp_path / "01", x=0.15, filename="CONTCAR")
    _write_structure(tmp_path / "02", x=0.2)

    vaspdir = IMDGVaspDir(str(tmp_path))
    summary = _get_neb_summary(vaspdir)

    assert "IMAGE DISTANCES (initial)" in summary
    assert "IMAGE DISTANCES" in summary


## print_seconds


def test_print_seconds_zero_and_sub_minute() -> None:
    """Zero and sub-minute durations render as 'now'."""
    assert print_seconds(0) == 'now'
    assert print_seconds(59) == 'now'
    assert print_seconds(-59) == 'now'


def test_print_seconds_future() -> None:
    """Positive durations use 'in' and decompose into d/h/m."""
    assert print_seconds(3600) == 'in 1h'
    assert print_seconds(3660) == 'in 1h 1m'
    assert print_seconds(90061) == 'in 1d 1h 1m'


def test_print_seconds_past() -> None:
    """Negative durations omit 'in' and append 'ago'."""
    assert print_seconds(-3600) == '1h ago'


## format_duration


def test_format_duration_undefined() -> None:
    """None and NaN both render as N/A."""
    assert format_duration(None) == 'N/A'
    assert format_duration(float('nan')) == 'N/A'


def test_format_duration_units() -> None:
    """The value is rounded to its most significant non-zero unit."""
    assert format_duration(6.1) == '6s'
    assert format_duration(59) == '59s'
    assert format_duration(60) == '1m'
    assert format_duration(891) == '15m'
    assert format_duration(3600) == '1h'


## format_timing_stats


def _stats(scf: tuple, ionic: tuple) -> RunTimingStats:
    return RunTimingStats(
        scf=MeanStd(*scf),
        ionic=MeanStd(*ionic),
    )


def test_format_timing_stats_empty() -> None:
    """No recorded SCF cycle produces an empty string."""
    nan = (0, float('nan'), float('nan'))
    assert format_timing_stats(_stats(nan, nan)) == ''


def test_format_timing_stats_with_std() -> None:
    """At least two samples include the +/- std term."""
    assert format_timing_stats(
        _stats((2, 6.1, 2.5), (2, 891.0, 30.0))
    ) == 'SCF: 6s±2s relax: 15m±30s'


def test_format_timing_stats_single_sample_omits_std() -> None:
    """A single sample (or NaN std) drops the +/- std term."""
    nan = float('nan')
    assert format_timing_stats(
        _stats((1, 6.1, nan), (1, 891.0, nan))
    ) == 'SCF: 6s relax: 15m'


## _collect_warnings


def _warnings() -> VaspWarnings:
    return VaspWarnings([
        VaspWarningRecord('alpha', 'msg A', tips=['tip1', 'tip2']),
        VaspWarningRecord('beta', 'msg B'),
    ])


def test_collect_warnings_returns_names_and_meta() -> None:
    """Names and (message, tips) metadata are collected per record."""
    names, meta = _collect_warnings(_warnings())
    assert names == {'alpha', 'beta'}
    assert meta['alpha'] == ('msg A', ['tip1', 'tip2'])
    assert meta['beta'] == ('msg B', [])


def test_collect_warnings_honours_ignore_list() -> None:
    """Ignored warning names are excluded from both outputs."""
    names, meta = _collect_warnings(_warnings(), ignore_list=['alpha'])
    assert names == {'beta'}
    assert 'alpha' not in meta


## _strip_markup / _details_text


def test_strip_markup_removes_colour_tags() -> None:
    """Simple rich colour tags are removed, plain text preserved."""
    assert _strip_markup('[red]foo[/red] bar') == 'foo bar'


def _row(**overrides) -> _RunRow:
    defaults = dict(
        dir_full='dir',
        when='now',
        running=False,
        converged=False,
        is_neb=False,
        status_label='unconverged',
        status_color='red',
        warn_names=(),
        energy='',
        fmax='',
        per_step='',
        progress='',
        cpu_time='',
        n_cores='',
        neb_summary='',
    )
    defaults.update(overrides)
    return _RunRow(**defaults)


def test_details_text_empty() -> None:
    """An empty row renders an empty details string."""
    assert _details_text(_row()) == ''


def test_details_text_combines_fields() -> None:
    """Progress, CPU time, and NEB summary are ' | '-joined."""
    row = _row(
        progress='step 3/5',
        cpu_time='1h',
        n_cores='4',
        neb_summary='[red]0.00Å[/red]',
    )
    assert _details_text(row) == 'step 3/5 | CPU 1h (4 cores) | 0.00Å'


def test_details_text_cpu_without_cores() -> None:
    """CPU time without a core count omits the parenthetical."""
    assert _details_text(_row(cpu_time='1h')) == 'CPU 1h'


## vasp_output_time


def test_vasp_output_time_outcar(tmp_path) -> None:
    """OUTCAR mtime wins over any other file."""
    outcar = tmp_path / 'OUTCAR'
    outcar.write_text('x')
    assert vasp_output_time(str(tmp_path)) == pytest.approx(
        os.path.getmtime(str(outcar)))


def test_vasp_output_time_vasprun_fallback(tmp_path) -> None:
    """Without OUTCAR, the vasprun.xml mtime is reported."""
    vasprun = tmp_path / 'vasprun.xml'
    vasprun.write_text('<modeling/>')
    assert vasp_output_time(str(tmp_path)) == pytest.approx(
        os.path.getmtime(str(vasprun)))


def test_vasp_output_time_none(tmp_path) -> None:
    """No VASP output file yields None."""
    assert vasp_output_time(str(tmp_path)) is None


def test_vasp_output_time_neb_max(tmp_path) -> None:
    """A NEB directory reports the latest image mtime."""
    (tmp_path / 'INCAR').write_text('IMAGES = 1\n')
    for name in ('00', '01', '02'):
        image = tmp_path / name
        image.mkdir()
        (image / 'OUTCAR').write_text('x')
    times = [
        os.path.getmtime(str(tmp_path / name / 'OUTCAR'))
        for name in ('00', '01', '02')
    ]
    assert vasp_output_time(str(tmp_path)) == pytest.approx(max(times))


## slurm_runningp


def test_slurm_runningp_empty_queue(monkeypatch, tmp_path) -> None:
    """An empty slurm queue and no NEB parent means not running."""
    monkeypatch.setattr(imdg_status, '_slurm_get_queue', lambda: [])
    assert slurm_runningp(str(tmp_path)) is False


def test_slurm_runningp_path_in_queue(monkeypatch, tmp_path) -> None:
    """A path present in the queue is reported as running."""
    monkeypatch.setattr(
        imdg_status,
        '_slurm_get_queue',
        lambda: [os.path.abspath(str(tmp_path)).encode('utf-8')],
    )
    assert slurm_runningp(str(tmp_path)) is True


def test_slurm_runningp_neb_parent_recursion(monkeypatch, tmp_path) -> None:
    """A numbered dir under a NEB parent recurses to the parent."""
    parent = tmp_path / 'neb'
    parent.mkdir()
    (parent / 'INCAR').write_text('IMAGES = 1\n')
    image = parent / '00'
    image.mkdir()

    calls: list[int] = []
    monkeypatch.setattr(
        imdg_status,
        '_slurm_get_queue',
        lambda: calls.append(1) or [],
    )
    assert slurm_runningp(str(image)) is False
    assert len(calls) == 2  # once for image, once for parent
