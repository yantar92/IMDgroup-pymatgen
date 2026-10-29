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

import argparse
import io
import os
import time
from pathlib import Path

import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from rich.console import Console

import IMDgroup.pymatgen.cli.imdg_status as imdg_status
from IMDgroup.pymatgen.cli.imdg_status import (
    _RunRow,
    _add_run_row,
    _build_status_table,
    _collect_warnings,
    _details_text,
    _get_neb_summary,
    _get_progress,
    _source_label,
    _strip_markup,
    add_args,
    custom_showwarning,
    format_duration,
    format_timing_stats,
    print_seconds,
    slurm_runningp,
    status,
    vasp_output_time,
)
from IMDgroup.pymatgen.io.vasp.diagnostics import VaspWarnings, VaspWarningRecord
from IMDgroup.pymatgen.io.vasp.inputs import Incar
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
        VaspWarningRecord(
            'alpha', 'msg A', tips=['tip1', 'tip2'], source='/a/b'),
        VaspWarningRecord('beta', 'msg B'),
    ])


def test_collect_warnings_returns_names_and_meta() -> None:
    """Names and (message, tips, source) metadata are collected per record."""
    names, meta = _collect_warnings(_warnings())
    assert names == {'alpha', 'beta'}
    assert meta['alpha'] == ('msg A', ['tip1', 'tip2'], '/a/b')
    assert meta['beta'] == ('msg B', [], None)


def test_collect_warnings_honours_ignore_list() -> None:
    """Ignored warning names are excluded from both outputs."""
    names, meta = _collect_warnings(_warnings(), ignore_list=['alpha'])
    assert names == {'beta'}
    assert 'alpha' not in meta


## _strip_markup / _details_text


def test_strip_markup_removes_colour_tags() -> None:
    """Simple rich colour tags are removed, plain text preserved."""
    assert _strip_markup('[red]foo[/red] bar') == 'foo bar'


def test_source_label_renders_relative_path() -> None:
    """A record source is shown relative to the current directory."""
    assert _source_label(None) == ""
    assert _source_label("") == ""
    assert _source_label("/a/b/c") == os.path.relpath("/a/b/c")


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


## custom_showwarning


def test_custom_showwarning_prints_category_and_message(capsys) -> None:
    """The warning printer includes the category name and message."""
    custom_showwarning('boom', UserWarning, 'f.py', 12)
    captured = capsys.readouterr()
    assert 'UserWarning:' in captured.err
    assert 'boom' in captured.err


def test_custom_showwarning_honours_file_argument() -> None:
    """An explicit file redirects the printed warning to that file."""
    buf = io.StringIO()
    custom_showwarning('msg', RuntimeWarning, 'f.py', 1, file=buf)
    assert 'RuntimeWarning:' in buf.getvalue()
    assert 'msg' in buf.getvalue()


## _slurm_get_queue


@pytest.fixture(autouse=True)
def _clear_slurm_queue_cache() -> None:
    """Invalidate the ttl_cache so monkeypatches do not leak across tests."""
    imdg_status._slurm_get_queue.cache_clear()
    yield
    imdg_status._slurm_get_queue.cache_clear()


def test_slurm_get_queue_without_squeue(monkeypatch) -> None:
    """When squeue is absent, the queue is empty."""
    monkeypatch.setattr(imdg_status.shutil, 'which', lambda name: None)
    assert imdg_status._slurm_get_queue() == []


def test_slurm_get_queue_parses_output(monkeypatch) -> None:
    """squeue output is split into a list of byte strings."""
    monkeypatch.setattr(
        imdg_status.shutil, 'which', lambda name: '/usr/bin/squeue')
    monkeypatch.setattr(
        imdg_status.subprocess, 'check_output',
        lambda *a, **k: b'/path/a\n/path/b\n')
    assert imdg_status._slurm_get_queue() == [b'/path/a', b'/path/b']


## add_args


def _parse_status_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_args(parser)
    return parser.parse_args(argv)


def test_add_args_defaults() -> None:
    """Positional dir and every flag have the documented defaults."""
    args = _parse_status_args(['some/dir'])
    assert args.dir == ['some/dir']
    assert args.exclude is None
    assert args.include is None
    assert args.nowarn is None
    assert args.problematic is False
    assert args.skip_converged is False
    assert args.skip_running is False
    assert args.fast is False


def test_add_args_skip_converged_alias() -> None:
    """The undocumented --skip_converged alias maps to skip_converged."""
    assert _parse_status_args(['--skip_converged', 'd']).skip_converged is True
    assert _parse_status_args(['--skip-converged', 'd']).skip_converged is True


## _get_progress


class _FakeFile:
    """A stand-in for a log's ``.file`` path attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeLog:
    """A log stand-in exposing the ``.progress`` and ``.file`` attributes."""

    def __init__(self, progress, file_name: str = 'fake.log') -> None:
        self.progress = progress
        self.file = _FakeFile(file_name)


def test_get_progress_returns_last_message() -> None:
    """The last of several recorded progress messages is returned."""
    progress = VaspWarnings([
        VaspWarningRecord('step_early', 'step: 1/5'),
        VaspWarningRecord('step_late', 'step: 3/5'),
    ])
    assert _get_progress([_FakeLog(progress)]) == 'step: 3/5'


def test_get_progress_empty_is_n_a() -> None:
    """No progress records degrade to N/A."""
    assert _get_progress([_FakeLog(VaspWarnings())]) == 'N/A'


## _build_status_table / _add_run_row


def _render_table(table) -> str:
    console = Console(record=True, width=200)
    console.print(table)
    return console.export_text()


def test_build_status_table_columns_and_cells() -> None:
    """Rows render their status, warning, energy, and NEB prefix cells."""
    rows = [
        _row(
            dir_full='a', warn_names=('alpha',),
            status_label='unconverged', status_color='red'),
        _row(
            dir_full='b', is_neb=True,
            status_label='converged', status_color='green',
            energy='-1.2', fmax='0.0'),
    ]
    table = _build_status_table(rows, dir_max=30)
    assert len(table.columns) == 8
    text = _render_table(table)
    assert 'a' in text
    assert 'IMAGES b' in text
    assert 'alpha' in text
    assert '-1.2' in text


## status()


class _FakeVaspdir:
    """Minimal IMDGVaspDir stand-in covering the status() surface."""

    def __init__(self, *, incar=None, nebp=False,
                 converged_electronic=True, converged_ionic=True,
                 converged_manual=True, converged_sequence=True,
                 logs=None, warnings=None, final_energy=-1.0,
                 run_stats=None, max_force=None, timing_stats=None,
                 vasprun=None, neb_dirs=None) -> None:
        self._incar = incar
        self.nebp = nebp
        self.converged_electronic = converged_electronic
        self.converged_ionic = converged_ionic
        self.converged_manual = converged_manual
        self.converged_sequence = converged_sequence
        self._logs = logs if logs is not None else []
        self.warnings = warnings if warnings is not None else VaspWarnings()
        self.final_energy = final_energy
        self.run_stats = run_stats
        self._max_force = max_force
        self.timing_stats = timing_stats
        self._vasprun = vasprun
        self._neb_dirs = neb_dirs

    def __getitem__(self, item):
        return {'INCAR': self._incar, 'vasprun.xml': self._vasprun}[item]

    def logs(self):
        return self._logs

    def max_force(self):
        return self._max_force

    def neb_dirs(self):
        return self._neb_dirs


def _status_dir(tmp_path, name, fake, *, marker='OUTCAR'):
    """Create a real directory backing *fake* and return (path, fake)."""
    d = tmp_path / name
    d.mkdir()
    (d / marker).write_text('')
    return str(d), fake


def _status(monkeypatch, dirs, *, slurm_paths=(), **ns) -> tuple[str, int]:
    """Run status() against fake vaspdirs and return (text, exit code).

    Forces a non-terminal record console so the non-Live printing path
    is exercised and captured deterministically.
    """
    monkeypatch.setattr(
        imdg_status.IMDGVaspDir, 'read_vaspdirs',
        staticmethod(lambda rootpath=None, path_filter=None: dict(dirs)))
    monkeypatch.setattr(
        imdg_status, 'slurm_runningp', lambda path: path in slurm_paths)
    monkeypatch.setattr(
        imdg_status, 'vasp_output_time', lambda path: time.time())
    console = Console(record=True, width=100)
    console._force_terminal = False
    monkeypatch.setattr(imdg_status, 'console', console)
    args = argparse.Namespace(**{
        'dir': ['.'], 'exclude': None, 'include': None, 'nowarn': None,
        'problematic': False, 'skip_converged': False,
        'skip_running': False, 'fast': False, **ns,
    })
    code = status(args)
    return console.export_text(), code


def test_status_converged(tmp_path, monkeypatch) -> None:
    """A converged run reports converged with energy and CPU time."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir(
        final_energy=-1.23456,
        run_stats={'Total CPU time used (sec)': 60.0, 'cores': 4},
        max_force=0.001,
        timing_stats=RunTimingStats(
            MeanStd(2, 6.1, 2.5), MeanStd(2, 891.0, 30.0)),
    ))
    text, code = _status(monkeypatch, {path: fake})
    assert code == 0
    assert 'converged' in text
    assert '1 runs: 0 running, 1 converged, 0 unconverged' in text
    assert '-1.2346' in text
    assert 'CPU 0:01:00' in text


def test_status_running(tmp_path, monkeypatch) -> None:
    """A running run reports progress and its warning category."""
    path, fake = _status_dir(
        tmp_path, 'run', _FakeVaspdir(
            logs=[_FakeLog(VaspWarnings([
                VaspWarningRecord('step', 'step: 3/5')]))],
            warnings=VaspWarnings([
                VaspWarningRecord('slurm_error', 'boom', tips=['tip']),
            ]),
        ), marker='slurm-1.out')
    text, code = _status(monkeypatch, {path: fake}, slurm_paths={path})
    assert code == 0
    assert 'running' in text
    assert '1 runs: 1 running, 0 converged, 0 unconverged' in text
    assert 'step: 3/5' in text
    assert 'slurm_error' in text


def test_status_unconverged_without_warnings(tmp_path, monkeypatch) -> None:
    """An unconverged run with no warnings lands in the UNCONVERGED group."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir(
        converged_electronic=False, converged_ionic=False,
        logs=[_FakeLog(VaspWarnings())], warnings=VaspWarnings()))
    text, code = _status(monkeypatch, {path: fake})
    assert code == 0
    assert '1 runs: 0 running, 0 converged, 1 unconverged' in text
    assert 'UNCONVERGED' in text


def test_status_skip_running(tmp_path, monkeypatch) -> None:
    """--skip-running drops running runs from the table."""
    path, fake = _status_dir(
        tmp_path, 'run', _FakeVaspdir(), marker='slurm-1.out')
    text, code = _status(
        monkeypatch, {path: fake}, slurm_paths={path}, skip_running=True)
    assert code == 0
    assert '1 skipped' in text
    assert '0 runs: 0 running, 0 converged, 0 unconverged' in text


def test_status_skip_converged(tmp_path, monkeypatch) -> None:
    """--skip-converged drops converged runs from the table."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir())
    text, code = _status(monkeypatch, {path: fake}, skip_converged=True)
    assert code == 0
    assert '1 skipped' in text
    assert '0 runs: 0 running, 0 converged, 0 unconverged' in text


def test_status_problematic_filters_clean(tmp_path, monkeypatch) -> None:
    """--problematic keeps only runs with warnings."""
    clean_path, clean_fake = _status_dir(tmp_path, 'clean', _FakeVaspdir())
    dirty_path, dirty_fake = _status_dir(tmp_path, 'dirty', _FakeVaspdir(
        converged_electronic=False, converged_ionic=False,
        logs=[_FakeLog(VaspWarnings())],
        warnings=VaspWarnings([VaspWarningRecord('slurm_error', 'boom')]),
    ))
    text, code = _status(monkeypatch, {
        clean_path: clean_fake, dirty_path: dirty_fake,
    }, problematic=True)
    assert code == 0
    assert 'dirty' in text
    assert 'clean' not in text


def test_status_fast_skips_timing(tmp_path, monkeypatch) -> None:
    """--fast does not parse OUTCAR timing for running runs."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir())
    calls = []
    monkeypatch.setattr(
        imdg_status, 'read_outcar_timing_stats',
        lambda p: calls.append(p) or None)
    text, code = _status(
        monkeypatch, {path: fake}, slurm_paths={path}, fast=True)
    assert code == 0
    assert calls == []
    assert 'SCF:' not in text


def test_status_fast_converged_omits_stats(tmp_path, monkeypatch) -> None:
    """--fast omits CPU time, max force, and per-step for converged runs."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir(
        run_stats={'Total CPU time used (sec)': 60.0, 'cores': 4},
        max_force=0.001,
        timing_stats=RunTimingStats(
            MeanStd(2, 6.1, 2.5), MeanStd(2, 891.0, 30.0)),
    ))
    text, code = _status(monkeypatch, {path: fake}, fast=True)
    assert code == 0
    assert 'CPU' not in text
    assert '0.0010' not in text
    assert 'SCF:' not in text
    assert '-1.0000' in text


def test_status_nowarn_filters_warning(tmp_path, monkeypatch) -> None:
    """--nowarn suppresses the named warning from the summary."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir(
        converged_electronic=False, converged_ionic=False,
        logs=[_FakeLog(VaspWarnings())],
        warnings=VaspWarnings([VaspWarningRecord('slurm_error', 'boom')]),
    ))
    text, code = _status(
        monkeypatch, {path: fake}, nowarn=['slurm_error'])
    assert code == 0
    assert 'slurm_error' not in text


def test_status_incomplete_vasprun(tmp_path, monkeypatch) -> None:
    """A present but unparseable vasprun.xml is flagged incomplete."""
    d = tmp_path / 'run'
    d.mkdir()
    (d / 'OUTCAR').write_text('')
    (d / 'vasprun.xml').write_text('<modeling/>')
    fake = _FakeVaspdir(
        vasprun=None, final_energy=None,
        converged_electronic=False, converged_ionic=False)
    text, code = _status(monkeypatch, {str(d): fake})
    assert code == 0
    assert 'incomplete vasprun.xml' in text


def test_status_neb(tmp_path, monkeypatch) -> None:
    """A NEB run renders the IMAGES prefix and its summary."""
    path, fake = _status_dir(tmp_path, 'neb', _FakeVaspdir(nebp=True))
    monkeypatch.setattr(
        imdg_status, '_get_neb_summary', lambda vaspdir: '[red]0.00Å[/red]')
    text, code = _status(monkeypatch, {path: fake})
    assert code == 0
    assert 'IMAGES' in text
    assert '0.00' in text


def test_status_multiple_incars(tmp_path, monkeypatch) -> None:
    """Directories with differing INCARs trigger the group report."""
    a_path, a_fake = _status_dir(
        tmp_path, 'a', _FakeVaspdir(incar=Incar({'ENCUT': 500})))
    b_path, b_fake = _status_dir(
        tmp_path, 'b', _FakeVaspdir(incar=Incar({'ENCUT': 600})))
    text, code = _status(monkeypatch, {a_path: a_fake, b_path: b_fake})
    assert code == 0
    assert 'Multiple INCARs found' in text


def test_status_no_output_listing(tmp_path, monkeypatch) -> None:
    """Input-only directories are listed as having no output."""
    d = tmp_path / 'empty'
    d.mkdir()
    (d / 'POSCAR').write_text('x')
    text, code = _status(monkeypatch, {str(d): _FakeVaspdir()})
    assert code == 0
    assert 'Directories containing VASP input but not output' in text


def test_status_running_per_step(tmp_path, monkeypatch) -> None:
    """A running run with OUTCAR computes a PER-STEP timing string."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir())
    stats = RunTimingStats(
        MeanStd(2, 6.1, 2.5), MeanStd(2, 891.0, 30.0))
    monkeypatch.setattr(
        imdg_status, 'read_outcar_timing_stats', lambda path: stats)
    text, code = _status(
        monkeypatch, {path: fake}, slurm_paths={path})
    assert code == 0
    assert 'SCF: 6s' in text


## status() against the real converged fixture


def test_status_real_converged_fixture(monkeypatch, converged_vasp_dir) -> None:
    """A genuine converged run reports converged with a summary line."""
    monkeypatch.setattr(imdg_status, 'slurm_runningp', lambda path: False)
    console = Console(record=True, width=100)
    console._force_terminal = False
    monkeypatch.setattr(imdg_status, 'console', console)
    args = argparse.Namespace(
        dir=[str(converged_vasp_dir)], exclude=None, include=None,
        nowarn=None, problematic=False, skip_converged=False,
        skip_running=False, fast=False)
    assert status(args) == 0
    text = console.export_text()
    assert '1 runs: 0 running, 1 converged, 0 unconverged' in text


## _get_neb_summary zero-distance branch


def test_get_neb_summary_zero_distance_marked_red(tmp_path) -> None:
    """Identical images render a zero distance wrapped in a red tag."""
    (tmp_path / 'INCAR').write_text('IMAGES = 1' + chr(10))
    _write_structure(tmp_path / '00', x=0.1)
    _write_structure(tmp_path / '01', x=0.1)
    _write_structure(tmp_path / '02', x=0.1)

    summary = _get_neb_summary(IMDGVaspDir(str(tmp_path)))
    assert '[red]0.00' in summary


## status() edge cases


def _real_status(monkeypatch, dirs, **ns) -> tuple[str, int]:
    """Run status() through the real read_vaspdirs (no fake vaspdirs)."""
    monkeypatch.setattr(imdg_status, 'slurm_runningp', lambda path: False)
    console = Console(record=True, width=100)
    console._force_terminal = False
    monkeypatch.setattr(imdg_status, 'console', console)
    args = argparse.Namespace(**{
        'dir': [str(dirs)], 'exclude': None, 'include': None, 'nowarn': None,
        'problematic': False, 'skip_converged': False,
        'skip_running': False, 'fast': False, **ns,
    })
    code = status(args)
    return console.export_text(), code


def test_status_exclude_pattern(tmp_path, monkeypatch) -> None:
    """--exclude drops matching directories from the scan."""
    for name in ('keep_a', 'drop_b'):
        d = tmp_path / name
        d.mkdir()
        (d / 'POSCAR').write_text('x')
    text, code = _real_status(
        monkeypatch, tmp_path, exclude='drop_b')
    assert code == 0
    assert 'keep_a' in text
    assert 'drop_b' not in text


def test_status_include_pattern(tmp_path, monkeypatch) -> None:
    """--include keeps only matching directories."""
    for name in ('keep_a', 'drop_b'):
        d = tmp_path / name
        d.mkdir()
        (d / 'POSCAR').write_text('x')
    text, code = _real_status(
        monkeypatch, tmp_path, include='keep_a')
    assert code == 0
    assert 'keep_a' in text
    assert 'drop_b' not in text


def test_status_warning_mixed_messages(tmp_path, monkeypatch) -> None:
    """Two dirs sharing a warning render the list and skip empty messages."""
    a_path, a_fake = _status_dir(tmp_path, 'a', _FakeVaspdir(
        converged_electronic=False, converged_ionic=False,
        logs=[_FakeLog(VaspWarnings())],
        warnings=VaspWarnings([VaspWarningRecord('slurm_error', 'boom')]),
    ))
    b_path, b_fake = _status_dir(tmp_path, 'b', _FakeVaspdir(
        converged_electronic=False, converged_ionic=False,
        logs=[_FakeLog(VaspWarnings())],
        warnings=VaspWarnings([VaspWarningRecord('slurm_error', '')]),
    ))
    text, code = _status(monkeypatch, {a_path: a_fake, b_path: b_fake})
    assert code == 0
    assert 'slurm_error' in text
    assert 'boom' in text


def test_status_details_cpu_time(tmp_path, monkeypatch) -> None:
    """An unconverged run's CPU time is repeated in the Details block."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir(
        converged_electronic=False, converged_ionic=False,
        run_stats={'Total CPU time used (sec)': 60.0, 'cores': 4},
    ))
    text, code = _status(monkeypatch, {path: fake})
    assert code == 0
    assert 'CPU 0:01:00 (4 cores)' in text


def test_status_details_neb_unconverged(tmp_path, monkeypatch) -> None:
    """An unconverged NEB run prints its summary lines in Details."""
    path, fake = _status_dir(tmp_path, 'neb', _FakeVaspdir(
        nebp=True, converged_electronic=False, converged_ionic=False))
    monkeypatch.setattr(
        imdg_status, '_get_neb_summary',
        lambda vaspdir: '[magenta]line1[/magenta]' + chr(10) + '[magenta]line2[/magenta]')
    text, code = _status(monkeypatch, {path: fake})
    assert code == 0
    assert 'line1' in text
    assert 'line2' in text


def test_status_live_terminal(tmp_path, monkeypatch) -> None:
    """The terminal (Live) path renders a summary and returns 0."""
    path, fake = _status_dir(tmp_path, 'run', _FakeVaspdir())
    monkeypatch.setattr(
        imdg_status.IMDGVaspDir, 'read_vaspdirs',
        staticmethod(lambda rootpath=None, path_filter=None: {path: fake}))
    monkeypatch.setattr(imdg_status, 'slurm_runningp', lambda p: False)
    monkeypatch.setattr(
        imdg_status, 'vasp_output_time', lambda p: time.time())
    console = Console(record=True, width=100)
    console._force_terminal = True  # force the Live render path
    monkeypatch.setattr(imdg_status, 'console', console)
    args = argparse.Namespace(
        dir=['.'], exclude=None, include=None, nowarn=None,
        problematic=False, skip_converged=False, skip_running=False,
        fast=False)
    assert status(args) == 0
    assert '1 runs: 0 running, 1 converged, 0 unconverged' in (
        console.export_text())
