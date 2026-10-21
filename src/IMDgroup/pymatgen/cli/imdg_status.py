# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
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


"""Check status of running VASP calculations."""
import argparse
import sys
import os
import re
import datetime
import logging
import warnings
import subprocess
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import cachetools.func
import numpy as np
from termcolor import colored
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning
from IMDgroup.pymatgen.io.vasp.diagnostics import VaspWarning, VaspWarnings
from IMDgroup.pymatgen.core.structure import structure_distance
from IMDgroup.pymatgen.io.vasp.outputs import (
    Outcar,
    RunTimingStats,
    Vasplog,
    read_outcar_timing_stats,
)
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
from IMDgroup.pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.cli.imdg_diff import print_incar_groups

logger = logging.getLogger(__name__)
console = Console()


def custom_showwarning(
        message, category, _filename, _lineno, file=None, _line=None):
    """Print warning in nicer way.

    Args:
        message: Warning message.
        category: Warning category class.
        _filename: File where the warning originated (unused).
        _lineno: Line number (unused).
        file: Output stream (default: stderr).
        _line: Line context (unused).
    """
    del _filename, _lineno, _line
    output = colored(
        f"{category.__name__}: ",
        "yellow",
        attrs=['bold']) + f'{message}'
    print(output, file=file or sys.stderr)


warnings.showwarning = custom_showwarning


@cachetools.func.ttl_cache(maxsize=None, ttl=10)
def _slurm_get_queue():
    """Get current slurm queue, as a list of strings.
    The returned list is from the output of squeue -O %Z command.
    """
    if shutil.which("squeue") is None:
        return []
    result = subprocess.check_output(
        "squeue -o %Z | tail -n +2",
        shell=True).split()
    return result


def slurm_runningp(path):
    """Check whether a Slurm job is running in the given directory.

    Args:
        path: Directory path to check.

    Returns:
        bool: True if a Slurm job is running in ``path`` or a
        parent NEB directory.
    """
    result = _slurm_get_queue()
    if len(result) > 0 and os.path.abspath(path) in [s.decode('utf-8') for s in result]:
        return True
    # For NEB and similar calculations, vasp might be running in parent dir.
    # Then, current directory must be named as a number and parent
    # INCAR should have IMAGES tag.
    # See https://www.vasp.at/wiki/index.php/IMAGES
    if re.match(r'[0-9]+', os.path.basename(path)):
        parent = os.path.dirname(path)
        if IMDGVaspDir(parent).nebp:
            return slurm_runningp(parent)
    return False


def add_args(parser):
    """Register subcommand arguments.

    Args:
        parser: Sub-parser from argparse.
    """
    parser.add_argument(
        "dir",
        help="""Directories to read (recusrively).  Defaults to current dir.""",
        type=str,
        nargs="*",
        default=["."])
    parser.add_argument(
        "--exclude",
        help="Dirs matching this Python regexp pattern will be excluded",
        type=str,
    )
    parser.add_argument(
        "--include",
        help="*Only* dirs matching this Python regexp pattern"
        " will be included",
        type=str,
    )
    parser.add_argument(
        "--nowarn",
        help="Warning names to ignore (names are printed in the "
        "'Warnings' summary)",
        nargs="+",
        default=None
    )
    parser.add_argument(
        "--problematic",
        help="Only show runs with warnings",
        action="store_true"
    )
    parser.add_argument(
        "--skip-converged",
        help="Do not check converged runs",
        action="store_true"
    )
    # Backward-compatible alias (undocumented).  `--skip_converged`
    # was renamed to `--skip-converged` in 1.2.0.
    parser.add_argument(
        "--skip_converged",
        dest="skip_converged",
        help=argparse.SUPPRESS,
        action="store_true",
    )
    parser.add_argument(
        "--skip-running",
        help="Do not check running runs",
        action="store_true"
    )
    parser.add_argument(
        "--fast",
        help="Do not parse OUTCAR to get run summary",
        action="store_true"
    )


def print_seconds(seconds):
    """Convert a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.  Negative values produce
            relative past-time strings.

    Returns:
        str: Human-readable duration (e.g. ``"in 2h 30m"``).
    """
    if seconds == 0:
        return "now"
    negative = seconds < 0
    seconds = int(abs(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days == 0 and hours == 0 and minutes == 0:
        return "now"
    output = []
    if not negative:
        output.append("in")
    if days != 0:
        output.append(f"{days}d")
    if hours != 0:
        output.append(f"{hours}h")
    if minutes != 0:
        output.append(f"{minutes}m")
    # if seconds != 0:
    #     output.append(f"{seconds}s")
    if negative:
        output.append("ago")
    return " ".join(output)


def format_duration(seconds):
    """Format a duration in seconds compactly.

    Rounds to the most significant non-zero unit so the string stays
    short in table cells: ``"6s"``, ``"15m"``, ``"1h"``.

    Args:
        seconds: Duration in seconds, or NaN for undefined.

    Returns:
        str: Human-readable duration, or ``"N/A"`` when seconds is NaN.
    """
    if seconds is None or np.isnan(seconds):
        return "N/A"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    return f"{seconds / 3600:.0f}h"


def format_timing_stats(stats: RunTimingStats) -> str:
    """Format RunTimingStats for display.

    Omits the ``+/-`` std term when fewer than two samples are
    available, and returns an empty string when no SCF cycle was
    recorded.

    Args:
        stats: RunTimingStats instance.

    Returns:
        str: e.g. ``"SCF: 6.1±2.5s relax: 14m51s"``.
    """
    if stats.scf.n == 0:
        return ""

    def fmt(mean_std):
        mean = format_duration(mean_std.mean)
        if mean_std.n < 2 or np.isnan(mean_std.std):
            return mean
        return f"{mean}±{format_duration(mean_std.std)}"

    return f"SCF: {fmt(stats.scf)} relax: {fmt(stats.ionic)}"


def vasp_output_time(path):
    """Return last VASP output modification time.

    Args:
        path: VASP directory path.

    Returns:
        float | None: Modification timestamp, or None if no VASP
        output file is found.  For NEB calculations, the maximum
        timestamp across all image directories is returned.
    """
    vaspdir = IMDGVaspDir(path)
    if vaspdir.nebp:
        neb_dirs = vaspdir.neb_dirs()
        assert neb_dirs is not None
        times = [vasp_output_time(d.path) for d in neb_dirs]
        times = [x for x in times if x is not None]
        return None if len(times) == 0 else max(times)
    outcar = os.path.join(path, 'OUTCAR')
    if os.path.isfile(outcar):
        return os.path.getmtime(outcar)
    # ATAT moves away OUTCAR and other files but not vasprun.xml
    vasprun = os.path.join(path, 'vasprun.xml')
    if os.path.isfile(vasprun):
        return os.path.getmtime(vasprun)
    return None


def _collect_warnings(
        warnings: VaspWarnings,
        ignore_list: None | list[str] = None
        ) -> tuple[set[str], dict[str, tuple[str, list[str]]]]:
    """Collect warning names and their message/tips metadata.

    Args:
        warnings: Structured warning container for one run.
        ignore_list: Warning names to skip.

    Returns:
        A tuple ``(names, meta)`` where ``names`` is the set of warning
        names present (after ignoring) and ``meta`` maps each name to a
        ``(message, tips)`` pair for the summary block.
    """
    names: set[str] = set()
    meta: dict[str, tuple[str, list[str]]] = {}
    for warn_name, record in warnings.items():
        if ignore_list is not None and warn_name in ignore_list:
            continue
        names.add(warn_name)
        if warn_name not in meta:
            meta[warn_name] = (record.message, list(record.tips))
    return names, meta


def _get_progress(logs: list[Vasplog | Outcar]) -> str:
    """Get formatted progress string from list of LOGS.
    """
    progress_data = logs[-1].progress
    if len(progress_data.values()) > 0:
        return list(progress_data.values())[-1].message
    return "N/A"


def _get_neb_summary(vaspdir: IMDGVaspDir) -> str:
    """Return a rich-markup string summarising NEB image distances."""
    neb_dirs = vaspdir.neb_dirs()
    assert neb_dirs is not None
    neb_structures = []
    neb_structures_initial = []
    for nebimagedir in neb_dirs:
        neb_structures.append(
            nebimagedir.final_structure or nebimagedir.initial_structure)
        neb_structures_initial.append(nebimagedir.initial_structure)

    def get_dists(structs):
        dists = [
            structure_distance(str1, str2, tol=0)
            for str1, str2 in zip(structs, structs[1:])]
        parts = []
        for idx, dist in enumerate(dists):
            value = f"{dist:.2f}\u00c5"
            if np.isclose(dist, 0):
                value = f"[red]{value}[/red]"
            parts.append(f"{idx + 1:02d}: {value}")
        return " ".join(parts)

    return (
        "[magenta]IMAGE DISTANCES (initial) [/magenta]"
        + get_dists(neb_structures_initial) + "\n"
        + "[magenta]IMAGE DISTANCES           [/magenta]"
        + get_dists(neb_structures))


@dataclass
class _RunRow:
    """Structured display data for a single VASP run."""
    dir_full: str
    dir_display: str
    truncated: bool
    ref: int
    when: str
    running: bool
    converged: bool
    is_neb: bool
    status_label: str
    status_color: str
    warn_names: tuple[str, ...]
    energy: str
    fmax: str
    per_step: str
    progress: str
    cpu_time: str
    n_cores: str
    neb_summary: str


_RICH_TAG_RE = re.compile(r"\[/?[a-z]+?\]")


def _strip_markup(text: str) -> str:
    """Remove simple rich color tags (e.g. ``[red]``) from TEXT."""
    return _RICH_TAG_RE.sub("", text)


def _details_text(row: _RunRow) -> str:
    """One-line plain-text details for the table column.

    Combines the running progress message, CPU time, and NEB image
    distances (markup stripped) into a single ``|``-separated string.
    """
    parts = []
    if row.progress:
        parts.append(row.progress)
    if row.cpu_time:
        cores = f" ({row.n_cores} cores)" if row.n_cores else ""
        parts.append(f"CPU {row.cpu_time}{cores}")
    if row.neb_summary:
        parts.append(_strip_markup(row.neb_summary))
    return " | ".join(parts)


def _add_run_row(table, row, has_ref):
    """Append one run's display cells to TABLE.

    Args:
        table: rich Table with the status columns already added.
        row: _RunRow to render.
        has_ref: Whether the table has the trailing footnote-ref column.
    """
    if row.warn_names:
        status_cell = Text(row.status_label, style="bold black on yellow")
    else:
        status_cell = Text(row.status_label, style=row.status_color)
    dir_cell = Text()
    if row.is_neb:
        dir_cell.append("IMAGES ", style="magenta")
    dir_cell.append(row.dir_display)
    warn_cell = (Text(", ".join(row.warn_names), style="yellow")
                 if row.warn_names else Text("-", style="dim"))
    cells = [
        row.when,
        status_cell,
        dir_cell,
        row.energy or "-",
        row.fmax or "-",
        row.per_step or "-",
        warn_cell,
        _details_text(row) or "-",
    ]
    if has_ref:
        cells.append(
            Text(f"[{row.ref}]", style="dim") if row.truncated else Text())
    table.add_row(*cells)


def _build_status_table(rows, any_truncated, dir_max):
    """Build a fresh status table holding the given ROWS.

    The table is rebuilt on every live update so it only ever contains
    the most recent ``rows`` (typically the last N scanned), keeping the
    render short enough for ``Live`` to redraw in place.

    Args:
        rows: Iterable of _RunRow to render.
        any_truncated: Whether the trailing ``#`` footnote column is
            needed (any scanned path exceeds ``dir_max``).
        dir_max: Maximum width of the DIR column in cells.

    Returns:
        Table: A populated rich Table.
    """
    table = Table(
        box=None, show_header=True, header_style="bold", padding=(0, 1))
    table.add_column("WHEN", justify="right", style="dim", no_wrap=True)
    table.add_column("STATUS", no_wrap=True)
    table.add_column(
        "DIR", max_width=dir_max, overflow="ellipsis", no_wrap=True)
    table.add_column("ENERGY (eV)", justify="right", no_wrap=True)
    table.add_column("FMAX (eV/\u00c5)", justify="right", no_wrap=True)
    table.add_column("PER-STEP", no_wrap=True)
    table.add_column("WARN", overflow="fold")
    table.add_column("DETAILS", overflow="ellipsis", no_wrap=True, max_width=20)
    if any_truncated:
        table.add_column("#", justify="right", no_wrap=True)
    for row in rows:
        _add_run_row(table, row, any_truncated)
    return table


def status(args):
    """Run the status subcommand.

    Args:
        args: Parsed command-line arguments from argparse.

    Returns:
        int: Exit code (0 on success).
    """
    def include_dirp(p):
        p = str(p)
        if args.exclude is not None and re.search(args.exclude, p):
            return False
        if args.include is not None and not re.search(args.include, p):
            return False
        return True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore", category=UnconvergedVASPWarning, append=True)
        warnings.filterwarnings(
            "ignore", category=VaspWarning, append=True)

        vaspdirs = IMDGVaspDir.read_vaspdirs(
            args.dir, path_filter=include_dirp)

    paths_no_output = []
    for wdir in vaspdirs:
        has_slurm = False
        # Assume that vaspdir is valid as long as it has slurm logs
        # Examples: ATAT input dir, top-level NEB input.
        for f in Path(wdir).iterdir():
            if re.match('slurm-[0-9]+.out', f.name):
                has_slurm = True
                break
        has_outcar = (Path(wdir) / 'OUTCAR').is_file()
        if not (has_slurm or has_outcar):
            paths_no_output.append(wdir)
    paths = sorted(vaspdirs)
    logger.debug("Found VASP dirs: %s", paths)
    if paths_no_output:
        console.print(
            "[yellow]Directories containing VASP input but not output:[/yellow]")
        for wdir in sorted(paths_no_output):
            console.print(Text.assemble(("  ", ""), (str(wdir), "")))

    incars = []
    # warning name -> category (running/converged/unconverged) -> dirs
    dirs_with_warnings: dict[str, dict[str, list[str]]] = {}
    warning_meta: dict[str, tuple[str, list[str]]] = {}
    rows: list[_RunRow] = []

    dir_max = max(30, min(48, console.width - 70))
    ref_counter = 0

    # Decide the footnote-ref column before streaming: a scanned path
    # will be ellipsized when it exceeds dir_max, so we know upfront
    # whether the column is needed (skipped dirs may leave it empty).
    any_truncated = any(
        len(w.replace("./", "")) > dir_max for w in paths)

    # Keep only the last N rows in the live render so the table stays
    # short and Live redraws in place (auto-scroll) instead of growing
    # past the terminal height.
    visible_max = max(10, console.height or 40)
    visible_rows: deque[_RunRow] = deque(maxlen=visible_max)

    if console.is_terminal:
        live = Live(
            Group(
                Text("Reading VASP outputs...", style="dim"),
                _build_status_table([], any_truncated, dir_max)),
            console=console,
            refresh_per_second=4)
        live.start()
    else:
        live = None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=VaspWarning)
            for i, wdir in enumerate(paths, 1):
                vaspdir = vaspdirs.get(wdir)
                assert vaspdir is not None
                incar = vaspdir['INCAR']
                if incar is not None:
                    incar['SYSTEM'] = wdir
                    incars.append(incar)
                # Avoid overflowing memory when reading too many dirs.
                del vaspdirs[wdir]
                nebp = vaspdir.nebp

                if slurm_runningp(wdir):
                    running, converged = True, False
                    status_label, status_color = "running", "yellow"
                else:
                    running = False
                    converged = (vaspdir.converged_electronic
                                 and vaspdir.converged_ionic
                                 and vaspdir.converged_manual)
                    if converged:
                        status_label = (
                            "converged; next step pending"
                            if not vaspdir.converged_sequence
                            else "converged")
                        status_color = "green"
                    else:
                        status_label, status_color = "unconverged", "red"

                logger.debug(
                    '%s: running = %s, converged = %s',
                    wdir, running, converged)

                if args.skip_running and running:
                    logger.debug('skipping running run')
                    continue

                if args.skip_converged and converged:
                    logger.debug('skipping converged run')
                    continue

                progress = ""
                warn_names: set[str] = set()
                if logs := (not nebp) and vaspdir.logs():
                    logger.debug(
                        "Found VASP logs in %s: %s",
                        wdir, [log.file.name for log in logs])
                    if not converged:
                        progress = _get_progress(logs)
                    nowarn_list = ['unconverged_energy'] if running else []
                    if args.nowarn:
                        nowarn_list += args.nowarn
                    warn_names, meta = _collect_warnings(
                        vaspdir.warnings, nowarn_list)
                    for name, pair in meta.items():
                        warning_meta.setdefault(name, pair)
                    dir_norm = wdir.replace("./", "")
                    if running:
                        category = "running"
                    elif converged:
                        category = "converged"
                    else:
                        category = "unconverged"
                    for warn_name in warn_names:
                        dirs_with_warnings.setdefault(
                            warn_name, {}).setdefault(
                                category, []).append(dir_norm)
                    if not warn_names and category == "unconverged":
                        dirs_with_warnings.setdefault(
                            'UNCONVERGED', {}).setdefault(
                                category, []).append(dir_norm)
                else:
                    if not nebp:
                        logger.debug("Slurm log file not found in %s", wdir)

                per_step = ""
                if running and not nebp and not args.fast:
                    outcar_path = Path(wdir) / 'OUTCAR'
                    if outcar_path.is_file():
                        per_step = format_timing_stats(
                            read_outcar_timing_stats(outcar_path))

                if args.problematic and not warn_names\
                        and (converged or running):
                    continue

                final_energy_str = ""
                fmax_str = ""
                cpu_time = ""
                n_cores = ""
                if not running:
                    if vaspdir['vasprun.xml'] is None and\
                       (Path(wdir) / 'vasprun.xml').is_file():
                        status_label = "incomplete vasprun.xml"
                        status_color = "red"
                    final_energy = vaspdir.final_energy
                    if final_energy is None or final_energy == np.nan:
                        final_energy_str = ""
                    else:
                        final_energy_str = "" if np.isnan(final_energy)\
                            else f"{final_energy:.4f}"
                        outcar = None if args.fast else vaspdir['OUTCAR']
                        if outcar is not None:
                            cpu_time_sec = outcar.run_stats.get(
                                'Total CPU time used (sec)')
                            if cpu_time_sec is not None:
                                cpu_time = str(datetime.timedelta(
                                    seconds=round(cpu_time_sec)))
                            cores = outcar.run_stats['cores']
                            if cores is not None:
                                n_cores = str(cores)
                            max_force = vaspdir.max_force()
                            if max_force is not None:
                                fmax_str = f"{max_force:.4f}"
                            per_step = format_timing_stats(
                                outcar.timing_stats)

                mtime = vasp_output_time(wdir)
                if mtime is None:
                    continue
                delta = mtime - datetime.datetime.now().timestamp()

                dir_full = wdir.replace("./", "")
                truncated = len(dir_full) > dir_max
                ref = 0
                if truncated:
                    ref_counter += 1
                    ref = ref_counter

                row = _RunRow(
                    dir_full=dir_full,
                    dir_display=dir_full,
                    truncated=truncated,
                    ref=ref,
                    when=print_seconds(delta),
                    running=running,
                    converged=converged,
                    is_neb=nebp,
                    status_label=status_label,
                    status_color=status_color,
                    warn_names=tuple(sorted(warn_names)),
                    energy=final_energy_str,
                    fmax=fmax_str,
                    per_step=per_step,
                    progress=progress,
                    cpu_time=cpu_time,
                    n_cores=n_cores,
                    neb_summary=_get_neb_summary(vaspdir) if nebp else "",
                )
                rows.append(row)
                visible_rows.append(row)
                if live is not None:
                    live.update(Group(
                        Text(
                            f"Reading VASP outputs: {i}/{len(paths)}",
                            style="dim"),
                        _build_status_table(
                            visible_rows, any_truncated, dir_max)))

        n_running = sum(1 for r in rows if r.running)
        n_converged = sum(1 for r in rows if r.converged)
        n_other = len(rows) - n_running - n_converged
        summary = Text.assemble(
            (f"{len(rows)} runs: ", "bold"),
            (f"{n_running} running, ", "yellow"),
            (f"{n_converged} converged, ", "green"),
            (f"{n_other} unconverged", "red"))

        # Final frame: swap the progress line for the summary and render
        # the complete table so the terminal ends with every row visible.
        if live is not None:
            live.update(Group(
                summary,
                _build_status_table(rows, any_truncated, dir_max)))

    finally:
        if live is not None:
            live.stop()

    if live is None:
        # Non-terminal output: emit the full table and summary once so
        # piping carries every row (Live would otherwise only print the
        # final auto-scroll window).
        if rows:
            console.print(_build_status_table(rows, any_truncated, dir_max))
        console.print()
        console.print(summary)

    truncated_rows = [r for r in rows if r.truncated]
    if truncated_rows:
        console.print()
        console.print("[bold]Paths:[/bold]")
        for r in truncated_rows:
            console.print(Text.assemble(
                ("  ", ""), (f"[{r.ref}] ", "dim"), (r.dir_full, "")))

    if dirs_with_warnings:
        category_order = ("unconverged", "running", "converged")
        console.print()
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warn_name, by_category in dirs_with_warnings.items():
            msg, tips = warning_meta.get(warn_name, ("", []))
            for category in category_order:
                dir_list = by_category.get(category)
                if not dir_list:
                    continue
                tag = "" if warn_name == "UNCONVERGED" else f" [{category}]"
                console.print(Text.assemble(
                    ("  ", ""),
                    (warn_name, "yellow"),
                    (f"{tag} ({len(dir_list)}): ", ""),
                    (" ".join(dir_list), "")))
            if msg:
                console.print(Text.assemble(("      ", ""), (str(msg), "")))
            for tip in tips:
                console.print(Text.assemble(
                    ("      TIP: ", "magenta bold"), (str(tip), "")))

    # Full details are only useful for runs still in flight or that
    # failed; converged runs would only repeat the CPU time already
    # shown in the truncated DETAILS column.
    detail_rows = [
        r for r in rows
        if not r.converged and (r.progress or r.cpu_time or r.neb_summary)]
    if detail_rows:
        console.print()
        console.print("[bold]Details:[/bold]")
        for r in detail_rows:
            head = Text("  ")
            head.append(r.dir_full, style="bold")
            if r.truncated:
                head.append(f" [{r.ref}]", style="dim")
            console.print(head)
            if r.progress:
                console.print(Text.assemble(("      ", ""), (r.progress, "")))
            if r.cpu_time:
                cores = f" ({r.n_cores} cores)" if r.n_cores else ""
                console.print(Text.assemble(
                    ("      ", ""), (f"CPU {r.cpu_time}{cores}", "")))
            if r.neb_summary:
                for line in r.neb_summary.split("\n"):
                    console.print(Text.from_markup("      " + line))

    common_incar, grouped_incars = Incar.group_incars(incars)
    if common_incar is not None and len(grouped_incars) > 1:
        console.print(Text("Multiple INCARs found: ", style="yellow"))
        print_incar_groups(
            common_incar, grouped_incars, console=console, max_names=3)

    return 0
