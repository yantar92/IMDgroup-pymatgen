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


"""sbatch config generator.
"""
import argparse
import os
import subprocess
import tomllib
from importlib import resources

from pytimeparse import parse as _timeparse
from pymatgen.io.vasp.inputs import Incar

from IMDgroup.gorun.slurm import barf_if_no_cmd


def read_example_config() -> str:
    """Return the bundled example gorun config as a string.

    The example is shipped as package data, so it is always the copy
    that matches the installed version (unlike a source-tree file or a
    URL, which can drift ahead of the code).
    """
    return (
        resources.files("IMDgroup.gorun")
        .joinpath("data", "gorun-config.example.toml")
        .read_text(encoding="utf-8")
    )


def _candidate_config_paths() -> list[str]:
    """Config paths to try, in priority order.

    ``--config PATH`` is handled separately in :func:`load_config` and
    takes precedence over everything.  Otherwise, a group-wide
    ``$IMDGroup`` checkout is preferred when present, and a per-user XDG
    location is the default for everyone else.
    """
    paths: list[str] = []
    if "IMDGroup" in os.environ:
        paths.append(
            os.path.join(os.environ["IMDGroup"], "dist", "etc", "gorun.toml"))
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if not xdg:
        xdg = os.path.join(os.path.expanduser("~"), ".config")
    paths.append(os.path.join(xdg, "gorun", "gorun.toml"))
    return paths


def _missing_config_message(searched: list[str]) -> str:
    """Short explanation plus a minimal example when no config exists."""
    return "\n".join([
        "No gorun configuration found.  Looked for:",
        *(f"  {path}" for path in searched),
        "",
        "Create ~/.config/gorun/gorun.toml.  A minimal single-cluster",
        "example (replace the hostname, queue, and module lines for your",
        "cluster hostname):",
        "",
        "  [cluster.names]",
        "  ares = ['login01.ares.cyfronet.pl']",
        "",
        "  [ares]",
        "  queues = ['plgrid']",
        '  VASP-setup = "module load intel/2023b imkl/2023.2.0"',
        "",
        "  [ares.defaults.sbatch]",
        "  nodes = 4",
        "  time = '72:00:00'",
        "  ntasks-per-node = 48",
        "",
        "  [ares.plgrid.sbatch]",
        "  partition = 'plgrid'",
        "",
        "For the full example (more clusters, GPU/MACE/ATAT sections),",
        "write the bundled copy to the default location:",
        "",
        "  gorun --example-config",
        "",
        "or pass an explicit path with --config PATH.",
    ])


def missing_config_message() -> str:
    """The "no config found" guidance string (public wrapper)."""
    return _missing_config_message(_candidate_config_paths())


def _resolve_config_path(args: argparse.Namespace) -> str | None:
    """Return the config path to read, or None when none is found.

    An explicit ``--config`` path must exist; a miss raises
    :class:`FileNotFoundError` (the path was requested explicitly, so a
    miss is an error).  Otherwise the search path is consulted and None
    is returned when nothing is found.
    """
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Cannot find config in {args.config}")
        return args.config
    for path in _candidate_config_paths():
        if os.path.exists(path):
            return path
    return None


def load_config(args: argparse.Namespace) -> dict | None:
    """Read the gorun config file, or None when none is found.

    An explicit ``--config`` that is missing raises
    :class:`FileNotFoundError`; a search path with no result returns
    None instead of printing and exiting.
    """
    path = _resolve_config_path(args)
    if path is None:
        return None
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_server(config: dict) -> str | None:
    """Return the current server name, or None when it cannot be resolved.

    None when *config* has no ``[cluster.names]`` table, when the current
    host maps to no server name, or when the mapped server has no section
    in *config*.
    """
    names = config.get("cluster", {}).get("names", {})
    if not names:
        return None
    server = current_server(config)
    if isinstance(config.get(server), dict):
        return server
    return None


def current_server(config: dict) -> str:
    """Get current server name, as named in the CONFIG.
    """
    if "CLUSTER_NAME" in os.environ:
        uname = os.environ['CLUSTER_NAME']
    else:
        barf_if_no_cmd('uname')
        uname = subprocess.check_output(
            'uname -n', shell=True).decode('utf-8').strip()
    for server, names in config['cluster']['names'].items():
        if uname in names:
            return server
    return uname


def parse_time_limit(value: str) -> str:
    """Normalize a time limit to Slurm ``HH:MM:SS``.

    Accepts Slurm's native formats (``HH:MM:SS``, ``MM:SS``, bare
    minutes, and ``days-hours`` variants) or a human-readable duration
    such as ``1h``, ``1d``, ``30m``, ``1h30m``, or ``1.5h``.  Anything
    :func:`pytimeparse.parse` recognizes is converted to canonical
    ``HH:MM:SS`` (days folded into hours); the rest is returned
    unchanged so Slurm applies its own validation (e.g. bare minutes).

    A value that carries a duration unit letter but does not parse
    (e.g. the typo ``1h30``) raises :class:`ValueError` instead of
    being silently forwarded to Slurm.
    """
    seconds = _timeparse(value)
    if seconds is None:
        if any(unit in value.lower() for unit in "dhms"):
            raise ValueError(
                f"invalid time limit {value!r}: expected HH:MM:SS or a "
                "duration like 1h, 1d, 30m, 1h30m"
            )
        return value
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_user_sbatch_args(script_args: argparse.Namespace) -> dict[str, str]:
    """Extract explicit sbtach arguments from SCRIPT_ARGS.
    """
    sbatch_args = {}
    if script_args.number_of_nodes is not None:
        sbatch_args['nodes'] = str(script_args.number_of_nodes)
    if script_args.time_limit is not None:
        sbatch_args['time'] = parse_time_limit(str(script_args.time_limit))
    return sbatch_args


def get_default_job_name():
    """Generate a sensible job name from INCAR or the current directory."""
    try:
        incar = Incar.from_file('INCAR')
        return {'job-name': incar.get('SYSTEM', 'unknown')}
    except (FileNotFoundError, OSError):
        # Fall back to directory name if INCAR is absent (e.g., MACE runs).
        return {'job-name': os.path.basename(os.getcwd())}


def get_sbatch_args(script_args: argparse.Namespace, config: dict,
                    server: str, queue: str,
                    adapter_name: str | None = None) -> dict:
    """Return a dict of arguments for sbatch command.
    Combine SCRIPT_ARGS with script CONFIG for QUEUE in SERVER.

    When ADAPTER_NAME is given, adapter-specific defaults
    (``[server.adapter_name.defaults.sbatch]``) are merged on top
    of the server-level defaults (``[server.defaults.sbatch]``).
    This lets MACE and VASP target different queues with different
    resource requests (GPU vs CPU) on the same cluster.
    """
    server_defaults = config[server].get('defaults', {}).get('sbatch', {})
    adapter_defaults = {}
    if adapter_name:
        adapter_defaults = (
            config[server].get(adapter_name, {})
            .get('defaults', {})
            .get('sbatch', {})
        )
    return (
        server_defaults
        | adapter_defaults
        | config[server][queue]['sbatch']
        | get_default_job_name()
        | get_user_sbatch_args(script_args)
        | {'partition': queue}
    )

