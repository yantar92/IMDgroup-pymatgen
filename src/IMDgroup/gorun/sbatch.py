# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-gorun package
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
import os
import sys
import subprocess
import tomllib
from termcolor import colored
from pymatgen.io.vasp.inputs import Incar
from IMDgroup.gorun.slurm import barf_if_no_cmd


def _candidate_config_paths() -> list[str]:
    """Config paths to try, in priority order.

    ``--config PATH`` is handled separately in :func:`get_config` and
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
    """Short explanation plus a starting point when no config exists."""
    return "\n".join([
        "No gorun configuration found.  Looked for:",
        *(f"  {path}" for path in searched),
        "",
        "Create one by copying the example and editing it for your cluster:",
        "",
        "  mkdir -p ~/.config/gorun",
        "  cp examples/gorun-config.example.toml ~/.config/gorun/gorun.toml",
        "",
        "or pass an explicit path with --config PATH.",
    ])


def get_config(args: dict) -> dict:
    """Read config file, if any.

    Resolution order: ``--config PATH`` (explicit) > ``$IMDGroup``
    checkout > ``$XDG_CONFIG_HOME/gorun/gorun.toml`` (defaulting to
    ``~/.config/gorun/gorun.toml``).
    """
    if args.config is not None:
        config_path = args.config
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        raise FileNotFoundError(f"Cannot find config in {config_path}")

    searched = _candidate_config_paths()
    for config_path in searched:
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                return tomllib.load(f)
    print(colored(_missing_config_message(searched), "red"))
    sys.exit(1)


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


def get_user_sbatch_args(script_args) -> dict[str, str]:
    """Extract explicit sbtach arguments from SCRIPT_ARGS.
    """
    sbatch_args = {}
    if script_args.number_of_nodes is not None:
        sbatch_args['nodes'] = str(script_args.number_of_nodes)
    if script_args.time_limit is not None:
        sbatch_args['time'] = str(script_args.time_limit)
    return sbatch_args


def get_default_job_name():
    """Generate a sensible job name from INCAR or the current directory."""
    try:
        incar = Incar.from_file('INCAR')
        return {'job-name': incar.get('SYSTEM', 'unknown')}
    except (FileNotFoundError, OSError):
        # Fall back to directory name if INCAR is absent (e.g., MACE runs).
        return {'job-name': os.path.basename(os.getcwd())}


def get_sbatch_args(script_args: dict, config: dict,
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

