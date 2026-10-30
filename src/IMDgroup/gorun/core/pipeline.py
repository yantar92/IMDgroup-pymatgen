# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
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

"""Common dispatch pipeline shared by all gorun backends."""

from __future__ import annotations

import os
import glob
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from termcolor import colored

from IMDgroup.gorun.slurm import (
    barf_if_no_cmd,
    directory_queued_p,
    user_job_count,
    clear_slurm_logs,
    get_best_script,
)
from IMDgroup.gorun.sbatch import (
    load_config,
    resolve_server,
    missing_config_message,
    get_sbatch_args,
)
from IMDgroup.gorun.adapters.base import SoftwareAdapter

# ---------------------------------------------------------------------------
# RUNNING-file management snippets inserted around every run script
# ---------------------------------------------------------------------------

_RUNNING_SETUP = """\
cleanup(){ rm -f RUNNING; }
touch "RUNNING"
# start a background monitor that will delete the flag when the parent dies
{ while ps -o pid= -p $$ >/dev/null 2>&1; do sleep 30; done; cleanup; } &
monitor_pid=$!
# We set up SLURM to send USR1 (and scancel sends TERM by default) so
# the job can clean up before it is killed.  Bash only runs traps
# while it is waiting on a command via `wait`, not while a foreground
# command is running, so the computation body below is launched in the
# background and awaited.
_signal_exit(){ cleanup; kill $monitor_pid 2>/dev/null; exit $1; }
trap '_signal_exit 138' USR1
trap '_signal_exit 143' TERM
"""

_RUNNING_CLEANUP = """\
cleanup
kill $monitor_pid 2>/dev/null
"""

# ---------------------------------------------------------------------------
# Backup helpers
# ---------------------------------------------------------------------------

_BACKUP_PREFIX = "gorun"


def _last_run_number(path: Path) -> int | None:
    """Return the highest *gorun_N* folder number under *path*."""
    existing = glob.glob(str(path / (f"{_BACKUP_PREFIX}_*")))
    numbers = [
        int(f.split("_")[-1].rstrip(".tar.gz"))
        for f in existing
        if f.split("_")[-1].split(".")[0].isdigit()
    ]
    return max(numbers) if numbers else None


def _next_run_folder(path: Path) -> str:
    """Return the next *gorun_N* folder name."""
    last = _last_run_number(path)
    return f"{_BACKUP_PREFIX}_{(last or 0) + 1}"


def _rsync_exclude_args(excludes: list[str]) -> str:
    """Build the rsync ``--exclude`` argument string.

    ``gorun_*`` is always excluded first so previous backups are not
    copied recursively; adapter-specific *excludes* follow.
    """
    all_excludes = ["gorun_*"] + list(excludes)
    return " ".join(f"--exclude '{pat}'" for pat in all_excludes)


def _backup_current_dir(path: Path, excludes: list[str]) -> None:
    """Backup *path* to the next *gorun_N* subdirectory.

    Previous backup (except *gorun_1*) is compressed to a tarball.
    """
    barf_if_no_cmd("rsync")
    print(f"Backing up {path}")

    previous_num = _last_run_number(path)
    if previous_num is not None:
        previous_dir = path / f"{_BACKUP_PREFIX}_{previous_num}"
        if previous_num != 1 and previous_dir.is_dir():
            print(f"Compressing previous run: {previous_dir}")
            tar_path = path / f"{previous_dir.name}.tar.gz"
            with tarfile.open(tar_path, mode="w:gz") as tar:
                tar.add(str(previous_dir), arcname=previous_dir.name)
            shutil.rmtree(str(previous_dir))

    gorun_ready = path / "gorun_ready"
    if gorun_ready.is_file():
        print("Found gorun_ready.  Deleting.")
        gorun_ready.unlink()

    to = path / _next_run_folder(path)
    # Always exclude gorun's own backup directories and tarballs
    # to avoid recursive backups.  Adapter-specific patterns
    # (e.g. WAVECAR) are appended on top.
    exclude_args = _rsync_exclude_args(excludes)
    subprocess.check_call(
        f"rsync -avq {exclude_args} '{path}/' '{to}/'",
        shell=True,
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def dispatch_run(args, adapter: SoftwareAdapter) -> int:
    """Run the common submission pipeline for *adapter*.

    Parameters
    ----------
    args:
        argparse ``Namespace`` with at least these attributes:

        - ``force`` : ``bool``
        - ``local`` : ``bool``
        - ``mark``  : ``bool``
        - ``max_slurm_jobs`` : ``int``
        - ``keep``  : ``list[str] | None``
        - ``queue`` : ``str | None``
        - ``number_of_nodes`` : ``int | None``
        - ``time_limit``      : ``str | None``
        - ``config``          : ``str | None``
    adapter:
        Backend implementation; see ``SoftwareAdapter``.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    # ---- 1. Load config and resolve the server section ----
    config = load_config(args)

    if config is None:
        if not args.local:
            print(colored(missing_config_message(), "red"))
            return 1
        server = None
        server_config: dict = {}
    else:
        server = resolve_server(config)
        if server is not None:
            server_config = config[server]
        elif args.local:
            server = None
            server_config = {}
        else:
            print(colored("Running on unknown server.  Adjust the config.", "red"))
            return 1

    # ---- 2. Validate environment ----
    adapter.validate_environment()

    cwd = Path.cwd()

    # ---- 3. Guards ----
    if Path("RUNNING").is_file() and not args.force:
        print(
            colored(
                "A job is already running in this directory.  Exiting.",
                "yellow",
            )
        )
        return 0

    if not args.local and directory_queued_p(str(cwd)) and not args.force:
        print(
            colored(
                "A job is already queued for this directory.  Exiting.",
                "yellow",
            )
        )
        return 0

    if Path("gorun_ready").is_file() and not args.force:
        print(
            colored(
                "gorun_ready file present.  Exiting without submitting.",
                "yellow",
            )
        )
        return 0

    if not adapter.is_valid_input(cwd):
        print(colored("No valid input found in this directory.  Exiting.", "yellow"))
        return 1

    if adapter.is_converged(cwd) and not args.force:
        print(colored("Already converged.  Exiting.", "yellow"))
        return 1

    # ---- 4. Throttle ----
    if args.max_slurm_jobs > 0 and not args.local:
        while user_job_count() >= args.max_slurm_jobs:
            print("Waiting for submitted jobs to finish")
            time.sleep(10)

    # ---- 5. Backup ----
    if adapter.has_previous_output(cwd):
        _backup_current_dir(cwd, excludes=adapter.backup_excludes())

    # ---- 6. Prepare inputs ----
    keep = set(args.keep) if args.keep else set()
    adapter.prepare_inputs(cwd, keep=keep, force=args.force)

    # ---- 7. Clear old slurm logs ----
    log_file = getattr(adapter, "log_file", None)
    clear_slurm_logs(str(cwd), extra_logs=[log_file] if log_file else None)

    # ---- 8. Generate run script ----
    # When the parent process has already configured the environment
    # (e.g. maps spawning VASP via pollmach), the child gorun can
    # inherit GORUN_SKIP_SETUP=1 to avoid re-running module loads
    # and exports.
    if os.environ.get("GORUN_SKIP_SETUP") == "1":
        setup = ""
        print(colored("GORUN_SKIP_SETUP is set.  Skipping environment setup.", "yellow"))
    else:
        setup = adapter.setup_commands(server_config)
    body = adapter.generate_run_commands(cwd, server_config)

    # Capture the computation's exit status before the cleanup step (which
    # must run regardless) and re-exit with it.  The body runs in the
    # background so the USR1/TERM traps above fire immediately while the
    # computation is still running (see _RUNNING_SETUP).
    script_body = (
        f"{setup}\n{_RUNNING_SETUP}\n"
        f"{{\n{body}\n}} &\n"
        "_gorun_pid=$!\n"
        "wait $_gorun_pid\n"
        f"_gorun_status=$?\n{_RUNNING_CLEANUP}\nexit $_gorun_status"
    )

    if args.local:
        shebang = server_config.get("shebang", "#!/usr/bin/env bash")
        script = f"{shebang}\n{script_body}"
    else:
        adapter_cfg = server_config.get(adapter.name, {})
        queues = (
            [args.queue]
            if args.queue
            else adapter_cfg.get("queues")
            or server_config.get("queues", [])
        )
        if not queues:
            print(colored("No queues configured for this server.", "red"))
            return 1
        alt_args = [
            get_sbatch_args(args, config, server, queue_name, adapter.name)
            for queue_name in queues
        ]
        shebang = server_config.get("shebang", "#!/usr/bin/env bash")
        script = get_best_script(alt_args, script_body, shebang)

    # ---- 9. Submit ----
    if args.mark:
        with open("sub", "w", encoding="utf-8") as f:
            f.write(script)
        Path("gorun_ready").touch()
        print(
            colored(
                'Created "gorun_ready" file.  Submit manually or via gorun-all-ready.sh.',
                "green",
            )
        )
    elif args.local:
        log_name = getattr(adapter, "log_file", "gorun.out")
        print(colored("Running job locally...", "green"))
        with open(log_name, "a", encoding="utf-8") as f:
            result = subprocess.run(
                script,
                shell=True,
                stdout=f,
                stderr=f,
                text=True,
            )
        if result.returncode != 0:
            print(
                colored(
                    f"Local run failed with exit status {result.returncode}.  "
                    f"See {log_name} for details.",
                    "red",
                )
            )
            return 1
    else:
        with open("sub", "w", encoding="utf-8") as f:
            f.write(script)
        status = os.WEXITSTATUS(os.system("sbatch sub"))
        if status == 0:
            print(colored("Job submitted to SLURM scheduler.", "green"))
        else:
            print(colored("Failed to submit to SLURM scheduler.", "red"))
            return 1

    return 0
