"""Check status of running VASP calculations.
"""
import sys
import os
import re
import datetime
import logging
import warnings
import subprocess
import shutil
from pathlib import Path
import cachetools.func
import numpy as np
from termcolor import colored
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning
from IMDgroup.pymatgen.core.structure import structure_distance
from IMDgroup.pymatgen.io.vasp.outputs import Vasplog
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

logger = logging.getLogger(__name__)


def custom_showwarning(
        message, category, _filename, _lineno, file=None, _line=None):
    """Print warning in nicer way.
    """
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
        return False
    result = subprocess.check_output(
        "squeue -o %Z | tail -n +2",
        shell=True).split()
    return result


def slurm_runningp(path):
    """Is slurm running in DIR?
    """
    result = _slurm_get_queue()
    if os.path.abspath(path) in [s.decode('utf-8') for s in result]:
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
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Report running VASP status."""

    parser.add_argument(
        "dir",
        help="""Directory to read (recusrively).  Defaults to current dir.""",
        type=str,
        nargs="?",
        default=".")
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
        help="List of warnings to ignore",
        nargs="+",
        choices=[key for key in Vasplog.VASP_WARNINGS
                 if '__' not in key],
        default=None
    )
    parser.add_argument(
        "--problematic",
        help="Only show runs with warnings",
        action="store_true"
    )


def print_seconds(seconds):
    """Print SECONDS in human-readable form.
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


def vasp_output_time(path):
    """Return last VASP output modification time in PATH.
    If no VASP output is found, return None.
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


def _get_warning_list(
        logs: list[Vasplog],
        ignore_list: None | list[str] = None) -> tuple[str, set[str]]:
    """Get warning list from list of LOGS.
    Return a tuple (formatted_warning_list_string, warning_types_list)
    """
    warning_list = ""
    all_warn_names_present = set()
    for log in logs:
        for warn_name, data in log.warnings.items():
            if ignore_list is not None and warn_name in ignore_list:
                continue
            all_warn_names_present.add(warn_name)
            warning_list += "\n" +\
                colored(
                    f"⮤Warning ({data['count']}x) {warn_name}: ",
                    "yellow") + data['message']
            if tips := data.get('tips'):
                for tip in tips:
                    warning_list += '\n' + colored(
                        " ➙ TIP: ", "magenta", attrs=['bold'])\
                        + tip
    return (warning_list, all_warn_names_present)


def _get_progress(logs: list[Vasplog]) -> str:
    """Get formatted progress string from list of LOGS.
    """
    progress_data = logs[-1].progress
    if len(progress_data.values()) > 0:
        return list(progress_data.values())[-1]['message']
    return "N/A"


def _get_run_prefix(vaspdir: IMDGVaspDir) -> str:
    """Get prefix to be displayed for VASPDIR run.
    """
    if vaspdir.nebp:
        # NEB-like calculation
        return colored("IMAGES ", "magenta")
    return ""


def _get_neb_summary(vaspdir: IMDGVaspDir) -> str:
    """Get summary of NEB calculation.
    """
    neb_structures = []
    neb_structures_initial = []
    for nebimagedir in vaspdir.neb_dirs():
        contcar_struct = nebimagedir.structure
        poscar_struct = nebimagedir.initial_structure
        neb_structures.append(contcar_struct or poscar_struct)
        neb_structures_initial.append(poscar_struct)

    def get_dists(structs):
        dists = [
            structure_distance(str1, str2, tol=0)
            for str1, str2 in zip(structs, structs[1:])
        ]
        return [f"{idx+1:02d}: " +
                colored(f"{dist:.2f}Å",
                        "red" if np.isclose(dist, 0) else "white")
                for idx, dist in enumerate(dists)]
    neb_dists_initial =\
        colored("IMAGE DISTANCES (initial) ", "magenta")\
        + " ".join(get_dists(neb_structures_initial))
    neb_dists =\
        colored("IMAGE DISTANCES           ", "magenta")\
        + " ".join(get_dists(neb_structures))
    return neb_dists_initial + '\n' + neb_dists


def status(args):
    """Main routine.
    """
    def exclude_dirp(p):
        if args.exclude is not None and re.search(args.exclude, p):
            return True
        if args.include is not None and not re.search(args.include, p):
            return True
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings(
            "ignore", category=UnconvergedVASPWarning, append=True)

        vaspdirs = IMDGVaspDir.read_vaspdirs(
            args.dir, path_filter=exclude_dirp)

    paths = []
    paths_no_output = []
    for wdir in vaspdirs:
        has_slurm = False
        for f in Path(wdir).iterdir():
            if re.match('slurm-[0-9]+.out', f.name):
                paths.append(wdir)
                has_slurm = True
                break
        if not has_slurm and any(
                (Path(wdir) / f).is_file()
                for f in ['INCAR', 'POSCAR',
                          'KPOINTS', 'POTCAR']):
            paths_no_output.append(wdir)
    paths = sorted(paths)
    if len(paths_no_output) > 0:
        print(colored(
            "Directories containing VASP input but not output:",
            "yellow"
        ))
        for wdir in paths_no_output:
            print("  ", wdir)
    all_warn_names_present = set()
    for wdir in paths:
        vaspdir = vaspdirs.get(wdir)
        nebp = vaspdir.nebp
        converged = vaspdir.converged
        if logs := Vasplog.from_dir(wdir) and not nebp:
            logger.debug(
                "Found VASP logs in %s: %s",
                wdir, [log.file.name for log in logs])
            if not converged:
                progress = _get_progress(logs)
            else:
                progress = ""
            warning_list, warn_names = _get_warning_list(logs, args.nowarn)
            all_warn_names_present = all_warn_names_present.union(warn_names)
        else:
            if not nebp:
                logger.debug("Slurm log file not found in %s", wdir)
            progress = ""
            warning_list = ""

        run_status = colored("unknown", "red")
        converged = None
        running = False
        run_prefix = _get_run_prefix(vaspdir)

        if slurm_runningp(wdir):
            running = True
            run_status = colored("running", "yellow")
        else:
            run_status = colored("converged", "green") if converged\
                else colored("unconverged", "red")
            if vaspdir['vasprun.xml'] is None and\
               (Path(wdir) / 'vasprun.xml').is_file():
                run_status = colored("incomplete vasprun.xml", "red")
            outcar = vaspdir['OUTCAR']
            if outcar is not None:
                cpu_time_sec =\
                    outcar.get('run_stats', {}).get(
                        'Total CPU time used (sec)')
                cpu_time =\
                    str(datetime.timedelta(seconds=round(cpu_time_sec)))\
                    if cpu_time_sec is not None else None
                n_cores = outcar['run_stats']['cores']
            else:
                cpu_time = None
                n_cores = None
            final_energy = vaspdir.final_energy
            if final_energy is None:
                progress = " N/A" + progress
            else:
                progress = f" | {final_energy:.4f}eV" +\
                    (f" CPU time: {cpu_time} ({n_cores} cores)"
                     if n_cores is not None else "") + progress
        mtime = vasp_output_time(wdir)
        if mtime is None:
            continue
        delta = mtime - datetime.datetime.now().timestamp()
        if nebp:
            progress = progress + "\n" + _get_neb_summary(vaspdir)
        if args.problematic and warning_list == ""\
           and (converged or running):
            continue
        print(
            f"[{print_seconds(delta): >15}]",
            colored(f"{wdir.replace("./", "")}:", attrs=['bold']),
            run_prefix + run_status + progress + warning_list)

    if len(all_warn_names_present) > 0:
        print(colored("Warnings found: ", "yellow"), all_warn_names_present)

    return 0
