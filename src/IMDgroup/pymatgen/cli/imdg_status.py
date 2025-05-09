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
from xml.etree.ElementTree import ParseError
import cachetools.func
import numpy as np
from termcolor import colored
from pymatgen.io.vasp.outputs import UnconvergedVASPWarning
from pymatgen.core import Structure
from IMDgroup.pymatgen.io.vasp.outputs import Outcar
from IMDgroup.pymatgen.io.vasp.inputs import nebp, neb_dirs
from IMDgroup.pymatgen.cli.imdg_analyze\
    import read_vaspruns, IMDGVaspToComputedEnrgyDrone
from IMDgroup.pymatgen.core.structure import structure_distance
from IMDgroup.pymatgen.io.vasp.outputs import Vasplog

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


def convergedp(path, entries_dict, reread=False):
    """Return False when PATH is unconverged.
    When PATH has vasprun.xml, return final energy when it is
    converged.
    If PATH is not in ENTRIES_DICT, return False unless REREAD
    argument is True.  For REREAD=True, try reading vasprun.xml even
    if it is not present in the ENTRIES_DICT.
    """
    if nebp(path):
        logger.debug("NEB folder layout.  Scanning image folders for vasprun.xml/OUTCAR")
        for image_path in neb_dirs(path, include_ends=False):
            if not convergedp(image_path, entries_dict):
                return False
        return True
    if path not in entries_dict:
        logger.debug("%s not found in ENTRIES_DICT", path)
        if reread:
            drone = IMDGVaspToComputedEnrgyDrone(
                inc_structure=False,
                parameters=None,
                data=['final_energy', 'converged']
            )
            computed_entry = drone.assimilate(path)
            if computed_entry is None:
                return False
            return computed_entry.data['final_energy']\
                if computed_entry.data['converged'] else False
        return False
    converged = entries_dict[path].data['converged']
    final_energy = entries_dict[path].energy
    return final_energy if converged else False


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
        if nebp(parent):
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
        help="*Only* dirs matching this Python regexp pattern will be included",
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
    if nebp(path):
        times = [vasp_output_time(p) for p in neb_dirs(path)]
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


class ParseOutcarWarning(UserWarning):
    """Warning when there is a problem parsing OUTCAR."""


def status(args):
    """Main routine.
    """
    entries_dict = {}

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

        def read_dirp(p):
            if exclude_dirp(p):
                return False
            if slurm_runningp(p):
                return False
            return True

        entries = read_vaspruns(
            args.dir, path_filter=read_dirp)
        if entries is not None:
            entries_dict = {
                os.path.dirname(e.data['filename']): e
                for e in entries
            }

    paths = []
    paths_no_output = []
    for wdir, _, files in os.walk(args.dir):
        if exclude_dirp(wdir):
            continue
        if 'vasprun.xml' in files or 'OUTCAR' in files:
            paths.append(wdir)
        else:
            has_slurm = False
            for f in files:
                if re.match('slurm-[0-9]+.out', f):
                    paths.append(wdir)
                    has_slurm = True
                    break
            if not has_slurm and any(
                    f in files
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
        outcar_path = os.path.join(wdir, 'OUTCAR')
        if not os.path.isfile(outcar_path):
            outcar_path = False
        else:
            outcar_path = [outcar_path]
        if logs := Vasplog.from_dir(wdir):
            logger.debug(
                "Found VASP logs in %s: %s",
                wdir, [log.file.name for log in logs])
            progress_data = logs[-1].progress
            if len(progress_data.values()) > 0:
                progress = " | " + list(progress_data.values())[-1]['message']
            else:
                progress = " N/A"
            warning_list = ""
            for log in logs:
                for warn_name, data in log.warnings.items():
                    if args.nowarn is not None and warn_name in args.nowarn:
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
        else:
            logger.debug("Slurm log file not found in %s", wdir)
            progress = ""
            warning_list = ""

        run_status = colored("unknown", "red")
        converged = None
        running = False
        if nebp(wdir):
            # NEB-like calculation
            run_prefix = colored("IMAGES ", "magenta")
        else:
            run_prefix = ""
        if slurm_runningp(wdir):
            running = True
            run_status = colored("running", "yellow")
        else:
            try:
                converged = convergedp(wdir, entries_dict, reread=True)
                if converged is not None:
                    run_status = colored("converged", "green")\
                        if converged else colored("unconverged", "red")
            except (ParseError, FileNotFoundError):
                converged = False
                run_status = colored("incomplete vasprun.xml", "red")
            if not isinstance(converged, bool):
                final_energy = converged
            else:
                final_energy = None
            if wdir in entries_dict:
                outcar = entries_dict[wdir].data['outcar']
            elif nebp(wdir):
                outcar = None
                progress = ""
            else:
                logger.debug('Reading OUTCAR in %s', wdir)
                try:
                    outcar = Outcar(os.path.join(wdir, "OUTCAR")).as_dict()
                except Exception as exc:
                    outcar = None
                    warnings.warn(
                        f"Failed to read {os.path.join(wdir, "OUTCAR")}: {exc}",
                        ParseOutcarWarning
                    )
            cpu_time = "N/A"
            n_cores = "N/A"
            if outcar is not None:
                if final_energy is None:
                    final_energy = outcar['final_energy']
                try:
                    cpu_time_sec =\
                        outcar['run_stats']['Total CPU time used (sec)']
                    cpu_time =\
                        str(datetime.timedelta(seconds=round(cpu_time_sec)))
                except KeyError:
                    cpu_time = "N/A"
                n_cores = outcar['run_stats']['cores']
            if converged:
                # Clear progress logs
                progress = ""
            if final_energy is None:
                progress = " N/A" + progress
            else:
                progress = f" | {final_energy:.4f}eV" +\
                    (f" CPU time: {cpu_time} ({n_cores} cores)"
                     if outcar is not None else "") + progress
        mtime = vasp_output_time(wdir)
        if mtime is None:
            continue
        delta = mtime - datetime.datetime.now().timestamp()
        if nebp(wdir):
            neb_structures = []
            neb_structures_initial = []
            for p in neb_dirs(wdir):
                contcar = Path(p) / "CONTCAR"
                poscar = Path(p) / "POSCAR"
                contcar_struct = None
                if contcar.is_file():
                    try:
                        contcar_struct = Structure.from_file(contcar)
                    # There is VASP bug when we have -1.00000-2.000
                    # numbers without space.  Bail out when encoutnered.
                    except ValueError:
                        contcar_struct = None
                poscar_struct = Structure.from_file(poscar)
                if contcar_struct is not None:
                    neb_structures.append(contcar_struct)
                else:
                    neb_structures.append(poscar_struct)
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
            progress = progress + "\n" + neb_dists_initial + "\n" + neb_dists
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
