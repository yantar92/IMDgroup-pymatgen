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
import datetime
from monty.io import zopen
from termcolor import colored
from xml.etree.ElementTree import ParseError
from pymatgen.io.vasp.outputs import (Vasprun, Outcar)
from pymatgen.io.vasp.inputs import Incar
from IMDgroup.pymatgen.cli.imdg_analyze import read_vaspruns

logger = logging.getLogger(__name__)
# Adapted (and modified) from custodian/src/custodian/vasp/handlers.py
VASP_WARNINGS = {
    "__exclude": [
        # false positives to be filtered out
        " *kinetic energy error for atom=.+",
    ],
    "slurm_error": [
        "slurmstepd: error.+",
        "prterun noticed.+",
    ],
    "electron_convergance": [
        "The electronic self-consistency was not achieved in the given.+"
    ],
    "tet": [
        "Tetrahedron method fails",
        "tetrahedron method fails",
        "Routine TETIRR needs special values",
        "Tetrahedron method fails (number of k-points < 4)",
        "BZINTS",
    ],
    "ksymm": [
        "Fatal error detecting k-mesh",
        "Fatal error: unable to match k-point",
    ],
    "inv_rot_mat": ["rotation matrix was not found (increase SYMPREC)"],
    "brmix": ["BRMIX: very serious problems"],
    "subspacematrix": ["WARNING: Sub-Space-Matrix is not hermitian in DAV"],
    "tetirr": ["Routine TETIRR needs special values"],
    "incorrect_shift": ["Could not get correct shifts"],
    "real_optlay": ["REAL_OPTLAY: internal error", "REAL_OPT: internal ERROR"],
    "rspher": ["ERROR RSPHER"],
    "dentet": ["DENTET"],  # reason for this warning is
    # that the Fermi level cannot be determined accurately
    # enough by the tetrahedron method
    # https://vasp.at/forum/viewtopic.php?f=3&t=416&p=4047&hilit=dentet#p4047
    "too_few_bands": ["TOO FEW BANDS"],
    "triple_product": ["ERROR: the triple product of the basis vectors"],
    "rot_matrix": [
        "Found some non-integer element in rotation matrix", "SGRCON"],
    "brions": ["BRIONS problems: POTIM should be increased"],
    "pricel": ["internal error in subroutine PRICEL"],
    "zpotrf": ["LAPACK: Routine ZPOTRF failed", "Routine ZPOTRF ZTRTRI"],
    "amin": ["One of the lattice vectors is very long (>50 A), but AMIN"],
    "zbrent": [
        "ZBRENT: fatal internal in", "ZBRENT: fatal error in bracketing",
        "ZBRENT:  can not reach accuracy",
        # "ZBRENT: can't locate minimum, use default step"
    ],
    # Note that PSSYEVX and PDSYEVX errors are identical up to LAPACK routine:
    # P<prec>SYEVX uses <prec> = S(ingle) or D(ouble) precision
    "pssyevx": ["ERROR in subspace rotation PSSYEVX"],
    "pdsyevx": ["ERROR in subspace rotation PDSYEVX"],
    "eddrmm": ["WARNING in EDDRMM: call to ZHEGV failed"],
    "edddav": ["Error EDDDAV: Call to ZHEGV failed"],
    "algo_tet": ["ALGO=A and IALGO=5X tend to fail"],
    "grad_not_orth": ["EDWAV: internal error, the gradient is not orthogonal"],
    "nicht_konv": ["ERROR: SBESSELITER : nicht konvergent"],
    "zheev": ["ERROR EDDIAG: Call to routine ZHEEV failed!"],
    "eddiag": ["ERROR in EDDIAG: call to ZHEEV/ZHEEVX/DSYEV/DSYEVX failed"],
    "elf_kpar": ["ELF: KPAR>1 not implemented"],
    "elf_ncl": ["WARNING: ELF not implemented for non collinear case"],
    "rhosyg": ["RHOSYG"],
    "posmap": ["POSMAP"],
    "point_group": ["group operation missing"],
    "pricelv": [
        "PRICELV: current lattice and primitive lattice are incommensurate"],
    "symprec_noise": [
        "determination of the symmetry of your systems shows a strong"],
    "dfpt_ncore": [
        "PEAD routines do not work for NCORE",
        "remove the tag NPAR from the INCAR file"],
    "bravais": ["Inconsistent Bravais lattice"],
    "nbands_not_sufficient": ["number of bands is not sufficient"],
    "hnform": ["HNFORM: k-point generating"],
    "coef": ["while reading plane", "while reading WAVECAR"],
    "set_core_wf": ["internal error in SET_CORE_WF"],
    "read_error": ["Error reading item", "Error code was IERR= 5"],
    "auto_nbands": ["The number of bands has been changed"],
    "unclassified": [".*error.*"],
}

VASP_PROGRESS = {
    "relax": [
        r"step:.+harm=.+dis=.+next Energy=.+dE=.+",
        r"opt step +=.+harmonic.+distance.+",
        r"next E +=.+d E +=.+",
    ]
}


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


def nebp(path):
    """Return True when PATH is a NEB-like run.
    """
    incar_path = os.path.join(path, 'INCAR')
    if os.path.isfile(incar_path):
        incar = Incar.from_file(incar_path)
        if 'IMAGES' in incar:
            return True
    return False


def _neb_dirs(path):
    """Return a list of NEB dirs in PATH.
    """
    if nebp(path):
        incar = Incar.from_file(os.path.join(path, "INCAR"))
        nimages = incar['IMAGES']
        paths = [os.path.join(path, f"{n:02d}")
                 for n in range(1, nimages + 1)]
        return paths
        # return [p for p in paths if os.path.isdir(p)]
    return None


def convergedp(path, entries_dict, reread=False):
    """Return False when PATH is unconverged.
    When PATH has vasprun.xml, return final energy when it is
    converged.
    If PATH is not in ENTRIES_DICT, return False unless REREAD
    argument is True.  For REREAD=True, try reading vasprun.xml even
    if it is not present in the ENTRIES_DICT.
    """
    if nebp(path):
        for image_path in _neb_dirs(path):
            if not convergedp(image_path, entries_dict):
                return False
        return True
    if path not in entries_dict:
        logger.debug("%s not found in ENTRIES_DICT", path)
        if reread:
            run = Vasprun(
                os.path.join(path, 'vasprun.xml'),
                parse_dos=False,
                parse_eigen=False)
            return run.final_energy if run.converged else False
        return False
    converged = entries_dict[path].data['converged']
    final_energy = entries_dict[path].energy
    return final_energy if converged else False


def slurm_runningp(path):
    """Is slurm running in DIR?
    """
    if shutil.which("squeue") is None:
        return False
    result = subprocess.check_output(
        "squeue -u $USER -o %Z | tail -n +2",
        shell=True).split()
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


def slurm_log_file(path):
    """Return slurm log file in PATH.
    Return None, if the log file is not found.
    """
    files = [f for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f))]
    logger.debug("Searching slurm logs in %s across %s", path, files)
    matching = []
    for f in files:
        if "slurm" in f:
            matching.append(os.path.join(path, f))
    if len(matching) > 0:
        newest = matching[0]
        mtime_newest = os.stat(newest).st_mtime
        for f in matching:
            mtime = os.stat(f).st_mtime
            if mtime_newest < mtime:
                mtime_newest = mtime
                newest = f
        return newest
    return None


def get_vasp_logs(log_file, log_matchers):
    """Return VASP logs in LOG_FILE matching LOG_MATCHERS.
    Returns: Dict of
     {log_type :
       {'message': <message text>,
        'count': <number of occurances>}}
    """
    result = {}
    with zopen(log_file, mode="rt") as f:
        text = f.read()
        excluded = []
        if '__exclude' in log_matchers:
            excluded = log_matchers['__exclude']
        for warn_name, matchers in log_matchers.items():
            if warn_name == "__exclude":
                continue
            for matcher in matchers:
                matches = re.findall(matcher, text)
                filtered_matches = []
                for m in matches:
                    valid = True
                    for exclude_re in excluded:
                        if re.match(exclude_re, m):
                            valid = False
                            break
                    if valid:
                        filtered_matches.append(m)
                matches = filtered_matches
                num = len(matches)
                if num > 0:
                    if warn_name in result:
                        result[warn_name]['count'] += num
                    else:
                        result[warn_name] = {
                            'message': matches[-1],
                            'count': num
                        }
    return result


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
        output.append(f"{days} day" + ("s" if days != 1 else ""))
    if hours != 0:
        output.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes != 0:
        output.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    # if seconds != 0:
    #     output.append(f"{seconds} second" + ("s" if seconds != 1 else ""))
    if negative:
        output.append("ago")
    return " ".join(output)


def vasp_output_time(path):
    """Return last VASP output modification time in PATH.
    """
    if nebp(path):
        return max([vasp_output_time(p) for p in _neb_dirs(path)])
    outcar = os.path.join(path, 'OUTCAR')
    if os.path.isfile(outcar):
        return os.path.getmtime(outcar)
    return None


def status(args):
    """Main routine.
    """
    entries_dict = {}

    def exclude_dirp(p):
        if args.exclude is not None and re.match(args.exclude, p):
            return True
        return False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

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
    for wdir, _, files in os.walk(args.dir):
        if exclude_dirp(wdir):
            continue
        if 'vasprun.xml' in files:
            paths.append(wdir)
        else:
            for f in files:
                if re.match('slurm-[0-9]+.out', f):
                    paths.append(wdir)
                    break
    paths = sorted(paths)
    for wdir in paths:
        outcar_path = os.path.join(wdir, 'OUTCAR')
        if not os.path.isfile(outcar_path):
            outcar_path = False
        if log_file := (slurm_log_file(wdir) or outcar_path):
            logger.debug("Found slurm logs in %s: %s", wdir, log_file)
            progress_data = get_vasp_logs(log_file, VASP_PROGRESS)
            if len(progress_data.values()) > 0:
                progress = " | " + list(progress_data.values())[-1]['message']
            else:
                progress = " N/A"
            warn_data = get_vasp_logs(log_file, VASP_WARNINGS)
            warning_list = ""
            for _, data in warn_data.items():
                warning_list += "\n" +\
                    colored(f"⮤Warning ({data['count']}x): ", "yellow") +\
                    data['message']
        else:
            logger.debug("Slurm log file not found in %s", wdir)
            progress = ""
            warning_list = ""

        run_status = colored("unknown", "red")
        if nebp(wdir):
            # NEB-like calculation
            run_prefix = colored("IMAGES ", "magenta")
        else:
            run_prefix = ""
        if slurm_runningp(wdir):
            run_status = colored("running", "yellow")
        else:
            try:
                converged = convergedp(wdir, entries_dict, reread=True)
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
                    outcar = Outcar(os.path.join(wdir, "OUTCAR")).as_dict()
                if outcar is not None:
                    cpu_time_sec =\
                        outcar['run_stats']['Total CPU time used (sec)']
                    cpu_time =\
                        str(datetime.timedelta(seconds=round(cpu_time_sec)))
                    n_cores = outcar['run_stats']['cores']
                    progress = f" | {final_energy:.2f}eV" +\
                        f" CPU time: {cpu_time} ({n_cores} cores)"
                if converged is not None:
                    run_status = colored("converged", "green")\
                        if converged else colored("unconverged", "red")
            except (ParseError, FileNotFoundError):
                run_status = colored("incomplete vasprun.xml", "red")
        mtime = vasp_output_time(wdir)
        delta = datetime.datetime.now().timestamp() - mtime
        print(
            colored(f"[{print_seconds(delta): >20}]", color="yellow"),
            colored(f"{wdir.replace("./", "")}:", attrs=['bold']),
            run_prefix + run_status + progress + warning_list)

    return 0
