"""Check status of running VASP calculations.
"""
import sys
import os
import logging
import warnings
import subprocess
from xml.etree.ElementTree import ParseError
from monty.io import zopen
from termcolor import colored
from pymatgen.io.vasp.outputs import Vasprun
from IMDgroup.pymatgen.cli.imdg_analyze import read_vaspruns

logger = logging.getLogger(__name__)
# Adapted (and modified) from custodian/src/custodian/vasp/handlers.py
VASP_WARNINGS = {
    "electron_convergance": [
        "The electronic self-consistency was not achieved in the given"
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
        "ZBRENT: fatal internal in", "ZBRENT: fatal error in bracketing"],
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
}


def custom_showwarning(
        message, category, filename, lineno, file=None, line=None):
    """Print warning in nicer way.
    """
    output = colored(
        f"{category.__name__}: ",
        "yellow",
        attrs=['bold']) + f'{message}'
    print(output, file=file or sys.stderr)


warnings.showwarning = custom_showwarning


def slurm_runningp(path):
    """Is slurm running in DIR?
    """
    result = subprocess.check_output(
        "squeue -u $USER -o %Z | tail -n +2",
        shell=True).split()
    if os.path.abspath(path) in [s.decode('utf-8') for s in result]:
        return True
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


def get_vasp_warnings(log_file):
    """Return VASP warnings logged in LOG_FILE.
    Returns: Dict of
     {warning_type :
       {'message': <message text>,
        'count': <number of occurances>}}
    """
    result = {}
    with zopen(log_file, mode="rt") as f:
        text = f.read()
        for warn_name, matchers in VASP_WARNINGS.items():
            for matcher in matchers:
                num = text.count(matcher)
                if num > 0:
                    if warn_name in result:
                        result[warn_name]['count'] += num
                    else:
                        result[warn_name] = {
                            'message': matcher,
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


def status(args):
    """Main routine.
    """
    entries_dict = {}
    entries = read_vaspruns(args.dir, True)
    if entries is not None:
        entries_dict = {os.path.dirname(e.data['filename']): e for e in entries}

    paths = []
    for wdir, _, files in os.walk(args.dir):
        if 'vasprun.xml' in files:
            paths.append(wdir)
    paths = sorted(paths)
    for wdir in paths:
        run_status = colored("unknown", "red")
        if slurm_runningp(wdir):
            run_status = colored("running", "yellow")
        else:
            try:
                if wdir in entries_dict:
                    converged = entries_dict[wdir].data['converged']
                else:
                    run = Vasprun(
                        os.path.join(wdir, 'vasprun.xml'),
                        parse_dos=False,
                        parse_eigen=False)
                    converged = run.converged
                run_status = colored("converged", "green") if converged\
                    else colored("unconverged", "red")
            except ParseError:
                run_status = colored("incomplete vasprun.xml", "red")
        if log_file := slurm_log_file(wdir):
            logger.debug("Found slurm logs in %s: %s", wdir, log_file)
            warn_data = get_vasp_warnings(log_file)
            warning_list = ""
            for _, data in warn_data.items():
                warning_list += "\n" +\
                    colored(f"Warning ({data['count']}x): ", "yellow") +\
                    data['message']
        else:
            logger.debug("Slurm log file not found in %s", wdir)
            warning_list = ""
        print(colored(
            f"{wdir.replace("./", "")}: ", attrs=['bold'])
              + run_status + warning_list)

    return 0
