"""Check status of running VASP calculations.
"""
import sys
import os
import logging
import warnings
import subprocess
from xml.etree.ElementTree import ParseError
from termcolor import colored
from pymatgen.io.vasp.outputs import Vasprun
from IMDgroup.pymatgen.cli.imdg_analyze import read_vaspruns

logger = logging.getLogger(__name__)


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


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Report running VASP status."""

    parser.add_argument(
        "dir",
        help="""Directory to read (recusrively)""",
        type=str)


def status(args):
    """Main routine.
    """
    entries = read_vaspruns(args.dir, True)
    entries_dict = {os.path.dirname(e.data['filename']): e for e in entries}

    for wdir, _, files in os.walk(args.dir):
        if 'vasprun.xml' not in files:
            continue
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
        print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold']) + run_status)

    return 0
