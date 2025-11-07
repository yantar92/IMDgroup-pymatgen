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


"""This module implements extensions to pymatgen.io.vasp.outputs module.
"""

import warnings
import logging
import re
import os
from pathlib import Path
from monty.json import MSONable
from monty.io import zopen
from pymatgen.util.typing import PathLike
from pymatgen.io.vasp.outputs import Vasprun as pmgVasprun
from pymatgen.io.vasp.outputs import Outcar as pmgOutcar
from IMDgroup.pymatgen.io.vasp.inputs import Incar

logger = logging.getLogger(__name__)


class VasprunWarning(Warning):
    """Warning for VASP run."""


class Vasprun(pmgVasprun):
    """Modified version of pymatgen's Vasprun class, which see.
    """

    @property
    def final_energy(self) -> float:
        """Final energy from the VASP run."""
        energy = super().final_energy

        if self.incar.get('IBRION') in Incar.IBRION_IONIC_RELAX_values and\
           self.incar.get('ISIF') != Incar.ISIF_FIX_SHAPE_VOL:
            warnings.warn(
                "Energy may not be accurate when using "
                f"ISIF({self.incar.get('ISIF')})!={Incar.ISIF_FIX_SHAPE_VOL}",
                VasprunWarning
            )

        return energy


class Outcar(pmgOutcar):
    """Modified version of pymatgen's Outcar class that stores all the fields.
    """

    def as_dict(self) -> dict:
        """MSONable dict."""
        dct = super().as_dict()
        for key, value in vars(self).items():
            if key not in dct:
                dct[key] = value
        return dct


class Vasplog(MSONable):
    """Vasp logs parser.
    Attributes:
      filename (Path): Log file
      warnings: Warning records in the log
      progress: Progress records in the log
    """

    # Maximum log file size to be read
    # Larger files are read partially (first MAX_SIZE bytes)
    MAX_SIZE = 10 * 1000 * 1000

    # Adapted (and modified) from custodian/src/custodian/vasp/handlers.py
    VASP_WARNINGS = {
        "__exclude": [
            # false positives to be filtered out
            " *kinetic energy error for atom=.+",
        ],
        # Additional context lines to include in the match
        "__context": {
            "kpoints_parser": 3,
            "vasp_runtime_error": 2,
            "mag_init": 1,
            "zbrent": 2,
        },
        # Additional message to clarify a warning
        "__extra_message": {
            'slurm_error': [
                "VASP crashed.  Possible causes: time limit exceeded, not enough memory, VASP bug, cluster problem"
            ],
            'brmix': [
                "This is expected to happen once in charged systems"
            ],
            # Test data in 2025.graphite.Li.CE/03.CE.fix_lattice.KPOINTS.10k/AA/strain.c.0.00/18111/ATAT.SCF.FIXSUBSPACEMATRIX
            'subspacematrix': [
                "As long as converged, should not affect final energy"
            ],
        },
        "time_limit": [
            r"JOB [0-9]+ ON [0-9a-z]+ CANCELLED AT [^ ]+ DUE TO TIME LIMIT",
        ],
        "canceled": [
            r"JOB [0-9]+ ON [0-9a-z]+ CANCELLED AT",
        ],
        "slurm_error": [
            "slurmstepd: error",
            "prterun noticed",
            "srun: error",
        ],
        # "mag_init": [
        #     r"You use a magnetic or noncollinear calculation"
        # ],
        "kpoints_parser": [
            "Error reading KPOINTS file"
        ],
        "vasp_bug": [
            "Please submit a bug report.",
        ],
        "fortran_runtime_error": [
            "Fortran runtime error"
        ],
        "vasp_runtime_error": [
            "Error termination"
        ],
        # "relaxation_step": [
        #     # Very large atom displacement during relaxation
        #     # >=0.06
        #     "^.*dis= [1-9].*$",
        #     "^.*dis= 0.[1-9].*$",
        #     "^.*dis= 0.0[6-9].*$",
        #     "^.*maximal distance =[1-9].*$",
        #     "^.*maximal distance =0.[1-9].*$",
        #     "^.*maximal distance =0.0[6-9].*$",
        # ],
        "fexcf": [
            "ERROR FEXCF: supplied exchange-correlation table"
        ],
        "electron_convergance": [
            "The electronic self-consistency was not achieved in the given"
        ],
        "tet": [
            "Tetrahedron method fails",
            "tetrahedron method fails",
            "Routine TETIRR needs special values",
            "Tetrahedron method fails (number of k-points < 4)",
            # "BZINTS",
        ],
        "ksymm": [
            "Fatal error detecting k-mesh",
            "Fatal error: unable to match k-point",
        ],
        "ibzkpt": [
            "IBZKPT: unable to construct a generating k-lattice suitable for use",
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
            "ZBRENT: fatal internal in",
            "ZBRENT: fatal error in bracketing",
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
        "unclassified": ["error"],
    }

    VASP_PROGRESS = {
        "00SCF": [
            r"DAV:.+",
        ],
        "01relax": [
            r"step:.+harm=.+dis=.+next Energy=.+dE=.+",
            r"opt step +=.+harmonic.+distance.+",
            r"next E +=.+d E +=.+",
            r"BRION:.+",
            r"g.Force. *= .+g.Stress.=.+",
        ],
    }

    VASP_LOG_FILES = [r'slurm.+', r'stdout', r'OUTCAR', r'vasp.out']

    def __init__(self, filename: PathLike) -> None:
        """
        Args:
          fil (str): Filename to parse.
        """
        self.file = Path(filename)
        self._warnings = None
        self._progress = None
        self.line_counts = {}
        self.lines = []
        logger.debug("Reading VASP log file: %s", self.file)
        file_size = self.file.stat().st_size
        if file_size > self.MAX_SIZE:
            warnings.warn(
                f"{os.path.relpath(filename)} is too large: {file_size} > {self.MAX_SIZE}."
                " Reading partially",
                ResourceWarning
            )
        with zopen(self.file, mode="rt", encoding="UTF-8") as f:
            full_lines = f.readlines(self.MAX_SIZE)
            for line in full_lines:
                line = line.strip()
                if line in self.line_counts:
                    self.line_counts[line] += 1
                else:
                    self.lines.append(line)
                    self.line_counts[line] = 1

    @property
    def warnings(self):
        """Get all the warnings in the log.
        See VASP_WARNINGS for the full list of warnings.
        Returns: Dict of
         {log_type :
           {'message': <matched text>,
            'count': <number of occurances>}}
        """
        if self._warnings is None:
            self._warnings = self.parse(Vasplog.VASP_WARNINGS)
        return self._warnings

    @property
    def progress(self):
        """Get all the progress messages in the log.
        See VASP_PROGRESS for the full list of messages.
        Returns: Dict of
         {log_type :
           {'message': <matched text>,
            'count': <number of occurances>}}
        """
        if self._progress is None:
            self._progress = self.parse(Vasplog.VASP_PROGRESS)
        return self._progress

    @staticmethod
    def from_dir(dirname: PathLike) -> list['Vasplog']:
        """Parse all the logs in DIRNAME.
        Return a list of Vasplog instances.
        """
        return [Vasplog(f) for f in Vasplog.vasp_log_files(dirname)]

    @staticmethod
    def vasp_log_files(path: PathLike) -> list[str]:
        """Return a list of VASP log files in PATH.
        The files are sorted by modification date.
        Return [], if log files are not found.
        """
        path = Path(path)
        if not path.is_dir():
            return None
        files = [f for f in path.iterdir() if f.is_file()]
        # logger.debug("Searching slurm logs in %s across %s", path, files)
        matching = []
        for f in files:
            if any(re.match(regexp, f.name)
                   for regexp in Vasplog.VASP_LOG_FILES):
                matching.append(f)
        if len(matching) > 1:
            # Ignore OUTCAR (huge) unless we have no choice
            matching = [f for f in matching if 'OUTCAR' != f.name]
        return sorted(matching, key=lambda f: f.stat().st_mtime)

    def parse(self, log_matchers):
        """Return VASP logs matching LOG_MATCHERS.
        LOG_MATCHERS is a dict
        {"log_item_name": ["regexp1", "regexp2", ...]}
        Regexps are regular expressions matching against log text
        for specific log record.

        Additionally, the dict may contain
        "__exclude": ["regexp1", "regexp2", ...]
        to unconditionally exclude matching certain regexps (false-positives)

        "__extra_message": {'log_item_name': ["message_line_1", "message_line_2", ...]}
        contains extra information about log records (e.g. tips about some warnings).

        "__context": {'log_item_name': number of extra context lines to include}
        will make <matched text> include that number of following lines.

        Returns: Dict of
         {log_type :
           {'message': <matched text>,
            'tips': [<__extra_message lines about log_type>],
            'count': <number of occurances>}}
        """
        # Pre-compile all patterns
        exclude_re = [re.compile(p) for p in log_matchers.get('__exclude', [])]
        context_rules = log_matchers.get('__context', {})
        extra_msgs = log_matchers.get('__extra_message', {})

        # Build matcher dictionary {compiled_pattern: warn_name}
        warn_patterns = {}
        for warn_name, matchers in log_matchers.items():
            if warn_name.startswith('__'):
                continue
            warn_patterns[warn_name] = {
                'patterns': [re.compile(m) for m in matchers],
                'context': context_rules.get(warn_name, 0)
            }

        result = {}
        # Process text line by line for memory efficiency
        i = 0
        n_lines = len(self.lines)
        while i < n_lines:
            line = self.lines[i]
            i += 1
            # Check exclusions first
            if any(re.search(p, line) for p in exclude_re):
                continue
            # Check warning patterns
            for warn_name, config in warn_patterns.items():
                patterns = config['patterns']
                context = config['context']
                if any(p.search(line) for p in patterns):
                    # Collect context
                    end_idx = min(i-1 + context + 1, n_lines)
                    context_block = '\n'.join(self.lines[i-1:end_idx])

                    # Update results
                    if warn_name not in result:
                        result[warn_name] = {
                            'message': context_block,
                            'tips': extra_msgs.get(warn_name),
                            'count': 0
                        }
                    result[warn_name]['count'] += self.line_counts[line]
                    result[warn_name]['message'] = context_block
                    break  # only count one match per line

        logger.debug("Found %d log patterns", len(result))
        return result


