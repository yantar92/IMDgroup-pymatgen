"""This module implements extensions to pymatgen.io.vasp.outputs module.
"""

import warnings
import logging
import re
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

    # Adapted (and modified) from custodian/src/custodian/vasp/handlers.py
    VASP_WARNINGS = {
        "__exclude": [
            # false positives to be filtered out
            " *kinetic energy error for atom=.+",
        ],
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
        "slurm_error": [
            "slurmstepd: error.+",
            "prterun noticed.+",
        ],
        # "mag_init": [
        #     r"You use a magnetic or noncollinear calculation.+\n.+"
        # ],
        "kpoints_parser": [
            "Error reading KPOINTS file.+\n.+\n.+\n.+"
        ],
        "vasp_bug": [
            "Please submit a bug report.",
        ],
        "fortran_runtime_error": [
            "Fortran runtime error.+"
        ],
        "vasp_runtime_error": [
            "Error termination.+\n.+\n.+"
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
        "electron_convergance": [
            "The electronic self-consistency was not achieved in the given.+"
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
            "ZBRENT: fatal error in bracketing.+\n.+\n.+",
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
        "unclassified": ["^.*error.*$"],
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
        with zopen(self.file, mode="rt", encoding="UTF-8") as f:
            self.text = f.read()

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
    def vasp_log_files(path: PathLike):
        """Return a list of VASP log files in PATH.
        The files are sorted by modification date.
        Return None, if log files are not found.
        """
        path = Path(path)
        if not path.is_dir():
            return None
        files = [f for f in path.iterdir() if f.is_file()]
        logger.debug("Searching slurm logs in %s across %s", path, files)
        matching = []
        for f in files:
            if any(re.match(regexp, f.name)
                   for regexp in Vasplog.VASP_LOG_FILES):
                matching.append(f)
        if len(matching) > 1:
            # Ignore OUTCAR (huge) unless we have no choice
            matching = [f for f in matching if 'OUTCAR' != f.name]
        if len(matching) > 0:
            return sorted(matching, key=lambda f: f.stat().st_mtime)
        return None

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

        Returns: Dict of
         {log_type :
           {'message': <matched text>,
            'tips': [<__extra_message lines about log_type>],
            'count': <number of occurances>}}
        """
        result = {}
        excluded = []
        if '__exclude' in log_matchers:
            excluded = log_matchers['__exclude']
        for warn_name, matchers in log_matchers.items():
            if warn_name == "__exclude":
                continue
            for matcher in matchers:
                matches = re.findall(matcher, self.text, flags=re.MULTILINE)
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
                        extra_lines = None
                        if '__extra_message' in log_matchers:
                            extra_lines =\
                                log_matchers['__extra_message'].get(warn_name)
                        result[warn_name] = {
                            'message': matches[-1],
                            'tips': extra_lines,
                            'count': num
                        }
        logger.debug("Finished processing")
        return result

