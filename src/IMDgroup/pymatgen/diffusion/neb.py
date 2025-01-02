"""NEB pair generator for diffusion paths.
"""
import logging
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
from IMDgroup.pymatgen.core.structure import merge_structures
from IMDgroup.pymatgen.transformations.symmetry_clone\
    import SymmetryCloneTransformation
from IMDgroup.pymatgen.core.structure import structure_distance

logger = logging.getLogger(__name__)


def _struct_is_equiv(
        struct: Structure,
        known_structs: list[Structure]):
    """Return True when STRUCT is equivalent to any KNOWN_STRUCTS.
    Otherwise, return False.
    """
    matcher = StructureMatcher(attempt_supercell=True, scale=False)
    for known in known_structs:
        if matcher.fit(struct, known):
            return True
    return False


class _StructFilter():
    """Structure filter that rejects equivalent diffusion pairs.
    Given a structure STRUCT and ORIGIN, STRUCT+ORIGIN combined will
    always be symmetrically non-equivalent if not rejected.
    Also, any STRUCT further than CUTOFF or closer than TOL from
    ORIGIN will be rejected.
    """

    def __init__(
            self,
            origin: Structure,
            cutoff: float | None,
            tol: float = 0.5) -> None:
        """Setup structure filter.
        ORIGIN is the beginning of diffusion pair (Structure).
        CUTOFF and TOL are the largest and smallest distances between
        ORIGIN and filtered structure for structure to be accepted.
        """
        self.rejected = []
        self.origin = origin
        if cutoff is None:
            self.cutoff = float("inf")
        else:
            self.cutoff = cutoff
        self.tol = tol

    def is_equiv(self, end1, end2):
        """Return True when END1 and END2 form equivalent pairs with ORIGIN.
        """
        matcher = StructureMatcher(attempt_supercell=True, scale=False)
        if matcher.fit(
                merge_structures([self.origin, end1], tol=self.tol),
                merge_structures([self.origin, end2], tol=self.tol)):
            return True
        return False

    def filter(self, clone, clones):
        """Return False if CLONE should be rejected.
        Return True otherwise.
        CLONE is rejected when:
        (1) It is too far/close from ORIGIN
        (2) It is too close to any of CLONES
        (3) Its diffusion pair with ORIGIN is symmetrically equivalent
            to ORIGIN + any of CLONES.
        """
        dist = structure_distance(self.origin, clone)
        if dist > self.cutoff or dist < self.tol:
            return False
        for rej in self.rejected:
            dist = structure_distance(clone, rej)
            if dist < self.tol:
                return False
        for other in clones:
            if self.is_equiv(clone, other):
                self.rejected.append(clone)
                return False
        return True

    def final_filter(self, clones):
        """Filter out diffusion paths that are multiples of other paths.
        """
        # Sort structures inversely by distance from reference STRUCTURE
        # This way, we will filter out longer paths first.
        filtered = sorted(
            clones,
            key=lambda clone: structure_distance(self.origin, clone),
            reverse=True)

        return filtered


def get_neb_pairs_1(
        origin: Structure,
        target: Structure,
        prototype: Structure,
        cutoff: float | None = None) -> list[tuple[Structure, Structure]]:
    """Construct all possible unique diffusion pairs between ORIGIN and TARGET.
    ORIGIN is always taken as beginning of diffusion.
    Diffusion end points are taken by applying all possible symmetry
    operations of PROTOTYPE onto TARGET.
    End points that are further than CUTOFF are discarded.

    Return a list of tuples representing begin/end structure pairs.
    """
    trans = SymmetryCloneTransformation(
        prototype,
        filter_cls=_StructFilter(origin, cutoff))
    clones = trans.get_all_clones(target)

    logger.info('Found %d pairs', len(clones))
    logger.info(
        'Distances: %s',
        [(idx, float(structure_distance(origin, clone, tol=0.5)))
         for idx, clone in enumerate(clones)])
    return list((origin, clone) for clone in clones)


def get_neb_pairs(
        structures: list[Structure],
        prototype: Structure,
        cutoff: float | None = None)\
        -> list[tuple[Structure, Structure]]:
    """Construct all possible unique diffusion pairs from STRUCTURES.
    The STRUCTURES must all be derived from PROTOTYPE structure (have
    the same lattice parameter).  Usually STRUCTURES, contain a list
    of possible unique sites for interstitial/substitutional atom (or
    atoms).

    The algorithm assumes that applying symmetry operation for
    PROTOTYPE on any element STRUCTURES produces a valid alternative
    diffusion start/end point.

    When optional argument CUTOFF is provided, all the diffusion pairs
    that require moving atoms more than CUTOFF ans will be discarded.
    The exact criterion is: sum of all atom displacements to change
    one structure into another must be no larger than CUTOFF.

    Returns a list of tuples containing begin/end structures.
    """
    uniq_structures = []
    logger.info(
        "gen_neb_pairs: removing duplicates among %d structures...",
        len(structures))
    for struct in structures:
        if not _struct_is_equiv(struct, uniq_structures):
            uniq_structures.append(struct)
    logger.info(
        "gen_neb_pairs: removing duplicates... done (removed %d)",
        len(structures)-len(uniq_structures))

    pairs = []
    for idx, origin in enumerate(uniq_structures):
        for idx2, target in enumerate(uniq_structures[idx:]):
            logger.info(
                "gen_neb_pairs: searching pairs %d -> %d ...",
                idx, idx2+idx)
            pairs += get_neb_pairs_1(
                origin, target, prototype, cutoff)
    return pairs
