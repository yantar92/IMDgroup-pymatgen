"""NEB pair generator for diffusion paths.
"""
import logging
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from IMDgroup.pymatgen.core.structure import merge_structures
from IMDgroup.pymatgen.transformations.symmetry_clone\
    import SymmetryCloneTransformation
import numpy as np

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


class _struct_filter():
    """Structure filter that rejects equivalent diffusion pairs.
    Given a structure STRUCT and ORIGIN, STRUCT+ORIGIN combined will
    always be symmetrically non-equivalent if not rejected.
    Also, any STRUCT further than CUTOFF or closer than TOL from
    ORIGIN will be rejected.
    """

    def __init__(
            self,
            origin: Structure,
            cutoff: float,
            tol: float = 0.5) -> None:
        """Setup structure filter.
        ORIGIN is the beginning of diffusion pair (Structure).
        CUTOFF and TOL are the largest and smallest distances between
        ORIGIN and filtered structure for structure to be accepted.
        """
        self.rejected = []
        self.origin = origin
        self.cutoff = cutoff
        self.tol = tol

    def is_equiv(self, end1, end2):
        """Return True when END1 and END2 form equivalent pairs with ORIGIN.
        """
        matcher = StructureMatcher(attempt_supercell=True, scale=False)
        if matcher.fit(
                merge_structures([self.origin, end1]),
                merge_structures([self.origin, end2])):
            return True
        return False

    # Repeated diffusion paths are 100% equivalent
    # A natural extension of this approach would be checking if a new
    # diffusion path is a linear combination (via natural
    # coefficients) of known paths, but that may, in general, miss
    # some physically non-equivalent paths. Consider, for example,
    # 1-3 vs 1-2-3 path. 1-2 and 2-3 barrier may, in some cases be
    # higher than 1-3, especially for more complex structures. So,
    # such an approach would only be approximation applicable to
    # certain simple systems.
    def is_multiple(self, end1, end2):
        """Return True when END2 path is a multiple of END1 path wrt ORIGIN.
        TOL is tolerance - ORIGIN->END vector components smaller than
        TOL are ignored.
        """
        v1 = np.array(end1.coords) - np.array(self.origin.coords)
        v2 = np.array(end2.coords) - np.array(self.origin.coords)

        # Remove small displacements according to TOL

        def zero_small_vec(vec):
            """When norm of VEC is less than TOL, return 0.
            Otherwise, return VEC.
            """
            return vec if np.linalg.norm(vec) > self.tol\
                else np.array([0, 0, 0])

        v1 = np.array([zero_small_vec(vec) for vec in v1])
        v2 = np.array([zero_small_vec(vec) for vec in v2])

        fracs = []
        for vec1, vec2 in zip(v1, v2):
            frac = []
            for x, y in zip(vec1, vec2):
                if np.isclose(x, 0, atol=self.tol) and\
                   np.isclose(y, 0, atol=self.tol):
                    frac.append(True)
                elif np.isclose(y/x, int(y/x), atol=self.tol):
                    frac.append(y/x)
                else:
                    return False
        return np.all(np.isclose(fracs, fracs[0], atol=self.tol))

    def filter(self, clone, clones):
        """Return False if CLONE should be rejected.
        Return True otherwise.
        CLONE is rejected when:
        (1) It is too far/close from ORIGIN
        (2) It is too close to any of CLONES
        (3) Its diffusion pair with ORIGIN is symmetrically equivalent
            to ORIGIN + any of CLONES.
        (4) Its diffusion path can be formed by repeating another
            known diffusion path.
        """
        dist_fn = SymmetryCloneTransformation.structure_distance
        dist = dist_fn(self.origin, clone)
        if dist > self.cutoff or dist < self.tol:
            return False
        for rej in self.rejected:
            dist = SymmetryCloneTransformation.structure_distance(clone, rej)
            if dist < self.tol:
                return False
        for other in clones:
            if self.is_equiv(clone, other):
                self.rejected.append(clone)
                return False
        for other in clones + self.rejected:
            if self.is_multiple(other, clone):
                return False
        return True


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
        filter_cls=_struct_filter(origin, cutoff))
    clones = trans.get_all_clones(target)
    logger.debug('Found %d pairs', len(clones))
    logger.debug(
        'Distances: %s',
        list(SymmetryCloneTransformation.structure_distance(origin, clone)
             for clone in clones))
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
    logger.debug(
        "gen_neb_pairs: removing duplicates among %d structures...",
        len(structures))
    for struct in structures:
        if not _struct_is_equiv(struct, uniq_structures):
            uniq_structures.append(struct)
    logger.debug(
        "gen_neb_pairs: removing duplicates... done (removed %d)",
        len(structures)-len(uniq_structures))

    pairs = []
    for idx, origin in enumerate(uniq_structures):
        for idx2, target in enumerate(uniq_structures[idx:]):
            logger.debug(
                "gen_neb_pairs: searching pairs %d -> %d ...",
                idx, idx2+idx)
            pairs += get_neb_pairs_1(
                origin, target, prototype, cutoff)
    return pairs
