"""NEB pair generator for diffusion paths.
"""
import logging
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
from IMDgroup.pymatgen.core.structure import merge_structures
from IMDgroup.pymatgen.transformations.symmetry_clone\
    import SymmetryCloneTransformation
from IMDgroup.pymatgen.core.structure import\
    (structure_distance, structure_diff)

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
            cutoff: float,
            discard_combinations=False,
            tol: float = 0.5) -> None:
        """Setup structure filter.
        ORIGIN is the beginning of diffusion pair (Structure).
        CUTOFF and TOL are the largest and smallest distances between
        ORIGIN and filtered structure for structure to be accepted.
        DISCARD_COMBINATIONS, when True, filters out diffusion
        paths are are combinations of multiple other paths (approximation).
        """
        self.rejected = []
        self.origin = origin
        self.cutoff = cutoff
        self.discard_combinations = discard_combinations
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

    def _zero_small(self, v):
        """Return v when it is large. Return 0 vector otherwise.
        """
        norm = np.linalg.norm(self.origin.lattice.get_cartesian_coords(v))
        if norm > self.tol:
            return v
        return np.array([0, 0, 0])

    def _is_linear_combination_1(self, vector, base, limit) -> bool:
        """Return True if VECTOR is an integer linear combination of BASE.
        """
        if len(base) == 0:
            return False
        if np.array_equal(self._zero_small(vector), [0, 0, 0]):
            logger.debug("Small vector is a linear combination of anything")
            return True
        for mult in range(limit):
            if self._is_linear_combination_1(
                    vector+mult*base[0], base[1:], limit):
                return True
            if mult != 0 and self._is_linear_combination_1(
                    vector-mult*base[0], base[1:], limit):
                return True
        return False

    def is_linear_combination(
            self, end: Structure, base: list[Structure]) -> bool:
        """Return True when END structure is a linear combination of BASE.
        Linear combination here implies only natural coefficients.
        BASE is a list of structures.

        The idea of this filter is that diffusion path that can be
        constructed from shorted paths will end up going exactly
        through them.  Such idea is only an approximation that is
        likely valid for very distant paths, but may not be valid for
        shorter. Consider, for example, 1-3 vs 1-2-3 path. 1-2 and 2-3
        barrier may, in some cases be higher than 1-3, especially for
        more complex structures. So, such an approach would only be
        approximation applicable to certain simple systems.
        """
        v1 = structure_diff(self.origin, end)
        v_base = [structure_diff(self.origin, s) for s in base]

        # Ignore too small displacement (according to self.tol)
        v1 = np.array([self._zero_small(v) for v in v1])
        v_base = [np.array([self._zero_small(v) for v in v2]) for v2 in v_base]

        logger.debug(
            "Linear combination? %s",
            [v for v in v1 if not np.array_equal(v, [0, 0, 0])])

        result = self._is_linear_combination_1(v1, v_base, limit=10)
        return result

    def is_multiple(self, end1: Structure, end2: Structure) -> bool:
        """Return True when END2 path is a multiple of END1 path wrt ORIGIN.
        """
        v1 = structure_diff(self.origin, end1)
        v2 = structure_diff(self.origin, end2)

        # Ignore too small displacement (according to self.tol)
        v1 = np.array([self._zero_small(v) for v in v1])
        v2 = np.array([self._zero_small(v) for v in v2])

        logger.debug(
            "Multipe? %s -> %s",
            [v for v in v1 if not np.array_equal(v, [0, 0, 0])],
            [v for v in v2 if not np.array_equal(v, [0, 0, 0])])

        # 0/0 would throw a warning.  We don't care about it here.
        with np.errstate(divide='ignore'):
            fracs = np.divide(v2, v1)
        max_mult = np.round(np.nanmax(fracs))
        min_mult = np.round(np.nanmin(fracs))

        logger.debug("Multipliers: %s...%s", min_mult, max_mult)

        if max_mult != min_mult:
            return False

        end1_mult = self.origin.copy()
        for idx in range(len(end1_mult)):
            end1_mult.translate_sites([idx], v1[idx]*max_mult)
        if structure_distance(end1_mult, end2, tol=self.tol) == 0:
            logger.debug("Found multiple: %dx", max_mult)
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
        filtered = []
        for clone in clones:
            uniq = True
            for other in (clones + self.rejected):
                if clone != other and\
                   not self.is_equiv(clone, other) and\
                   self.is_multiple(other, clone):
                    uniq = False
                    break
            if uniq and self.discard_combinations and\
               self.is_linear_combination(
                    clone, [c for c in clones
                            if not self.is_equiv(c, clone)]):
                logger.debug(
                    "Found linear combination (dist=%f)",
                    structure_distance(self.origin, clone)
                )
                uniq = False
            if uniq:
                filtered.append(clone)
        return filtered


def get_neb_pairs_1(
        origin: Structure,
        target: Structure,
        prototype: Structure,
        cutoff: float | None = None,
        discard_combinations=False) -> list[tuple[Structure, Structure]]:
    """Construct all possible unique diffusion pairs between ORIGIN and TARGET.
    ORIGIN is always taken as beginning of diffusion.
    Diffusion end points are taken by applying all possible symmetry
    operations of PROTOTYPE onto TARGET.
    End points that are further than CUTOFF are discarded.
    When DISCARD_COMBINATIONS is True, discard diffusion paths
    that can be constructed as a combination of other paths. (This is
    an approximation and thus disabled by default).

    Return a list of tuples representing begin/end structure pairs.
    """
    trans = SymmetryCloneTransformation(
        prototype,
        filter_cls=_StructFilter(origin, cutoff, discard_combinations))
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
        cutoff: float | None = None,
        discard_combinations=False)\
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

    When DISCARD_COMBINATIONS is True, discard diffusion paths
    that can be constructed as a combination of other paths. (This is
    an approximation and thus disabled by default).

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
                origin, target, prototype, cutoff, discard_combinations)
    return pairs
