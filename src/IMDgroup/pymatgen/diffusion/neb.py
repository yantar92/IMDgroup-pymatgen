"""NEB pair generator for diffusion paths.
"""
import logging
from alive_progress import alive_bar
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
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
            discard_equivalent: bool = True,
            tol: float = 0.5) -> None:
        """Setup structure filter.
        ORIGIN is the beginning of diffusion pair (Structure).
        CUTOFF and TOL are the largest and smallest distances between
        ORIGIN and filtered structure for structure to be accepted.
        DISCARD_EQUIVALENT control whether to filter out symmetrycally
        equivalent pairs.
        """
        self.rejected = []
        self.origin = origin
        self.discard_equivalent = discard_equivalent
        if cutoff is None:
            self.cutoff = float("inf")
        else:
            self.cutoff = cutoff
        self.tol = tol

    def is_equiv(self, end1, end2):
        """Return True when END1 and END2 form equivalent pairs with ORIGIN.
        """
        matcher = StructureMatcher(attempt_supercell=True, scale=False)
        if np.isclose(
                structure_distance(self.origin, end1),
                structure_distance(self.origin, end2)) and\
            matcher.fit(
                merge_structures([self.origin, end1]),
                merge_structures([self.origin, end2])):
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
        if self.discard_equivalent:
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
        cutoff: float | None = None,
        discard_equivalent: bool = True) -> list[tuple[Structure, Structure]]:
    """Construct all possible unique diffusion pairs between ORIGIN and TARGET.
    ORIGIN is always taken as beginning of diffusion.
    Diffusion end points are taken by applying all possible symmetry
    operations of PROTOTYPE onto TARGET.
    End points that are further than CUTOFF are discarded.

    Symmatrically equivalent diffusion pairs are discarded, unless
    DISCARD_EQUIVALENT is False.

    Return a list of tuples representing begin/end structure pairs.
    """
    trans = SymmetryCloneTransformation(
        prototype,
        filter_cls=_StructFilter(origin, cutoff, discard_equivalent))
    clones = trans.get_all_clones(target)

    logger.info('Found %d pairs', len(clones))
    logger.info(
        'Distances: %s',
        [(idx, float(structure_distance(origin, clone, tol=0.5)))
         for idx, clone in enumerate(clones)])
    return list((origin, clone) for clone in clones)


def _pair_post_filter(unique_pairs, all_clones):
    """Minimize UNIQUE_PAIRS keeping ALL_CLONES reachable.
    UNIQUE_PAIRS is a list of unique diffusion paths.
    ALL_CLONES is a list of diffusion points.
    The return value will be a subset of shortest possible
    UNIQUE_PAIRS sufficient to cover ALL_CLONES.
    """
    use_pair = [False] * len(unique_pairs)
    matcher = StructureMatcher(attempt_supercell=True, scale=False)

    def add_pair_maybe(pair):
        for idx, known_pair in enumerate(unique_pairs):
            if np.isclose(
                    structure_distance(pair[0], pair[1]),
                    structure_distance(known_pair[0], known_pair[1])) and\
                matcher.fit(
                    merge_structures([pair[0], pair[1]]),
                    merge_structures([known_pair[0], known_pair[1]])):
                use_pair[idx] = True
                return
        # must not happen
        raise AssertionError("This must not happen")

    visited = [False] * len(all_clones)

    with alive_bar(len(all_clones), title='Post-filtering') as progress_bar:
        def dfs(from_idx):
            visited[from_idx] = True
            progress_bar()  # pylint: disable=not-callable
            from_struct = all_clones[from_idx]
            distances = [(structure_distance(from_struct, to_struct), to_idx)
                         for to_idx, to_struct in enumerate(all_clones)
                         if not visited[to_idx]]
            for _, to_idx in sorted(distances):
                add_pair_maybe((from_struct, all_clones[to_idx]))
                dfs(to_idx)
        dfs(0)
    return [p for idx, p in enumerate(unique_pairs) if use_pair[idx]]


def get_neb_pairs(
        structures: list[Structure],
        prototype: Structure,
        cutoff: float | None = None,
        remove_compound: bool = False)\
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

    When optional argument REMOVE_COMPOUND is provided (False by default),
    find the smallest subset of the shortest diffusion pairs that is
    sufficient to cover all possible diffusion sites for STRUCTURES
    (including symmetrically equivalent).  For example, given 1-2,
    2-3, and 1-3 diffusion pairs, 1-3 == 1-2 + 2-3 and 1-3 will be dropped.
    This is heuristics as the diffusion barrier for 1-3 might
    generally be lower compared to 1-2 + 2-3 combination.

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

    if remove_compound:
        logger.info("Removing compound paths")
        all_clones = []
        for idx, origin in enumerate(uniq_structures):
            if not _struct_is_equiv(origin, all_clones):
                all_clones.append(origin)
            for idx2, target in enumerate(uniq_structures[idx:]):
                logger.info(
                    "gen_neb_pairs: searching all pairs %d -> %d ...",
                    idx, idx2+idx)
                all_pairs = get_neb_pairs_1(
                    origin, target, prototype, cutoff,
                    discard_equivalent=False)
                all_clones += [target_clone for _, target_clone in all_pairs
                               if not _struct_is_equiv(target_clone, all_clones)]
        pairs = _pair_post_filter(pairs, all_clones)

    return pairs
