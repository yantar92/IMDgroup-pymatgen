"""NEB pair generator for diffusion paths.
"""
import logging
import warnings
from multiprocessing import Pool
from alive_progress import alive_bar
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from IMDgroup.pymatgen.core.structure import merge_structures
from IMDgroup.pymatgen.transformations.symmetry_clone\
    import SymmetryCloneTransformation, apply_operation_keep_lattice
from IMDgroup.pymatgen.core.structure import structure_distance

logger = logging.getLogger(__name__)


class StructureDuplicateWarning(UserWarning):
    """Warning class for duplicate input structures."""


def _struct_is_equiv(
        struct: Structure,
        known_structs: list[Structure],
        warn=False,
        multithread=False):
    """Return True when STRUCT is equivalent to any KNOWN_STRUCTS.
    Otherwise, return False.
    When WARN is True, display warning when duplicate is found.
    When MULTITHREAD is True, use multithreading.
    """
    matcher = StructureMatcher(attempt_supercell=True, scale=False)

    def _warn(duplicate_of):
        if warn:
            origin_path = struct.properties.get('origin_path')
            origin_path_2 = duplicate_of.properties.get('origin_path')
            warnings.warn(
                "Duplicate structures found" +
                f" ({origin_path} and {origin_path_2})"
                if origin_path and origin_path_2 else "",
                StructureDuplicateWarning
            )

    if multithread:
        with Pool() as pool:
            equivs = pool.starmap(
                matcher.fit,
                [(struct, known) for known in known_structs
                 if known is not None]
            )
        for idx, val in enumerate(equivs):
            if val:
                _warn(known_structs[idx])
                return True
    else:
        for known in known_structs:
            if known is None:
                continue
            if matcher.fit(struct, known):
                _warn(known)
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
            tol: float = 0.5,
            multithread: bool = False) -> None:
        """Setup structure filter.
        ORIGIN is the beginning of diffusion pair (Structure).
        CUTOFF and TOL are the largest and smallest distances between
        ORIGIN and filtered structure for structure to be accepted.
        DISCARD_EQUIVALENT control whether to filter out symmetrycally
        equivalent pairs.
        MULTITHREAD, when True, enable multithreading.
        """
        self.rejected = []
        self.multithread = multithread
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
        if matcher.fit(
                merge_structures([self.origin, end1]),
                merge_structures([self.origin, end2]),
                symmetric=True):
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
        dist = structure_distance(
            self.origin, clone, tol=self.tol, autosort_tol=None)
        logger.debug("Considering pair %f", dist)
        if dist > self.cutoff or dist < self.tol:
            logger.debug("Too long/short (%f vs. %f)", dist, self.cutoff)
            return False
        for known in clones + self.rejected:
            dist = structure_distance(
                clone, known, tol=self.tol, autosort_tol=None)
            if dist < self.tol:
                logger.debug("Exact duplicate")
                return False
        if self.discard_equivalent:
            if self.multithread:
                with Pool() as pool:
                    equivs = pool.starmap(
                        self.is_equiv,
                        [(clone, other) for other in clones]
                    )
                    if True in equivs:
                        self.rejected.append(clone)
                        logger.debug("Equivalent path")
                        return False
            else:
                for other in clones:
                    if self.is_equiv(clone, other):
                        self.rejected.append(clone)
                        logger.debug("Equivalent path")
                        return False
        logger.debug("accepted")
        return True

    def final_filter(self, clones):
        """Filter out diffusion paths that are multiples of other paths.
        """
        # Sort structures inversely by distance from reference STRUCTURE
        # This way, we will filter out longer paths first.
        filtered = sorted(
            clones,
            key=lambda clone: structure_distance(
                self.origin, clone, tol=self.tol, autosort_tol=None),
            reverse=True)

        return filtered


def get_neb_pairs_1(
        origin: Structure,
        targets: list[Structure],
        cutoff: float | None = None,
        discard_equivalent: bool = True,
        rejected: list[tuple[Structure, Structure]] | None = None,
        multithread: bool = False
) -> list[tuple[Structure, Structure]]:
    """Return all unique diffusion pairs between ORIGIN and TARGETS.
    ORIGIN is always taken as beginning of diffusion.
    Diffusion end points are listed in TARGETS.
    End points that are further than CUTOFF are discarded.

    Symmatrically equivalent diffusion pairs are discarded, unless
    DISCARD_EQUIVALENT is False.

    Return a list of tuples representing begin/end structure pairs.

    When REJECTED argument is provided, it must be a list that will me
    modified by side effect.  The list will store all the
    symmetrically equivalent pairs not included in teh return value.

    MULTITHREAD (default: False) controls multithreading.
    """
    filter_cls = _StructFilter(
        origin, cutoff, discard_equivalent, multithread=multithread)
    clones = []
    for target in targets:
        if not clones or filter_cls.filter(target, clones):
            clones.append(target)
    clones = filter_cls.final_filter(clones)

    logger.info('Found %d pairs', len(clones))
    logger.info(
        'Distances: %s',
        [(idx, float(structure_distance(
            origin, clone, tol=0.5, autosort_tol=None)))
         for idx, clone in enumerate(clones)])
    if rejected is not None:
        for rej in filter_cls.rejected:
            rejected.append((origin, rej))
    return list((origin, clone) for clone in clones)


def graph_connected(distance_matrix):
    """Return True when graph represented by DISTANCE_MATRIX is connected.
    Otherwise, return False.

    DISTANCE_MATRIX is a two-dimentional square array representing
    graph edges.  Values in DISTANCE_MATRIX may either be float edge
    lengths or np.inf to represent missing edge.
    """
    n_vertices = len(distance_matrix[0])
    # Visit matrix: False when we cannot reach a site; True - we can.
    visited = [False for _ in range(n_vertices)]

    queue = [0]

    while len(queue) > 0:
        from_idx = queue.pop(0)
        visited[from_idx] = True
        for to_idx, distance in enumerate(distance_matrix[from_idx]):
            if not visited[to_idx] and distance != np.inf:
                queue.append(to_idx)

    logger.debug("%s", visited)
    return all(visited)


def _get_min_cutoff(distance_matrix):
    """Find the smallest distance cutoff keeping graph connected.
    DISTANCE_MATRIX is a square matrix representing diffusion path lengthes.
    """
    # Visit matrix: False when we cannot reach a site; True - we can.
    visited = [False] * len(distance_matrix[0])

    with alive_bar(
            len(visited), title='Auto-detecting cutoff') as progress_bar:

        def bfs1(queue, dist_cutoff):
            while len(queue) > 0:
                from_idx = queue.pop(0)
                if not visited[from_idx]:
                    visited[from_idx] = True
                    progress_bar()  # pylint: disable=not-callable
                distances = [(dist, to_idx)
                             for to_idx, dist
                             in enumerate(distance_matrix[from_idx])
                             if not visited[to_idx]]
                for distance, to_idx in sorted(distances):
                    if not visited[to_idx] and not distance > dist_cutoff:
                        logger.info(
                            "coverage: %s -> %s (%f)",
                            from_idx, to_idx, distance
                        )
                        queue.append(to_idx)
        visited[0] = True
        progress_bar()  # pylint: disable=not-callable
        all_distances_sorted = np.unique(distance_matrix, axis=None)
        for max_dist in all_distances_sorted:
            logger.debug(
                "Trying to reach all the sites via <=%.2f long paths",
                max_dist
            )
            queue = [idx for idx, v in enumerate(visited) if v]
            bfs1(queue, max_dist)
            if all(visited):
                for distance in all_distances_sorted:
                    # Return _larger_ distance to avoid float
                    # comparison precision problems
                    if distance != np.inf and distance > max_dist:
                        logger.debug(
                            "Max required distance %f.  Next: %f",
                            max_dist, distance)
                        # Use slightly larger value to avoid
                        # comparison errors.
                        return (max_dist + distance) / 2.0 * 1.01
                # cutoff is the largest distance, return something
                # slightly higher
                logger.debug(
                    "Cutoff is the largest diffusion path length, returning x2")
                return max_dist * 2

        raise AssertionError(f"bfs: This must not happen (visited: {visited})")


def __get_edge(all_structures, from_idx, to_idx, progress_bar=None):
    distance = structure_distance(
        all_structures[from_idx], all_structures[to_idx],
        tol=0.5, autosort_tol=None
    )
    if progress_bar is not None:
        progress_bar()  # pylint: disable=not-callable
    return (from_idx, to_idx, distance)


def get_neb_pairs(
        structures: list[Structure],
        prototype: Structure,
        cutoff: float | None | str = None,
        remove_compound: bool = False,
        multithread: bool = False
) -> list[tuple[Structure, Structure]]:
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
    The exact criterion is: sum squares of all atom displacements to
    change one structure into another must be no larger than CUTOFF^2.

    When CUTOFF is a string 'auto', auto-detect the CUTOFF, choosing
    the smallest number that is sufficient to cover all possible
    diffusion sites for STRUCTURES (each site is reachable from each
    site using diffusion paths shorter than CUTOFF).

    When REMOVE_COMPOUND is True, find the smallest subset of the
    shortest diffusion pairs that is sufficient to cover all possible
    diffusion sites for STRUCTURES (including symmetrically
    equivalent).  For example, given 1-2, 2-3, and 1-3 diffusion
    pairs, 1-3 == 1-2 + 2-3 and 1-3 will be dropped.  This is
    heuristics as the diffusion barrier for 1-3 might generally be
    lower compared to 1-2 + 2-3 combination.

    MULTITHREAD (default: False) enables multithreading.

    Returns a list of tuples containing begin/end structures.
    """
    uniq_structures = []
    logger.info(
        "gen_neb_pairs: Checking duplicates among %d structures...",
        len(structures))
    with alive_bar(len(structures), title='Checking duplicates')\
         as progress_bar:
        for struct in structures:
            if not _struct_is_equiv(
                    struct, uniq_structures, warn=True,
                    multithread=multithread):
                uniq_structures.append(struct)
            else:
                uniq_structures.append(None)
            progress_bar()  # pylint: disable=not-callable
    logger.info(
        "gen_neb_pairs: Checking duplicates... done (removed %d)",
        len(structures)-len([x for x in uniq_structures if x is not None]))
    if len(structures)-len(uniq_structures) > 0:
        logger.info(
            "gen_neb_pairs: Using structure enumeration"
            " preserving the original order (including duplicates)")

    all_clones = []
    for idx, struct in enumerate(uniq_structures):
        if struct is None:
            continue
        logger.info("Enumerating clones in structure #%d", idx)
        trans = SymmetryCloneTransformation(prototype)
        clones = trans.get_all_clones(struct, multithread=multithread)
        for clone in clones:
            clone.properties['_orig_idx'] = idx
        all_clones += clones
    logger.info("Found %d clones", len(all_clones))

    distance_matrix = np.inf * np.ones((len(all_clones), len(all_clones)))
    with alive_bar(
            len(all_clones) ** 2 if not multithread else None,
            title='Computing distance matrix') as progress_bar:
        if multithread:
            with Pool() as pool:
                # pylint: disable=not-callable
                progress_bar(len(all_clones) ** 2 / 2)
                all_edges = pool.starmap(
                    __get_edge,
                    [(all_clones, from_idx, to_idx)
                     for from_idx, _ in enumerate(all_clones)
                     for to_idx, _ in enumerate(all_clones)
                     if from_idx < to_idx]
                )
        else:
            all_edges = [
                __get_edge(all_clones, from_idx, to_idx, progress_bar)
                for from_idx, _ in enumerate(all_clones)
                for to_idx, _ in enumerate(all_clones)
                if from_idx < to_idx]

        for from_idx, to_idx, distance in all_edges:
            distance_matrix[from_idx][to_idx] = distance
            distance_matrix[to_idx][from_idx] = distance

    if cutoff == 'auto':
        logger.info("Determining minimal possible diffusion distance cutoff")

        cutoff = _get_min_cutoff(distance_matrix)
        logger.info('Found optimal cutoff to cover all sites: %f', cutoff)

    assert isinstance(cutoff, float)
    # Only keep egdes shorter than cutoff
    n_edges = 0
    for from_idx, _ in enumerate(distance_matrix):
        for to_idx, edge_len in enumerate(distance_matrix[from_idx]):
            if from_idx > to_idx:
                continue
            if not edge_len < cutoff:
                distance_matrix[from_idx][to_idx] = np.inf
                distance_matrix[to_idx][from_idx] = np.inf
            else:
                n_edges += 1
    logger.info("Found %d paths shorter than cutoff (%f)", n_edges, cutoff)

    def __verify_distance_matrix():
        for from_idx, row in enumerate(distance_matrix):
            targets = []
            for to_idx, edge_len in enumerate(row):
                if edge_len != np.inf:
                    targets.append(all_clones[to_idx])
            assert len(targets) > 0, f"isolated {from_idx}: {row}"

    __verify_distance_matrix()

    if remove_compound:
        logger.info("Removing compound paths")

        edges = []
        for from_idx, row in enumerate(distance_matrix):
            for to_idx, edge_len in enumerate(row):
                if from_idx > to_idx:
                    continue
                edges.append((edge_len, from_idx, to_idx))

        with alive_bar(
                n_edges,
                title='Removing compound paths') as progress_bar:
            assert graph_connected(distance_matrix)
            for edge_len, from_idx, to_idx in sorted(edges, reverse=True):
                if edge_len == np.inf:
                    continue
                distance_matrix[from_idx, to_idx] = np.inf
                distance_matrix[to_idx, from_idx] = np.inf
                if graph_connected(distance_matrix):
                    logger.debug(
                        "%d -> %d (%f): removed",
                        from_idx, to_idx, edge_len
                    )
                    n_edges -= 1
                    __verify_distance_matrix()
                else:
                    distance_matrix[from_idx, to_idx] = edge_len
                    distance_matrix[to_idx, from_idx] = edge_len
                    logger.info(
                        "%d -> %d (%f): kept",
                        from_idx, to_idx, edge_len
                    )
                progress_bar()  # pylint: disable=not-callable
        logger.info("Found %d non-compound paths", n_edges)

    logger.info("Searching unique diffusion paths")

    # Now, we have a number of diffusion paths, possibly starting from
    # a clone of the original structure.
    # Reduce everything in such a way that we always start from a
    # member of uniq_structures.
    pair_matrix = [[] for _ in uniq_structures]

    with alive_bar(
            len(distance_matrix),
            title='Mapping paths to origin') as progress_bar:
        for from_idx, row in enumerate(distance_matrix):
            targets = []
            orig_idx = all_clones[from_idx].properties.get('_orig_idx')
            invop = None
            if op := all_clones[from_idx].properties.get('symop'):
                invop = op.inverse
            for to_idx, edge_len in enumerate(row):
                if from_idx > to_idx:
                    continue
                if edge_len != np.inf:
                    target = all_clones[to_idx].copy()
                    if invop:
                        apply_operation_keep_lattice(target, invop)
                    pair_matrix[orig_idx].append(target)
            progress_bar()  # pylint: disable=not-callable

    pairs = []
    for idx, targets in enumerate(pair_matrix):
        logger.info(
            "gen_neb_pairs: searching paths from %d -> %s",
            idx, [target.properties.get('_orig_idx') for target in targets])
        pairs += get_neb_pairs_1(
            uniq_structures[idx],
            targets,
            cutoff,
            discard_equivalent=True,
            multithread=multithread
        )
    logger.info("Found %d unique paths", len(pairs))

    return pairs
