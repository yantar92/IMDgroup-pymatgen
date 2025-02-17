"""NEB pair generator for diffusion paths.
"""
import logging
import warnings
from multiprocessing import Pool
from alive_progress import alive_bar
import numpy as np
import networkx as nx
from networkx import MultiDiGraph
from networkx.algorithms.cycles import _johnson_cycle_search\
    as johnson_cycle_search
from pymatgen.core import Structure
from IMDgroup.pymatgen.core.structure import\
    merge_structures, structure_matches
from IMDgroup.pymatgen.transformations.symmetry_clone\
    import SymmetryCloneTransformation
from IMDgroup.pymatgen.core.structure import\
    structure_diff, get_matched_structure

logger = logging.getLogger(__name__)


class get_neb_pairs_warning(UserWarning):
    """Warning class for get_neb_pairs."""


class NEB_Graph(MultiDiGraph):
    """Graph representing diffusion paths.
    """

    def __init__(
            self,
            structures: list[Structure] | None = None,
            jimage_idxs: list[int] | None = None,
            multithread: bool = False):
        """Create complete NEB graph from STRUCTURES.
        Assume that all the structures have the same lattice and atoms
        corresponding to each other.

        When JIMAGE_IDXS is None, NEB graph edges will be constructed
        using the shortest distances between STRUCTURES (accounting
        for periodic boundary conditions).  This means that self-self
        diffusion paths will be ignored.

        JIMAGE_IDXS, when provided, lists the STRUCTURE site indices
        to consider for computing multiple possible paths between the
        same pair of structures (including self-self).  The diffusion
        paths will be constructed considering [-1..1, -1..1, -1..1]
        images of JIMAGE_IDXS sites, but not any other sites (where
        only the shortest path will be considered).  This is useful to
        compute self-self diffusion where there are too many ways to
        compute possible displacement vectors from one structure to
        another.
        """

        super().__init__()

        self.structures = structures
        self.multithread = multithread

        if structures is None:
            return

        all_vecs = self.__get_all_vecs()

        with alive_bar(
                len(all_vecs),
                title="Adding diffusion paths to graph") as progress_bar:
            for from_idx, to_idx, vec in all_vecs:
                if jimage_idxs is None:
                    self.__add_edge(from_idx, to_idx, vec)
                else:
                    for jimage in [[i, j, k]
                                   for i in range(-1, 2)
                                   for j in range(-1, 2)
                                   for k in range(-1, 2)]:
                        vec2 = vec.copy()
                        for idx in jimage_idxs:
                            site_from = structures[from_idx][idx].to_unit_cell()
                            site_to = structures[to_idx][idx].to_unit_cell()
                            assert site_from is not None
                            assert site_to is not None
                            lattice = structures[from_idx].lattice
                            vec2[idx] = lattice.get_cartesian_coords(
                                site_to.frac_coords + jimage
                                - site_from.frac_coords)
                        self.__add_edge(from_idx, to_idx, vec2)
                progress_bar()  # pylint: disable=not-callable

    def __add_edge(self, from_idx: int, to_idx: int, vector):
        def get_distance(vec):
            distance = 0
            for v in vec:
                d = np.linalg.norm(v)
                if d > 0.5:
                    distance += d
            return distance

        distance = get_distance(vector)
        from_energy = self.structures[from_idx].properties['final_energy']
        to_energy = self.structures[to_idx].properties['final_energy']
        energy_barrier = to_energy - from_energy
        self.add_edge(
            from_idx, to_idx,
            distance=distance,
            vector=np.array(vector),
            energy_barrier=energy_barrier
        )
        if to_idx != from_idx:
            self.add_edge(
                to_idx, from_idx,
                distance=distance,
                vector=-np.array(vector),
                energy_barrier=-energy_barrier
            )

    @staticmethod
    def _get_vec(
            from_idx: int, to_idx: int,
            from_struct: Structure, to_struct: Structure,
            progress_bar=None):
        """Compute minimal vector connecting from_struct and to_struct.
        Return (from_idx, to_idx, vec).
        """
        vec = structure_diff(from_struct, to_struct, tol=0, match_first=False)
        if progress_bar is not None:
            progress_bar()  # pylint: disable=not-callable
        return (from_idx, to_idx, vec)

    def __get_all_vecs(self):
        """Compute all structure diff vectors for the NEB graph."""
        if self.multithread:
            return self._compute_vecs_multithreaded()
        else:
            return self._compute_vecs_singlethreaded()

    def _compute_vecs_multithreaded(self):
        """Compute vectors using multithreading."""
        with alive_bar(
                None, title='Computing distance matrix') as progress_bar:
            with Pool() as pool:
                progress_bar(len(self.structures) ** 2 / 2)
                return pool.starmap(
                    self._get_vec,
                    [(from_idx, to_idx, from_struct, to_struct)
                     for from_idx, from_struct in enumerate(self.structures)
                     for to_idx, to_struct in enumerate(self.structures)
                     if from_idx <= to_idx]
                )

    def _compute_vecs_singlethreaded(self):
        """Compute vectors without multithreading."""
        with alive_bar(
                int(len(self.structures) ** 2 / 2),
                title='Computing distance matrix') as progress_bar:
            return [
                self._get_vec(
                    from_idx, to_idx, from_struct, to_struct, progress_bar)
                for from_idx, from_struct in enumerate(self.structures)
                for to_idx, to_struct in enumerate(self.structures)
                if from_idx <= to_idx
            ]

    def debug_print_graph(self):
        for from_idx, to_idx, key, data in self.edges(data=True, keys=True):
            max_v = [0, 0, 0]
            for v in data['vector']:
                if np.linalg.norm(v) > np.linalg.norm(max_v):
                    max_v = v
            with np.printoptions(precision=2, suppress=True):
                logger.debug(
                    "%d -> %d (%d): %f.2Å; %s",
                    from_idx, to_idx, key,
                    data['distance'], max_v
                )

    def connected(self, idxs: list[int] | None = None):
        """Return True when graph is connected.
        Otherwise, return False.
        If IDXS is a list, only check connectivity of IDXS vertices.
        """
        n_vertices = len(self.structures)
        # Visit matrix: False when we cannot reach a site; True - we can.
        visited = [False for _ in range(n_vertices)]

        if idxs is None:
            idxs = [idx for idx, _ in enumerate(self.structures)]
        queue = [idxs[0]]

        while len(queue) > 0:
            from_idx = queue.pop(0)
            visited[from_idx] = True
            for _, to_idx in self.edges(from_idx):
                if not visited[to_idx]:
                    queue.append(to_idx)

        return all(visited[idx] for idx in idxs)

    def diffusion_path_infinite(self, start_idx):
        """Return True when START_IDX is within infinite diffusion path.

        Infinite diffusion path is a path that will move through the
        material infinitely or, in other words, path is not bound within a
        finite volume.  Mathematically, sum of displacement vectors along
        the graph cycle containing START_IDX must be non-zero.
        """
        # Note: It is sufficient to check one START_IDX and skip all the
        # symmetrically equivalent clones - by definition they should
        # yield exactly the same results.
        # Of course, such simplification won't work when comparing
        # non-equivalent starting positions for the diffusion.
        # visited = np.full(len(self.structures), False)

        logger.debug(
            "Searching infinite diffusion paths including %d", start_idx)

        components = nx.strongly_connected_components(self)
        G = None
        for c in components:
            if start_idx in c:
                G = self.subgraph(c)
                break
        assert G is not None
        assert start_idx in G.nodes()

        def _check_cycle(cycle):
            if start_idx in cycle:
                cycle += [cycle[0]]
                tot_vecs = [0]
                prev_idx = cycle[0]
                for next_idx in cycle[1:]:
                    new_vecs = []
                    for _, to, vec in self.edges(prev_idx, data='vector'):
                        if to == next_idx:
                            for tot_v in tot_vecs:
                                new_vecs.append(tot_v + vec)
                    tot_vecs = new_vecs
                    prev_idx = next_idx
                for v in tot_vecs:
                    if not np.isclose(np.linalg.norm(v), 0):
                        logger.debug(
                            "Found infinite diffusion path for %d: %s (%f)",
                            start_idx,
                            " -> ".join([str(i) for i in cycle]),
                            np.linalg.norm(v)
                        )
                        return True
            return False

        n_skipped = 0
        max_skipped = int(1E6)
        for cycle in johnson_cycle_search(G, [start_idx]):
            if _check_cycle(cycle):
                return True
            if start_idx in cycle:
                n_skipped += 1
                # logger.debug("skipped closed cycle: %d", n_skipped)
                if n_skipped > max_skipped:
                    warnings.warn(
                        "Unable to find infinite diffusion path"
                        f" for {start_idx} after attempting"
                        f" {max_skipped} paths"
                    )
                    break
        return False

    def all_diffusion_paths_infinite(
            self, idxs: list[int] | None = None) -> bool:
        """Return True when all IDXS are within infinite diffusion paths.
        If IDXS is None, check all the structures in the graph.

        Assume that structures with the same
        structure.properties['_orig_idx'] are symmetrically equivalent
        and do not need to be checked individually whether they lay in
        infinite diffusion path.
        """
        _checked = []
        if idxs is None:
            idxs = [idx for idx, _ in enumerate(self.structures)]
        for idx in idxs:
            struct = self.structures[idx]
            if struct.properties['_orig_idx'] in _checked:
                continue
            if not self.diffusion_path_infinite(idx):
                return False
            _checked.append(struct.properties['_orig_idx'])
        return True

    def get_min_cutoff(self, idx_connected: list[int] | None = None):
        """Find the smallest distance cutoff keeping graph connected.
        Also, make sure that vertice structures are also a part of
        infinite diffusion path.

        Assume that structures with the same
        structure.properties['_orig_idx'] are symmetrically equivalent
        and do not need to be checked individually whether they lay in
        infinite diffusion path.

        When IDX_CONNECTED is provided, it should be a list of indices
        to check for connectivity.
        """
        # Visit matrix: False when we cannot reach a site; True - we can.
        visited = [False] * len(self.structures)
        if idx_connected is None:
            start_idx = 0
            must_visit = [True] * len(self.structures)
        else:
            start_idx = idx_connected[0]
            must_visit = [False] * len(self.structures)
            for idx in idx_connected:
                must_visit[idx] = True

        with alive_bar(
                None, title='Auto-detecting cutoff'):

            def bfs1(queue, dist_cutoff):
                while len(queue) > 0:
                    from_idx = queue.pop(0)
                    if not visited[from_idx]:
                        visited[from_idx] = True
                    distances = [
                        (dist, to_idx)
                        for _, to_idx, dist
                        in self.edges(from_idx, data='distance')
                        if not visited[to_idx]]
                    for distance, to_idx in sorted(distances):
                        if not visited[to_idx] and not distance > dist_cutoff:
                            logger.debug(
                                "coverage: %s -> %s (%fÅ)",
                                from_idx, to_idx, distance
                            )
                            queue.append(to_idx)
            visited[start_idx] = True
            all_distances_sorted = np.unique(
                [dist for _, _, dist in self.edges(data='distance')])
            for max_dist in all_distances_sorted:
                logger.debug(
                    "Trying to reach all the sites via <=%.2fÅ long paths",
                    max_dist
                )
                queue = [idx for idx, v in enumerate(visited) if v]
                bfs1(queue, max_dist)
                all_visited = all(did for must, did in zip(must_visit, visited)
                                  if must)
                all_infinite = True
                if all_visited:
                    data = [
                        (from_idx, to_idx, key, data)
                        for from_idx, to_idx, key, data
                        in self.edges(keys=True, data=True)
                        if data['distance'] > max_dist
                    ]
                    for from_idx, to_idx, key, _ in data:
                        self.remove_edge(from_idx, to_idx, key)
                    all_infinite =\
                        self.all_diffusion_paths_infinite(idx_connected)
                    for from_idx, to_idx, _, d in data:
                        self.add_edges_from([(from_idx, to_idx, d)])
                if all_visited and all_infinite:
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
                        "Cutoff is the largest diffusion path length"
                        ", returning x2")
                    return max_dist * 2

            raise AssertionError(
                f"bfs: This must not happen (visited: {visited})")


def _remove_duplicates(
        structures: list[Structure],
        idxs: list[int] | None = None,
        multithread: bool = False,
) -> list[Structure | None]:
    """Remove duplicates from STRUCTURES.
    Return list of the same length with duplicate structures replaced
    with None.

    When IDXS is a list of indices, only compare site proximity of the
    listed sites.

    When MULTITHREAD is True, use multithreading.
    """
    uniq_structures = []
    logger.info(
        "gen_neb_pairs: Checking duplicates among %d structures...",
        len(structures))

    if idxs is not None:
        def __check_idxs(struct1, struct2):
            for idx in idxs:
                if struct1[idx].distance(struct2[idx]) > 0.5:
                    return False
            return True
        cmp_fun = __check_idxs
        # Can't pickle local function
        multithread = False
    else:
        cmp_fun = None

    with alive_bar(len(structures), title='Checking duplicates')\
         as progress_bar:
        for struct in structures:
            if struct is None:
                uniq_structures.append(None)
                progress_bar()  # pylint: disable=not-callable
                continue
            if structure_matches(
                    struct, uniq_structures, warn=True,
                    cmp_fun=cmp_fun,
                    multithread=multithread):
                uniq_structures.append(None)
            else:
                uniq_structures.append(struct)
            progress_bar()  # pylint: disable=not-callable
    logger.info(
        "gen_neb_pairs: Checking duplicates... done (removed %d)",
        len(structures)-len([x for x in uniq_structures if x is not None]))
    if len(structures)-len(uniq_structures) > 0:
        logger.info(
            "gen_neb_pairs: Using structure enumeration"
            " preserving the original order (including duplicates)")
    return uniq_structures


def get_neb_pairs(
        structures: list[Structure],
        prototype: Structure,
        cutoff: float | None | str = None,
        remove_compound: bool = False,
        multithread: bool = False,
        return_unfiltered: bool = False
) -> list[tuple[Structure, Structure]] |\
     tuple[list[tuple[Structure, Structure]],
           list[tuple[Structure, Structure]]]:
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
    When RETURN_UNFILTERED (default: False) is True, return a tuple of
    two lists: (unique_pairs, all_pairs). all_pairs will represent the
    full resulting diffusion graph including symmetrically equivalent
    diffusion pairs.

    The returned tuples will contain initial and final points of
    diffusion without normalizing coordinates to unit cell.  The
    returned structures will also have 1-to-1 site matching.
    """
    # Arrange 1-to-1 site matching in all the provided structures
    # including prototype (needed to detect insertion sites)
    for idx, struct in enumerate(structures):
        if struct is not None:
            structures[idx] =\
                get_matched_structure(prototype, struct)

    uniq_structures = structures
    # Remove duplicates in the inserted sites.
    if len(structures[0]) > len(prototype):
        idxs = list(range(len(prototype), len(structures[0])))
        logger.info(
            "Removing duplicate positions for insertions into prototype")
        uniq_structures = _remove_duplicates(structures, idxs, multithread)

    # Remove duplicates.
    logger.info("Removing symmetry duplicates")
    uniq_structures = _remove_duplicates(
        uniq_structures, multithread=multithread)

    # Inform user about structure numbers
    logger.info("Assigning indices")
    for idx, struct in enumerate(uniq_structures):
        logger.info(
            "#%d: %s",
            idx, struct.properties['origin_path']
            if struct is not None
            else "ignore (duplicate)"
            )

    # Compute all the symmetrically equivalent structure clones
    all_clones = []
    for idx, struct in enumerate(uniq_structures):
        if struct is None:
            continue
        logger.info("Enumerating clones in structure #%d", idx)
        trans = SymmetryCloneTransformation(prototype)
        clones = trans.get_all_clones(struct, multithread=multithread)
        for clone_idx, clone in enumerate(clones):
            clone.properties['_orig_idx'] = idx
            clone.properties['_clone_idx'] =\
                len(all_clones) + clone_idx
        logger.info(
            "#%d clones assigned indices #%d..#%d",
            idx, len(all_clones), len(all_clones)+len(clones)-1
            )
        all_clones += clones
    logger.info("Found %d clones", len(all_clones))

    # Build diffusion graph connecting all the clones
    neb_graph = NEB_Graph(
        all_clones,
        jimage_idxs=list(range(len(prototype), len(structures[0])))
        if len(structures[0]) > len(prototype) else None,
        multithread=multithread)

    # Select structures that must remain connected in the graph
    # Currently, we simply maintain connectivity of the lowest-energy
    # structures (maybe through higher-energy intermediates).
    energies = []
    assert neb_graph.structures is not None
    for idx, struct in enumerate(neb_graph.structures):
        energy = struct.properties.get('final_energy')
        if energy is None:
            origin_path = struct.properties.get('origin_path')
            raise ValueError(f"Energy data is missing for {origin_path}")
        logger.debug("Energy %d: %f", idx, energy)
        energies.append(energy)
    # Only take the lowest-energy structure
    # +1E-9 is to counter floating point error where structure clones
    # may have slight variation of energies
    energy_threshold = sorted(np.unique(energies))[0] + 1E-9
    logger.info("Energy theshold: %f", energy_threshold)
    low_en_idxs = [
        idx for idx, s in enumerate(neb_graph.structures)
        if not s.properties['final_energy'] > energy_threshold]
    logger.info(
        "Diffusion graph connectivity will be limited"
        " to lowest-energy configurations: %s",
        low_en_idxs
    )

    # If diffusion distance cutoff is not provided, detect it
    # automatically, choosing the minimal possible cutoff that
    # maintains structure connectivity.
    if cutoff == 'auto':
        logger.info("Determining minimal possible diffusion distance cutoff")
        cutoff = neb_graph.get_min_cutoff(low_en_idxs)
        logger.info('Found optimal cutoff to cover all sites: %f', cutoff)

    # Only keep graph egdes shorter than cutoff
    assert isinstance(cutoff, float)
    n_edges = 0
    # Cast to list because MultiDiGraph.edges cannot handle
    # changing edges on the fly.
    for from_idx, to_idx, key, edge_len in\
            list(neb_graph.edges(data='distance', keys=True)):
        if not edge_len < cutoff:
            neb_graph.remove_edge(from_idx, to_idx, key)
        else:
            n_edges += 1
    logger.info("Found %d paths shorter than cutoff (%f)", n_edges, cutoff)

    neb_graph.debug_print_graph()

    def _connected_and_infinite():
        is_connected = neb_graph.connected(low_en_idxs)
        is_infinite = False
        if is_connected:
            is_infinite =\
                neb_graph.all_diffusion_paths_infinite(low_en_idxs)
        return is_connected and is_infinite

    # Loop over largest known (or estimated as energy difference)
    # barriers and remove as many as possible.  Always keep <=0 barriers.
    n_removed = 0
    barriers = [(barrier, from_idx, to_idx, key)
                for from_idx, to_idx, key, barrier in
                neb_graph.edges(keys=True, data='energy_barrier')
                if barrier >= 1E-9]
    logger.info('Removing high-energy barriers')
    with alive_bar(
            len(barriers),
            title='Removing high-energy barriers'
    ) as progress_bar:
        # Try removing one by one, starting from the highest.
        for en, from_idx, to_idx, key in sorted(barriers, reverse=True):
            data = neb_graph.edges[from_idx, to_idx, key]
            neb_graph.remove_edge(from_idx, to_idx, key)
            if _connected_and_infinite():
                logger.info(
                    "%d -> %d (%d): removed (%feV)",
                    from_idx, to_idx, key, en)
                n_removed += 1
            else:
                logger.info(
                    "%d -> %d (%d): kept (%feV)",
                    from_idx, to_idx, key, en)
                neb_graph.add_edges_from([(from_idx, to_idx, data)])
            progress_bar()  # pylint: disable=not-callable
    logger.info("Removed %d high-energy barriers", n_removed)

    # If requested, remove as many as possible diffusion paths,
    # starting from the longest, while keeping graph connectivity.
    if remove_compound:
        logger.info("Removing compound paths")
        edges = [(distance, from_idx, to_idx, key)
                 for from_idx, to_idx, key, distance
                 in neb_graph.edges(keys=True, data='distance')]
        n_edges = len(edges)
        with alive_bar(
                n_edges, title='Removing compound paths') as progress_bar:
            for edge_len, from_idx, to_idx, key in sorted(edges, reverse=True):
                data = neb_graph.edges[from_idx, to_idx, key]
                neb_graph.remove_edge(from_idx, to_idx, key)
                if _connected_and_infinite():
                    logger.debug(
                        "%d -> %d (%d): removed (%fÅ)",
                        from_idx, to_idx, key, edge_len
                    )
                    n_edges -= 1
                else:
                    logger.info(
                        "%d -> %d (%d): kept (%fÅ)",
                        from_idx, to_idx, key, edge_len
                    )
                    neb_graph.add_edges_from([(from_idx, to_idx, data)])
                progress_bar()  # pylint: disable=not-callable
        logger.info("Found %d non-compound paths", n_edges)

    logger.info(
        "Final diffusion graph: %s",
        [(from_idx, to_idx) for from_idx, to_idx, _ in neb_graph.edges]
    )
    neb_graph.debug_print_graph()

    # Get rid of symmetrically equivalent diffusion paths.
    logger.info("Searching unique diffusion paths")
    unique_edges = []
    merged_pairs = []
    _known_dists = []
    edges = []
    dists = []
    for from_idx, to_idx, key, dist in neb_graph.edges(data='distance', keys=True):
        edges.append((from_idx, to_idx, key))
        dists.append(dist)
    n_edges = len(edges)
    with alive_bar(
            n_edges,
            title='Removing equivalent paths') as progress_bar:
        for (from_idx, to_idx, key), dist in zip(edges, dists):
            # Two paths are equivalent when they (1) form a
            # symmetrically unique structure when combined;
            # (2) when they length/vector is the same.
            merged = merge_structures(
                [all_clones[from_idx], all_clones[to_idx]])
            # Equivalent paths must have the same length.
            close_pair_idxs = [
                idx for idx, d in enumerate(_known_dists)
                if np.isclose(dist, d)]
            if len(close_pair_idxs) > 0 and structure_matches(
                    merged, [merged_pairs[idx] for idx in close_pair_idxs],
                    multithread=multithread):
                logger.debug("%s path is non-unique", (from_idx, to_idx))
                progress_bar()  # pylint: disable=not-callable
                continue
            unique_edges.append((dist, from_idx, to_idx, key))
            merged_pairs.append(merged)
            _known_dists.append(dist)
            progress_bar()  # pylint: disable=not-callable
    logger.info("Found %d unique paths", len(unique_edges))

    # Sort by lentgh
    unique_edges = sorted(unique_edges)

    def get_pair(from_idx, to_idx, key):
        origin_struct = all_clones[from_idx]
        target_struct = origin_struct.copy()
        for idx, v in enumerate(neb_graph.edges[from_idx, to_idx, key]['vector']):
            target_struct.translate_sites(
                [idx], v,
                frac_coords=False, to_unit_cell=False)
        return (origin_struct, target_struct)

    # Use full displacement vector to produce final diffusion point.
    pairs = []
    for _, from_idx, to_idx, key in unique_edges:
        pairs.append(get_pair(from_idx, to_idx, key))

    if return_unfiltered:
        unfiltered_pairs = [
            get_pair(from_idx, to_idx, key)
            for from_idx, to_idx, key in neb_graph.edges(keys=True)
        ]
        return pairs, unfiltered_pairs

    return pairs
