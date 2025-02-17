"""NEB pair generator for diffusion paths.
"""
import logging
import warnings
from multiprocessing import Pool
from alive_progress import alive_bar
import numpy as np
import networkx as nx
from networkx.algorithms.cycles import _johnson_cycle_search\
    as johnson_cycle_search
from pymatgen.core import Structure
from IMDgroup.pymatgen.core.structure import\
    merge_structures, get_supercell_size, structure_matches
from IMDgroup.pymatgen.transformations.symmetry_clone\
    import SymmetryCloneTransformation
from IMDgroup.pymatgen.core.structure import\
    structure_diff, structure_distance, get_matched_structure

logger = logging.getLogger(__name__)


class get_neb_pairs_warning(UserWarning):
    """Warning class for get_neb_pairs."""


class _NEB_Graph:

    def __init__(
            self,
            structures: list[Structure],
            multithread: bool = False):
        """Create complete NEB graph from STRUCTURES.
        Assume that all the structures have the same lattice and atoms
        corresponding to each other.
        """
        self.__diffusion_path_cache = {}
        self.structures = structures
        self.multithread = multithread
        all_edges = self.__get_all_edges()
        edge_matrix = np.full(
            (len(structures), len(structures)), None, dtype=object)
        for from_idx, to_idx, data in all_edges:
            edge_matrix[from_idx, to_idx] = data
            data_rev = {
                'distance': data['distance'],
                'vector': -data['vector'],
                'energy_barrier': -data['energy_barrier']
            }
            edge_matrix[to_idx, from_idx] = data_rev
        self._edge_matrix = edge_matrix

    @property
    def edges(self):
        """Return a list of all edges, as a generator.
        Each element is (from_idx, to_idx, edge_data)
        where edge_data is a dict containing 'vector' and 'distance'
        entries representing distance between structures and vector
        moving from one structure to another.
        """
        for from_idx, row in enumerate(self._edge_matrix):
            for to_idx, data in enumerate(row):
                if data is not None:
                    yield (from_idx, to_idx, data)

    def remove_edge(self, from_idx, to_idx):
        """Remove edge between FROM_IDX and TO_IDX structures.
        Return the edge removed or None if there was no edge between
        FROM_IDX and TO_IDX.
        """
        edge = (from_idx, to_idx, self._edge_matrix[from_idx, to_idx])
        if edge[2] is None:
            return None
        self._edge_matrix[from_idx, to_idx] = None
        return edge

    def remove_vertice_edges(self, idx):
        """Remove vertice edges for IDX structure.
        This will keep the structrue itself, but remove all its edges.
        Return the list of edges removed.
        """
        removed = []
        for to_idx, _ in enumerate(self.structures):
            removed.append(self.remove_edge(idx, to_idx))
            removed.append(self.remove_edge(to_idx, idx))
        return [edge for edge in removed if edge is not None]

    def set_edges(self, edges):
        """Set EDGES.
        EDGES is a list of (from_idx, to_idx, edge_data_dict)
        or a single tuple representing one edge to be removed.
        """
        if not isinstance(edges, list):
            edges = [edges]
        for from_idx, to_idx, data in edges:
            self._edge_matrix[from_idx, to_idx] = data

    @staticmethod
    def _get_edge(
            from_idx: int, to_idx: int,
            from_struct: Structure, to_struct: Structure,
            progress_bar=None):
        vec = structure_diff(from_struct, to_struct, tol=0, match_first=False)
        distance = structure_distance(
            from_struct, to_struct, tol=0.5, match_first=False)
        to_energy = to_struct.properties['final_energy']
        from_energy = from_struct.properties['final_energy']
        energy_barrier = to_energy - from_energy
        data = {
            'distance': distance,
            'vector': np.array(vec),
            'energy_barrier': energy_barrier
        }
        if progress_bar is not None:
            progress_bar()  # pylint: disable=not-callable
        return (from_idx, to_idx, data)

    def __get_all_edges(self):
        """Compute all edges for the NEB graph."""
        if self.multithread:
            return self._compute_edges_multithreaded()
        else:
            return self._compute_edges_singlethreaded()

    def _compute_edges_multithreaded(self):
        """Compute edges using multithreading."""
        with alive_bar(
                None, title='Computing distance matrix') as progress_bar:
            with Pool() as pool:
                progress_bar(len(self.structures) ** 2 / 2)
                return pool.starmap(
                    self._get_edge,
                    [(from_idx, to_idx, from_struct, to_struct)
                     for from_idx, from_struct in enumerate(self.structures)
                     for to_idx, to_struct in enumerate(self.structures)
                     if from_idx < to_idx]
                )

    def _compute_edges_singlethreaded(self):
        """Compute edges without multithreading."""
        with alive_bar(
                int(len(self.structures) ** 2 / 2),
                title='Computing distance matrix') as progress_bar:
            return [
                self._get_edge(
                    from_idx, to_idx, from_struct, to_struct, progress_bar)
                for from_idx, from_struct in enumerate(self.structures)
                for to_idx, to_struct in enumerate(self.structures)
                if from_idx < to_idx
            ]

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
            for to_idx, edge in enumerate(self._edge_matrix[from_idx]):
                if edge is None:
                    continue
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

        cache = self.__diffusion_path_cache
        cycle = cache.get(start_idx) if cache is not None else None
        if cycle is not None:
            prev_idx = cycle[0]
            cycle_valid = True
            for next_idx in cycle[1:]:
                if self._edge_matrix[prev_idx, next_idx] is None:
                    cycle_valid = False
                    break
                prev_idx = next_idx
            if cycle_valid:
                logger.debug(
                    "(cached) Found infinite diffusion path for %d: %s",
                    start_idx,
                    " -> ".join([str(i) for i in cycle]),
                )
                return True

        logger.debug(
            "Searching infinite diffusion paths including %d", start_idx)
        nx_G = nx.DiGraph()
        for from_idx, to_idx, edge in self.edges:
            if edge is not None:
                nx_G.add_edge(from_idx, to_idx)
        components = nx.strongly_connected_components(nx_G)
        for c in components:
            if start_idx in c:
                nx_G = nx_G.subgraph(c)
                break
        assert start_idx in nx_G.nodes()

        def _check_cycle(cycle):
            if start_idx in cycle:
                cycle += [cycle[0]]
                tot_vec = 0
                prev_idx = cycle[0]
                for next_idx in cycle[1:]:
                    tot_vec += self._edge_matrix[prev_idx, next_idx]['vector']
                    prev_idx = next_idx
                if not np.isclose(np.linalg.norm(tot_vec), 0):
                    logger.debug(
                        "Found infinite diffusion path for %d: %s (%f)",
                        start_idx,
                        " -> ".join([str(i) for i in cycle]),
                        np.linalg.norm(tot_vec)
                    )
                    self.__diffusion_path_cache[start_idx] = cycle
                    return True
            return False

        n_skipped = 0
        max_skipped = int(1E6)
        for cycle in johnson_cycle_search(nx_G, [start_idx]):
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
                    distances = [(edge['distance'], to_idx)
                                 for to_idx, edge
                                 in enumerate(self._edge_matrix[from_idx])
                                 if not visited[to_idx] and edge is not None]
                    for distance, to_idx in sorted(distances):
                        if not visited[to_idx] and not distance > dist_cutoff:
                            logger.debug(
                                "coverage: %s -> %s (%f)",
                                from_idx, to_idx, distance
                            )
                            queue.append(to_idx)
            visited[start_idx] = True
            all_distances_sorted = np.unique(
                [edge['distance'] for _, _, edge in self.edges
                 if edge is not None])
            for max_dist in all_distances_sorted:
                logger.debug(
                    "Trying to reach all the sites via <=%.2f long paths",
                    max_dist
                )
                queue = [idx for idx, v in enumerate(visited) if v]
                bfs1(queue, max_dist)
                all_visited = all(did for must, did in zip(must_visit, visited)
                                  if must)
                all_infinite = True
                if all_visited:
                    deleted = [
                        self.remove_edge(from_idx, to_idx)
                        for from_idx, to_idx, edge in self.edges
                        if edge['distance'] > max_dist]
                    all_infinite =\
                        self.all_diffusion_paths_infinite(idx_connected)
                    self.set_edges(deleted)
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


# FIXME: When using PROTOTYPE to fill up the space, and inserted
# atom/molecule is close to the cell edge, we risk creating a
# supercell that is not fully relaxed.
# The easiest way to avoid this would be demanding at least 2x2x2
# structures as input.
def __enlarge_cell(structure, prototype, scales):
    """Enlarge STRUCTURE up to SCALES.
    Use PROTOTYPE to fill newly appended volume.
    Return the new structure.
    """
    if structure is None:
        return None
    # Create scaled up prototype
    scaled_structure = prototype.copy() * scales
    scaled_structure.properties = structure.properties
    # Remove all the points that are bound by the original
    # structure.
    sites_to_remove = []
    for site in scaled_structure:
        site.to_unit_cell(in_place=True)
    structure = structure.copy()
    for site in structure:
        site.to_unit_cell(in_place=True)
    for idx, site in enumerate(scaled_structure):
        inside = True
        for scale, coord in zip(scales, site.frac_coords):
            if scale == 2 and coord >= 0.5:
                inside = False
                break
        if inside:
            sites_to_remove.append(idx)
    scaled_structure.remove_sites(sites_to_remove)
    # Add sites from STRUCTURE in place of the removed from the
    # prototype
    for site in structure:
        scaled_structure.append(
            species=site.species,
            coords=site.coords,
            coords_are_cartesian=True,
            properties=site.properties
        )
    assert scaled_structure.is_valid()
    return scaled_structure


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


def __scale_structures_maybe(prototype, uniq_structures):
    """Scale PROTOTYPE and UNIQ_STRUCTURES up to at least 2x2x2 supercell.
    Return (prototype, structures), maybe unchanged.
    """
    dims = get_supercell_size(prototype)
    scales = [2 if dim == 1 else 1 for dim in dims]
    if dims[0] == 1 or dims[1] == 1 or dims[2] == 1:
        warnings.warn(
            "Prototype is not at least 2x2x2 supercell"
            f" ({dims[0]}x{dims[1]}x{dims[2]})."
            " Scaling up (inaccurate)"
            "\n Suggestion: better give relaxed supercell as input.",
            get_neb_pairs_warning
        )
        uniq_structures = [
            __enlarge_cell(struct, prototype, scales)
            for struct in uniq_structures]
        prototype = __enlarge_cell(prototype, prototype, scales)
        assert prototype is not None
        # FIXME: get_supercell_size does not catch scaling for PEO
        # structure.  Potential pymatgen bug.
        # dims = tuple(dim * scale for dim, scale in zip(dims, scales))
    return prototype, uniq_structures


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

    IMPORTANT: STRUCTURES must be at least 2x2x2 supercells.  If not,
    they will be scaled up to become 2x2x2 by appending PROTOTYPE.
    (This is done to ensure for technical reasons so that we can cover
    self->self diffusion paths)

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

    # Scale everything to at least 2x2x2 supercell.
    prototype, uniq_structures = \
        __scale_structures_maybe(prototype, uniq_structures)

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
    neb_graph = _NEB_Graph(all_clones, multithread=multithread)

    # Select structures that must remain connected in the graph
    # Currently, we simply maintain connectivity of the lowest-energy
    # structures (maybe through higher-energy intermediates).
    energies = []
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
    for from_idx, to_idx, edge in neb_graph.edges:
        edge_len = edge['distance']
        if from_idx > to_idx:
            continue
        if not edge_len < cutoff:
            neb_graph.remove_edge(from_idx, to_idx)
            # pylint: disable=arguments-out-of-order
            neb_graph.remove_edge(to_idx, from_idx)
        else:
            n_edges += 1
    logger.info("Found %d paths shorter than cutoff (%f)", n_edges, cutoff)

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
    barriers = [(data['energy_barrier'], from_idx, to_idx)
                for from_idx, to_idx, data in neb_graph.edges
                if data['energy_barrier'] >= 1E-9]
    logger.info('Removing high-energy barriers')
    with alive_bar(
            len(barriers),
            title='Removing high-energy barriers'
    ) as progress_bar:
        # Try removing one by one, starting from the highest.
        for en, from_idx, to_idx in sorted(barriers, reverse=True):
            removed = neb_graph.remove_edge(from_idx, to_idx)
            if _connected_and_infinite():
                logger.info("%d -> %d: removed (%feV)", from_idx, to_idx, en)
                n_removed += 1
            else:
                logger.info("%d -> %d: kept (%feV)", from_idx, to_idx, en)
                neb_graph.set_edges(removed)
            progress_bar()  # pylint: disable=not-callable
    logger.info("Removed %d high-energy barriers", n_removed)

    # If requested, remove as many as possible diffusion paths,
    # starting from the longest, while keeping graph connectivity.
    if remove_compound:
        logger.info("Removing compound paths")
        edges = [(edge['distance'], from_idx, to_idx)
                 for from_idx, to_idx, edge in neb_graph.edges]
        n_edges = len(edges)
        with alive_bar(
                n_edges, title='Removing compound paths') as progress_bar:
            for edge_len, from_idx, to_idx in sorted(edges, reverse=True):
                removed = neb_graph.remove_edge(from_idx, to_idx)
                if _connected_and_infinite():
                    logger.debug(
                        "%d -> %d (%fÅ): removed",
                        from_idx, to_idx, edge_len
                    )
                    n_edges -= 1
                else:
                    logger.info(
                        "%d -> %d (%fÅ): kept",
                        from_idx, to_idx, edge_len
                    )
                    neb_graph.set_edges(removed)
                progress_bar()  # pylint: disable=not-callable
        logger.info("Found %d non-compound paths", n_edges)

    logger.info(
        "Final diffusion graph: %s",
        [(from_idx, to_idx) for from_idx, to_idx, _ in neb_graph.edges]
    )

    # Get rid of symmetrically equivalent diffusion paths.
    logger.info("Searching unique diffusion paths")
    pairs = []
    merged_pairs = []
    _known_dists = []
    edges = []
    dists = []
    for from_idx, to_idx, data in neb_graph.edges:
        if from_idx > to_idx:
            from_idx, to_idx = to_idx, from_idx
        if (from_idx, to_idx) not in edges:
            edges.append((from_idx, to_idx))
            dists.append(data['distance'])
    n_edges = len(edges)
    with alive_bar(
            n_edges,
            title='Removing equivalent paths') as progress_bar:
        for (from_idx, to_idx), dist in zip(edges, dists):
            merged = merge_structures(
                [all_clones[from_idx], all_clones[to_idx]])
            # Equivalent paths must have the same length
            # (assuming that structure_distance is robust enough)
            close_pair_idxs = [
                idx for idx, d in enumerate(_known_dists)
                if np.isclose(dist, d)]
            if len(close_pair_idxs) > 0 and structure_matches(
                    merged, [merged_pairs[idx] for idx in close_pair_idxs],
                    multithread=multithread):
                logger.debug("%s path is non-unique", (from_idx, to_idx))
                progress_bar()  # pylint: disable=not-callable
                continue
            pairs.append((all_clones[from_idx], all_clones[to_idx]))
            merged_pairs.append(merged)
            _known_dists.append(dist)
            progress_bar()  # pylint: disable=not-callable
    logger.info("Found %d unique paths", len(pairs))

    # Sort by lentgh
    pairs = sorted(
        pairs,
        key=lambda pair: structure_distance(
            pair[0], pair[1], tol=0.5, match_first=False))

    if return_unfiltered:
        unfiltered_pairs = [
            (all_clones[from_idx], all_clones[to_idx])
            for from_idx, to_idx, _ in neb_graph.edges
        ]
        return pairs, unfiltered_pairs

    return pairs
