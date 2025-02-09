"""NEB pair generator for diffusion paths.
"""
import logging
import warnings
from multiprocessing import Pool
from alive_progress import alive_bar
import numpy as np
import networkx as nx
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
        self.structures = structures
        self.multithread = multithread
        all_edges = self.__get_all_edges()
        edge_matrix = np.full(
            (len(structures), len(structures)), None, dtype=object)
        for from_idx, to_idx, dist, vec in all_edges:
            edge_matrix[from_idx, to_idx] = {
                'distance': dist, 'vector': np.array(vec)}
            edge_matrix[to_idx, from_idx] = {
                'distance': dist, 'vector': -np.array(vec)}
        self._edge_matrix = edge_matrix

    @property
    def edges(self):
        """Return a list of all edges.
        Each element of the list is (from_idx, to_idx, edge_data)
        where edge_data is a dict containing 'vector' and 'distance'
        entries representing distance between structures and vector
        moving from one structure to another.
        """
        # FIXME: Ideally, we should return an interator
        result = []
        for from_idx, row in enumerate(self._edge_matrix):
            for to_idx, data in enumerate(row):
                if data is not None:
                    result.append((from_idx, to_idx, data))
        return result

    def remove_edge(self, from_idx, to_idx):
        """Remove edge between FROM_IDX and TO_IDX structures.
        Remove opposite edge as well.
        Return a list of edges removed.
        """
        edge1 = (from_idx, to_idx, self._edge_matrix[from_idx, to_idx])
        edge2 = (to_idx, from_idx, self._edge_matrix[to_idx, from_idx])
        self._edge_matrix[from_idx, to_idx] = None
        self._edge_matrix[to_idx, from_idx] = None
        return [edge1, edge2]

    def remove_vertice_edges(self, idx):
        """Remove vertice edges for IDX structure.
        This will keep the structrue itself, but remove all its edges.
        Return the list of edges removed.
        """
        removed = []
        for to_idx, _ in enumerate(self.structures):
            removed += self.remove_edge(idx, to_idx)
        return removed

    def set_edges(self, edges):
        """Set EDGES.
        EDGES is a list of (from_idx, to_idx, edge_data_dict)
        """
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
        if progress_bar is not None:
            progress_bar()  # pylint: disable=not-callable
        return (from_idx, to_idx, distance, vec)

    def __get_all_edges(self):
        """Compute all edges for the NEB graph."""
        if self.multithread:
            return self._compute_edges_multithreaded()
        else:
            return self._compute_edges_singlethreaded()

    def _compute_edges_multithreaded(self):
        """Compute edges using multithreading."""
        with alive_bar(None, title='Computing distance matrix') as progress_bar:
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
        with alive_bar(len(self.structures) ** 2, title='Computing distance matrix') as progress_bar:
            return [
                self._get_edge(
                    from_idx, to_idx, from_struct, to_struct, progress_bar)
                for from_idx, from_struct in enumerate(self.structures)
                for to_idx, to_struct in enumerate(self.structures)
                if from_idx < to_idx
            ]

    def connected(self):
        """Return True when graph is connected.
        Otherwise, return False.
        """
        n_vertices = len(self.structures)
        # Visit matrix: False when we cannot reach a site; True - we can.
        visited = [False for _ in range(n_vertices)]

        queue = [0]

        while len(queue) > 0:
            from_idx = queue.pop(0)
            visited[from_idx] = True
            for to_idx, edge in enumerate(self._edge_matrix[from_idx]):
                if edge is None:
                    continue
                if not visited[to_idx]:
                    queue.append(to_idx)

        return all(visited)

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

        nx_G = nx.DiGraph()
        for from_idx, to_idx, edge in self.edges:
            if edge is not None:
                nx_G.add_edge(from_idx, to_idx)

        # FIXME: The number of cycles in dense graphs grows
        # exponentially with the number of edges.  So, this may take
        # too long and we need some kind of cutoff or other smart idea
        # to make things reasonable. Maybe somehow ensure that the
        # graph passed is sparse.
        for cycle in nx.simple_cycles(nx_G):
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
                    return True
        return False

    def get_min_cutoff(self):
        """Find the smallest distance cutoff keeping graph connected.
        DISTANCE_MATRIX is a square matrix representing diffusion path lengthes.

        Note: connected diffusion graph may theoretcally lead to all the
        diffusion paths be bound within finite volume.  Most of the time
        it should be fine though.  Good enough as an approximation.
        FIXME: Consider using full "infinite diffusion path" criterion
        instead.
        """
        # Visit matrix: False when we cannot reach a site; True - we can.
        visited = [False] * len(self.structures)

        with alive_bar(
                len(visited), title='Auto-detecting cutoff') as progress_bar:

            def bfs1(queue, dist_cutoff):
                while len(queue) > 0:
                    from_idx = queue.pop(0)
                    if not visited[from_idx]:
                        visited[from_idx] = True
                        progress_bar()  # pylint: disable=not-callable
                    distances = [(edge['distance'], to_idx)
                                 for to_idx, edge
                                 in enumerate(self._edge_matrix[from_idx])
                                 if not visited[to_idx]]
                    for distance, to_idx in sorted(distances):
                        if not visited[to_idx] and not distance > dist_cutoff:
                            logger.debug(
                                "coverage: %s -> %s (%f)",
                                from_idx, to_idx, distance
                            )
                            queue.append(to_idx)
            visited[0] = True
            progress_bar()  # pylint: disable=not-callable
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
    return scaled_structure


def _remove_duplicates(
        structures: list[Structure],
        multithread: bool = False,
) -> list[Structure|None]:
    """Remove duplicates from STRUCTURES.
    Return list of the same length with duplicate structures replaced
    with None.

    When MULTITHREAD is True, use multithreading.
    """
    uniq_structures = []
    logger.info(
        "gen_neb_pairs: Checking duplicates among %d structures...",
        len(structures))
    with alive_bar(len(structures), title='Checking duplicates')\
         as progress_bar:
        for struct in structures:
            if not structure_matches(
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
        multithread: bool = False
) -> list[tuple[Structure, Structure]]:
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
    """
    # Remove duplicates.
    uniq_structures = _remove_duplicates(structures, multithread)
    # Scale everything to at least 2x2x2 supercell.
    prototype, uniq_structures = \
        __scale_structures_maybe(prototype, uniq_structures)
    # Arrange 1-to-1 site matching in all the provided structures
    for idx, struct in enumerate(uniq_structures):
        if idx != 0 and struct is not None:
            uniq_structures[idx] =\
                get_matched_structure(uniq_structures[0], struct)

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

    neb_graph = _NEB_Graph(all_clones, multithread=multithread)

    if cutoff == 'auto':
        logger.info("Determining minimal possible diffusion distance cutoff")
        cutoff = neb_graph.get_min_cutoff()
        logger.info('Found optimal cutoff to cover all sites: %f', cutoff)

    assert isinstance(cutoff, float)
    # Only keep egdes shorter than cutoff
    n_edges = 0
    for from_idx, to_idx, edge in neb_graph.edges:
        edge_len = edge['distance']
        if from_idx > to_idx:
            continue
        if not edge_len < cutoff:
            neb_graph.remove_edge(from_idx, to_idx)
        else:
            n_edges += 1
    logger.info("Found %d paths shorter than cutoff (%f)", n_edges, cutoff)

    # for idx, _ in enumerate(neb_graph.structures):
    #     assert neb_graph.diffusion_path_infinite(idx)

    logger.info("Removing high-energy structures")
    energies = []
    for idx, struct in enumerate(neb_graph.structures):
        energy = struct.properties.get('final_energy')
        if energy is None:
            origin_path = struct.properties.get('origin_path')
            raise ValueError(f"Energy data is missing for {origin_path}")
        logger.debug("Energy %d: %f", idx, energy)
        energies.append(energy)
    # Only take the lowest-energy structure
    energy_threshold = sorted(np.unique(energies))[0]
    logger.info("Energy theshold: %f", energy_threshold)

    # Loop over structures above the threshold and remove them if such
    # removal does not break infinite diffusion path for low-energy
    # structures.
    low_en_idxs = [
        idx for idx in np.argsort(energies)
        if not energies[idx] > energy_threshold]
    high_en_idxs = [
        idx for idx in reversed(np.argsort(energies))
        if energies[idx] > energy_threshold]

    def __infinitely_connected():
        for idx in low_en_idxs:
            connected = neb_graph.diffusion_path_infinite(idx)
            if not connected:
                return False
        return True

    # assert __infinitely_connected()
    n_removed = 0
    with alive_bar(
            len(high_en_idxs),
            title='Removing high-energy configurations'
    ) as progress_bar:
        for idx in high_en_idxs:
            removed_edges = neb_graph.remove_vertice_edges(idx)
            if not __infinitely_connected():
                logger.info("%d: kept (%f)", idx, energies[idx])
                neb_graph.set_edges(removed_edges)
            else:
                logger.debug("%d: removed (%f)", idx, energies[idx])
                n_removed += 1
            progress_bar()  # pylint: disable=not-callable
    logger.info("Removed %d high-energy configurations", n_removed)

    # if remove_compound:
    #     logger.info("Removing compound paths")

    #     edges = []
    #     for from_idx, to_idx, edge in neb_graph.edges:
    #         if from_idx > to_idx:
    #             continue
    #         edge_len = edge['distance']
    #         edge_vec = edge['vector']
    #         edges.append((edge_len, from_idx, to_idx))

    #     with alive_bar(
    #             n_edges,
    #             title='Removing compound paths') as progress_bar:
    #         assert _diffusion_path_infinite(distance_vector_matrix, 0)
    #         for edge_len, from_idx, to_idx in sorted(edges, reverse=True):
    #             if edge_len == np.inf:
    #                 continue
    #             edge_vec = distance_vector_matrix[from_idx][to_idx]
    #             edge_vec_rev = distance_vector_matrix[to_idx][from_idx]
    #             distance_matrix[from_idx, to_idx] = np.inf
    #             distance_matrix[to_idx, from_idx] = np.inf
    #             distance_vector_matrix[from_idx, to_idx] = None
    #             distance_vector_matrix[to_idx, from_idx] = None
    #             if _diffusion_path_infinite(distance_vector_matrix, 0):
    #                 logger.debug(
    #                     "%d -> %d (%f): removed",
    #                     from_idx, to_idx, edge_len
    #                 )
    #                 n_edges -= 1
    #             else:
    #                 distance_matrix[from_idx, to_idx] = edge_len
    #                 distance_matrix[to_idx, from_idx] = edge_len
    #                 distance_vector_matrix[from_idx, to_idx] = edge_vec
    #                 distance_vector_matrix[to_idx, from_idx] = edge_vec_rev
    #                 logger.info(
    #                     "%d -> %d (%f): kept",
    #                     from_idx, to_idx, edge_len
    #                 )
    #             progress_bar()  # pylint: disable=not-callable
    #     logger.info("Found %d non-compound paths", n_edges)

    logger.info("Searching unique diffusion paths")
    pairs = []
    merged_pairs = []
    _known_dists = []
    with alive_bar(
            len(neb_graph.structures)**2,
            title='Removing equivalent paths') as progress_bar:
        for from_idx, to_idx, edge in neb_graph.edges:
            progress_bar()  # pylint: disable=not-callable
            if from_idx > to_idx or edge is None:
                continue
            merged = merge_structures(
                [all_clones[from_idx], all_clones[to_idx]])
            # Equivalent paths must have the same length
            # (assuming that structure_distance is robust enough)
            if np.any([np.isclose(edge['distance'], d) for d in _known_dists])\
               and structure_matches(merged, merged_pairs,
                                     multithread=multithread):
                continue
            pairs.append((all_clones[from_idx], all_clones[to_idx]))
            merged_pairs.append(merged)
            _known_dists.append(edge['distance'])
    logger.info("Found %d unique paths", len(pairs))

    # Sort by lentgh
    pairs = sorted(
        pairs,
        key=lambda pair: structure_distance(
            pair[0], pair[1], tol=0.5, match_first=False))

    return pairs
