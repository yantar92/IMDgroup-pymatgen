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


"""Insert molecules and atoms into a given structure.
"""

import logging
from multiprocessing import Pool
from alive_progress import alive_bar
import numpy as np
from numpy.typing import ArrayLike
from pymatgen.transformations.transformation_abc import AbstractTransformation
from pymatgen.core import (Structure, Molecule, PeriodicSite)
from pymatgen.util.typing import SpeciesLike
from pymatgen.core.periodic_table import get_el_sp
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from IMDgroup.pymatgen.core.structure import reduce_supercell

__author__ = "Ihor Radchenko <yantar92@posteo.net>"

logger = logging.getLogger(__name__)


class InsertMoleculeTransformation(AbstractTransformation):
    """Create structures with inserted molecules."""

    def __init__(
            self,
            molecule: Molecule | SpeciesLike | str,
            step: float,
            step_noise: float | None = None,
            anglestep: float | None = None,
            proximity_threshold: float = 0.75,
            label: str | None = "insert",
            selective_dynamics: ArrayLike | None = None,
            reduce_supercell: bool = True,
            matcher: StructureMatcher =
            StructureMatcher(attempt_supercell=True, scale=False),
            multithread = False
    ):
        """Add molecule to structure.

        Args:
        molecule (Molecule or SpeciesLike or filename):
                molecule or atom to insert into structure
        step (float or None):
                step, in ans, when searching for insertion sites
                Default: 0.5 ans
        step_noise (float or None):
                standard deviation (as a fraction of step) for
                randomness added to the scan grid.
                When negative number, make the grid fully random, with
                the number of grid points equal to int(abs(step_noise)).
                Default: None (no randomness)
        anglestep (float or None):
                angle step, in radians, when trying different molecule
                rotations.  Must be None when inserting an atom
        proximity_threshold (float, default=0.75):
          Threshold multiplier to judge that two atoms are too close
          to each other.  The atoms are considered too close when
          distance < proximity_threshold * (atom1.atomic_radius + atom2.atomic_radius)
        label (str):
          Label to mark the inserted molecule.  The label will be used
          to construct labels for each atom in the molecule as
          label-<atom_name><site_index>
        selective_dynamics (array[bool]):
          selective_dynamics settings for the inserted atoms (only
          used when the structure also sets selective_dynamics
          properties).
        reduce_supercell (bool; default: True):
          Whether to try reducing structure from supercell to
          primitive cell before searching for insertions.
        matcher (StructureMatcher or None):
          Additional matcher to be used to detect duplicates.
        multithread (bool; default: False):
          Whether to use multithreading.
        """
        if step is None:
            step = 0.5
        if isinstance(molecule, SpeciesLike):
            try:
                molecule = Molecule([get_el_sp(molecule)], [[0, 0, 0]])
            except ValueError as e:
                if isinstance(molecule, str):
                    # Try to parse as filename
                    molecule = Molecule.from_file(molecule)
                else:
                    raise\
                        ValueError(
                            f"Can't parse Molecule or SpeciesLike\
                            or filename from {molecule!r}") from e
        if anglestep is not None:
            assert isinstance(molecule, Molecule) and len(molecule) > 1, \
                "Cannot rotate non-molecule or molecule with a single atom."
        self.molecule = molecule
        self.step = step
        self.step_noise = step_noise
        self.anglestep = anglestep
        self.proximity_threshold = proximity_threshold
        self.label = label
        self.selective_dynamics = selective_dynamics
        self._candidate_angles = None
        if len(molecule) > 1 and self.anglestep is not None:
            self._candidate_angles = self._get_angle_grid()
        self.reduce_supercell = reduce_supercell
        self.matcher = matcher
        self.multithread = multithread

    def _get_site_grid(self, structure: Structure):
        """Generate a list of candidate fractional coordinates.
                Args:
                structure (Strcuture): Structure to put candidates into
                Returns:
                List of 3x1 lists representing fractional coordinates.
                """
        # Try to scale down a supercell.  It is pointless to search
        # supercell beyond the underlying primitive structure.
        if self.reduce_supercell:
            logger.debug("Attempting to scale down the original structure")
            reduced_structure = reduce_supercell(structure)
            logger.info(
                "%s",
                'Detected '
                f'{structure.lattice.a/reduced_structure.lattice.a:.2}'
                f'x{structure.lattice.b/reduced_structure.lattice.b:.2}'
                f'x{structure.lattice.c/reduced_structure.lattice.c:.2}'
                " supercell. Limiting scan volume."
            )
        else:
            reduced_structure = structure

        # Fully random grid requested.
        if self.step_noise is not None and self.step_noise < 0:
            return [
                [np.random.uniform(
                    low=0.0,
                    high=reduced_structure.lattice.a/structure.lattice.a),
                 np.random.uniform(
                     low=0.0,
                     high=reduced_structure.lattice.b/structure.lattice.b),
                 np.random.uniform(
                     low=0.0,
                     high=reduced_structure.lattice.c/structure.lattice.c)]
                for _ in range(int(abs(self.step_noise)))
            ]

        xrange = np.arange(
            0.0, reduced_structure.lattice.a/structure.lattice.a,
            self.step/structure.lattice.a)
        yrange = np.arange(
            0.0, reduced_structure.lattice.b/structure.lattice.b,
            self.step/structure.lattice.b)
        zrange = np.arange(
            0.0, reduced_structure.lattice.c/structure.lattice.c,
            self.step/structure.lattice.c)

        def _random(scale):
            if self.step_noise is None:
                return 0
            return self.step_noise * scale\
                * np.random.standard_normal()

        def xrandom():
            return _random(self.step/structure.lattice.a)

        def yrandom():
            return _random(self.step/structure.lattice.b)

        def zrandom():
            return _random(self.step/structure.lattice.c)

        return [[x + xrandom(), y + yrandom(), z + zrandom()]
                for x in xrange for y in yrange for z in zrange]

    def _get_angle_grid(self):
        """Generate a list of candidate Euler angles to rotate molecule.
        Args:
          molecule (Molecule): Molecule to rotate
        Returns:
          List of 3x1 lists representing Euler angle triplets, in radians.
        """
        assert self.anglestep
        assert len(self.molecule) > 1
        rotation_sym_num =\
            PointGroupAnalyzer(self.molecule).get_rotational_symmetry_number()
        angrange = np.arange(0.0, 2*np.pi/rotation_sym_num, self.anglestep)
        return [[alpha, beta, gamma]
                for alpha in angrange
                for beta in angrange
                for gamma in angrange]

    def _get_largest_radius(self, obj):
        """Find the largest atomic radius across species in structurelike.
        obj is a Molecule or Structure.
        """
        return max(s.atomic_radius for s in obj.species)

    def _check_proximity(self, structure: Structure, site: int | PeriodicSite,
                         cutoff: float | None = None):
        """Return True if site is far enough from other sites in STRUCTURE.
        Far enough is further than a sum of the atomic radiuses.

        Args:
          structure (Structure): structure to be checked
          site (int | PeriodicSite): site index
          cutoff (float): cutoff radius in ans to search neighbours to test
        """
        if isinstance(site, int):
            site = structure[site]
        if cutoff is None:
            cutoff = self._get_largest_radius(structure)
        # Finding neighbors is extremely fast, faster than direct
        # distance calculation.
        _, distances, neighbor_indices, _ =\
            structure.lattice.get_points_in_sphere(
                frac_points=structure.frac_coords,
                center=site.coords,  # Cartesian
                r=cutoff + site.specie.atomic_radius,
                zip_results=False)
        for nidx, distance in zip(neighbor_indices, distances):
            max_distance =\
                self.proximity_threshold * (
                    structure[nidx].specie.atomic_radius
                    + site.specie.atomic_radius
                )
            if distance < max_distance and site != structure[nidx]:
                return False
        return True

    def _check_molecule_proximity(self, structure1, structure2, cutoff):
        """Check if molecules in the two structures are distinct.
        Args:
          structure1/2 (Structure):
            Structures containing molecule and nothing else.
          cutoff (float): cutoff radius

        Each structure provided must only contain molecule to be
        checked, with nodes corresponding to molecule atoms, in the
        same order.

        Structures are considered the same if each site in the
        structure is too close to the corresponding site in another
        structure.  Too close is closer than sum of atomic radii.

        The idea is to treat molecule rotations that bring at least
        one atom in the molecule more than its diameter away from the
        old location as distinct.

        Returns True when structures are distinct.
        """
        for atom in structure1:
            if self._check_proximity(structure2, atom, cutoff):
                # At least one atom is far enough.
                return True
        return False

    def rotate_molecule_euler(self, euler_angle: ArrayLike):
        """Rotate self.molecule according to EULER_ANGLE.

        Args:
                    molecule (Molecule):
                    Molecule to be rotate
                    euler_angle (3x1 vector):
                    Triplet of Euler angles in radians

        Returns:
                    Rotated molecule.
                    """
        if euler_angle is None:
            return self.molecule
        molecule = self.molecule.copy()
        # https://en.wikipedia.org/wiki/Euler_angles (Conventions
        # by extrinsic rotations)
        molecule.rotate_sites(
            theta=euler_angle[2], axis=[0, 0, 1],
            anchor=molecule[0].coords)
        molecule.rotate_sites(
            theta=euler_angle[1], axis=[1, 0, 0],
            anchor=molecule[0].coords)
        molecule.rotate_sites(
            theta=euler_angle[0], axis=[0, 0, 1],
            anchor=molecule[0].coords)
        return molecule

    def _insert_molecule(
            self,
            structure: Structure,
            coords: ArrayLike,
            euler_angle=None,
            known_inserts=None,
            cutoff=None
    ):
        """Insert self.molecule into STRUCTURE at COORDS.
        The MOLECULE will be rotated around its first atom
        according to EULER_ANGLE.  Then, placed at COORDS.  STRUCTURE
        is not modified.

        Args:
          structure (Structure):
            atructure to insert the molecule into
          coords (3x1 vector):
            fractional cell coordinates
          euler_angle (3x1 vector):
            triplet of Euler angles in radians
          known_inserts (list of Structure):
            List of structures containing known insert positions of
            MOLECULE.  The structures must only have sites
            corresponding to the molecule atoms and no other sites.
            known_inserts will be modified by side effect, adding the
            current insert.
          cutoff (float or None):
            Cutoff radius to search for close neighbors.

        Returns:
        Modified STRUCTRURE with MOLECULE inserted at the end of the
        site list or None when insertion fails.  The insertion fails,
        the structure is not modified.
        """
        # Apply rotation
        # Do it before translating to place, because otherwise boundary
        # conditions may cause funny things to happen.
        if euler_angle is not None:
            molecule = self.rotate_molecule_euler(euler_angle)
        else:
            molecule = self.molecule

        # Insert molecule atoms at arbitrary positions
        molecule_indices = list(range(len(molecule)))

        def undo_insert():
            """Undo the insertion."""
            structure.remove_sites(molecule_indices)

        structure_has_selective_dynamics =\
            'selective_dynamics' in structure[0].properties
        for idx, atom in enumerate(molecule):
            structure.insert(
                idx=idx, species=atom.species, coords=atom.coords,
                coords_are_cartesian=True, validate_proximity=False)
            if self.label is not None:
                structure[idx].label = self.label + "-"\
                    + structure[idx].specie.symbol + str(idx)
            if structure_has_selective_dynamics \
               and self.selective_dynamics is not None:
                structure[idx].properties['selective_dynamics'] = \
                    self.selective_dynamics

        # Now, move them as needed to the target coordinates.
        anchor = structure[0].frac_coords
        relative_coords = coords - anchor
        structure.translate_sites(
            molecule_indices, relative_coords, frac_coords=True)

        # Make sure that the inserted molecule is not
        # too close to existing sites.
        if cutoff is None:
            cutoff = self._get_largest_radius(structure)
        if False in [self._check_proximity(structure, index, cutoff)
                     for index in molecule_indices]:
            logger.debug("<skip>")
            undo_insert()
            return None

        # When known_inserts is provided, validate that we do not insert a
        # duplicate.
        if known_inserts is not None:
            if cutoff is None:
                cutoff = self._get_largest_radius(structure)
            # Create a copy of structure with only molecule sites
            structure_w_molecule = Structure(
                lattice=structure.lattice, species=molecule.species,
                coords=[structure[idx].frac_coords for idx in molecule_indices]
            )
            for known in known_inserts:
                if not self._check_molecule_proximity(
                        structure_w_molecule, known, cutoff):
                    logger.debug("<skip>")
                    undo_insert()
                    return None
            known_inserts.append(structure_w_molecule)

        return structure

    def _generate_inserts(
            self, structure: Structure,
            limit: int | None = None):
        """Generate all possible inserts of self.molecule into STRUCTURE.
        Args:
          structure (Structure): Structure to insert into
          limit (int): Limit the number of structures.  When negative,
          search across random grid.

        Returns:
        List of structures with inserted molecule.
        """
        logger.info("Generating inserts...")
        logger.info("Molecule:\n---\n%s\n---", self.molecule)
        logger.info("Matrix:\n---\n%s\n---", structure)

        candidate_coords = self._get_site_grid(structure)
        candidate_angles = self._candidate_angles
        structure_inserts = []
        # Inserted positions.  To be used for sorting.
        structure_inserts_positions = []
        known_inserts = []

        cutoff = max([self._get_largest_radius(structure),
                      self._get_largest_radius(self.molecule)])

        if candidate_angles is None:
            n_candidates = len(candidate_coords)
        else:
            n_candidates = len(candidate_angles)*len(candidate_coords)
        if limit is not None and limit < 0:
            limit = abs(limit)
            # randomize grids
            np.random.shuffle(candidate_coords)
            if candidate_angles is not None:
                np.random.shuffle(candidate_angles)

        with alive_bar(n_candidates, enrich_print=False,
                       dual_line=True, spinner='bubbles')\
             as progress_bar:

            def append_inserts_fixed_rotation(
                    euler_angle: ArrayLike | None = None):
                """Insert rotated molecule into all possible positions.
                Modify structure_inserts by side effect, adding all
                possible positions of self.molecule into structure with
                self.molecule rotated by a fixed euler_angle.

                Args:
                  euler_angle(3x1 vector or None):
                    Fixed rotation angle
                    """
                # Accumulate all the inserts together, to automatically
                # filter out the inserts that are too close to each other.
                accumulate = structure.copy()
                for coords in candidate_coords:
                    progress_bar()  # pylint: disable=not-callable
                    new = self._insert_molecule(
                        structure=accumulate, coords=coords,
                        euler_angle=euler_angle,
                        known_inserts=known_inserts,
                        cutoff=cutoff)
                    if new is not None:
                        insert = self._insert_molecule(
                            structure.copy(), coords, euler_angle,
                            cutoff=cutoff)
                        previous_matches = False
                        if self.matcher:
                            if self.multithread:
                                with Pool() as pool:
                                    equivs = pool.starmap(
                                        self.matcher.fit,
                                        [(insert, x)
                                         for x in structure_inserts])
                                    if True in equivs:
                                        previous_matches = True
                            else:
                                for x in structure_inserts:
                                    if insert != x and self.matcher.fit(insert, x):
                                        previous_matches = True
                                        break
                        if self.matcher is None or not previous_matches:
                            structure_inserts.append(insert)
                            structure_inserts_positions.append(
                                [coords, euler_angle]
                            )
                            if euler_angle is None:
                                log_message =\
                                    f"#{len(structure_inserts)} " +\
                                    "New insert :: " +\
                                    f"pos={coords} No rotation"
                                progress_bar.text = log_message
                                logger.info("%s", log_message)
                            else:
                                log_message =\
                                    f"#{len(structure_inserts)} " +\
                                    "New insert :: " +\
                                    f"euler={euler_angle} pos={coords}"
                                progress_bar.text = log_message
                                logger.info("%s", log_message)
                            if limit is not None\
                               and len(structure_inserts) > limit:
                                break
                        else:
                            logger.info('complex mather found a duplicate!')
                            del known_inserts[-1]

            if candidate_angles is None:
                append_inserts_fixed_rotation()
            else:
                for euler_angle in candidate_angles:
                    append_inserts_fixed_rotation(euler_angle)
                    if limit is not None\
                       and len(structure_inserts) > limit:
                        break

        logger.info("Found %d candidates", len(structure_inserts))

        result = structure_inserts
        sorted_idx =\
            [i[0] for i in sorted(
                enumerate(structure_inserts_positions),
                key=lambda x: x[1])]
        for i, idx in enumerate(sorted_idx):
            result[i] = structure_inserts[idx]
        return result

    def all_inserts(
            self,
            structure: Structure | str,
            limit: int | None = None
    ):
        """Generate all possible molecule inserts into structure.

        Args:
          structure (Structure or filename):
            Structure to insert into.
          limit (int | None, optional):
            If int, return no more than that number of structures.
            If negative, randomize the structure search.

        Returns:
          List of structures with self.molecule inserted.
        """
        if isinstance(structure, str):
            structure = Structure.from_file(structure)
        return self._generate_inserts(structure, limit)

    def apply_transformation(
            self,
            structure: Structure | str,
            return_ranked_list: bool | int = False
    ):
        """Transform structure, adding a single molecule.

        Args:
          structure (Structure or filename):
            Structure to insert into.
          return_ranked_list (bool | int, optional):
            If int, return that number of structures.

        Returns:
          Transformed structure or a list of dictionaries.
        """
        if not return_ranked_list:
            return self._generate_inserts(structure, 1)[0]
        return [{"structure": structure}
                for structure in self._generate_inserts(
                        structure, return_ranked_list)]

    @property
    def is_one_to_many(self) -> bool:
        """Transform one structure to many."""
        return True


def get_all_molecule_inserts(
        molecule: Molecule | SpeciesLike | str,
        structure: Structure | str,
        step: float,
        anglestep: float | None = None,
        label: str | None = "insert",
        limit: int | None = None):
    """Generate all possible molecule inserts into structure.
        Args:
          molecule (Molecule, SpeciesLike, or filename):
            molecule or atom to insert into structure
          structure (Structure or filename):
            structure to insert into.
          limit (int | None, optional):
            if int, return no more than that number of structures.
            if negative, randomize the structure search.

        Returns:
          List of structures with molecule inserted.
    """
    transformer = InsertMoleculeTransformation(
        molecule, step=step,
        anglestep=None if anglestep is None else np.radians(anglestep),
        label=label)
    return transformer.all_inserts(structure, limit)
