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


"""Visualization extension specific to IMD group.
"""
import logging
import os
import re
import warnings
from termcolor import colored
from pathlib import Path
from alive_progress import alive_it
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
from IMDgroup.pymatgen.core.structure import IMDStructure as Structure
from IMDgroup.pymatgen.core.structure import merge_structures, structure_distance
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
from IMDgroup.pymatgen.io.vasp.sets import write_selective_dynamics_summary_maybe
import IMDgroup.pymatgen.io.atat as IMDatat
import itertools
from pymatgen.core import Element, Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.computed_entries import (
    ComputedStructureEntry,
    GibbsComputedStructureEntry,
)
from pymatgen.apps.battery.insertion_battery import InsertionElectrode
from pymatgen.apps.battery.plotter import VoltageProfilePlotter
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

logger = logging.getLogger(__name__)


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Visualize vasp outputs."""

    parser.add_argument(
        "dir",
        help="""Directory to read (recursively) or a pickle file for hull subcommand; defaults to current dir""",
        type=str,
        nargs="?",
        default=".")

    subparsers = parser.add_subparsers(required=True)

    parser_neb = subparsers.add_parser("neb")
    neb_add_args(parser_neb)

    parser_selective_dynamics = subparsers.add_parser("selective_dynamics")
    selective_dynamics_add_args(parser_selective_dynamics)

    parser_atat = subparsers.add_parser("atat")
    atat_add_args(parser_atat)

    parser_hull = subparsers.add_parser("hull")
    hull_add_args(parser_hull)

    parser_voltage = subparsers.add_parser("voltage")
    voltage_add_args(parser_voltage)


def neb_add_args(parser):
    """Setup parser arguments for neb visualization.
    Args:
      parser: subparser
    """
    parser.help = "Visualize NEB outputs"
    parser.set_defaults(func_derive=neb)


def neb(args):
    """Create NEB visualization.
    """
    for wdir, subdirs, _ in os.walk(args.dir):
        subdirs.sort()  # this will make loop go in order
        vaspdir = IMDGVaspDir(wdir)
        if not vaspdir.nebp:
            continue
        if not vaspdir.converged:
            logger.info("Skipping unconverged run at %s", wdir)
            continue
        neb_structures = [
            (imagedir.final_structure or imagedir.structure)
            for imagedir in vaspdir.neb_dirs()]
        trajectory = merge_structures(neb_structures)
        cif_name = 'NEB_trajectory_converged.cif'
        output_cif = os.path.join(wdir, cif_name)
        logger.info("Saving final trajectory to %s", output_cif)
        print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
              + colored("NEB ", "magenta")
              + f"Saved final trajectory to {cif_name}")
        trajectory.to_file(output_cif)
    return 0


def selective_dynamics_add_args(parser):
    """Setup parser arguments for selective dynamics visualization.
    Args:
      parser: subparser
    """
    parser.help = "Visualize selective dynamics for POSCAR files"
    parser.set_defaults(func_derive=selective_dynamics)


def selective_dynamics(args):
    """Visualize selective_dynamics.
    """
    for parent, subdirs, files in os.walk(args.dir):
        subdirs.sort()  # this will make loop go in order
        if 'POSCAR' in files:
            structure = Structure.from_file(os.path.join(parent, 'POSCAR'))
            cif_name = 'selective_dynamics.cif'
            cif_path = os.path.join(parent, cif_name)
            write_selective_dynamics_summary_maybe(structure, cif_path)
            logger.info("Saving illustration to %s", cif_path)
            print(colored(f"{parent.replace("./", "")}: ", attrs=['bold'])
                  + f"Saved selective_dynamics to {cif_name}")


def atat_add_args(parser):
    """Setup parser arguments for ATAT visualization.
    Args:
      parser: subparser
    """
    parser.help = "Visualize ATAT outputs"
    parser.add_argument(
        "--plot_extra",
        help="Extra data to plot. "
        "It is a list of paths mirroring ATAT folder structure. "
        "the paths will be searched for structured from fit.out and "
        "their energy will be plotted.",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--plot_extra_threshold",
        help="Minimal energy difference for --plot_extra data to appear on the plot. "
        "(default: 0.001eV/reference cell)",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--cmin",
        help="Min concentration to be used for convex hull"
        "(default: read from maps.log)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--cmax",
        help="Max concentration to be used for convex hull"
        "(default: read from maps.log)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--emin",
        help="Min energy for displaying on convex hull"
        "(default: auto)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--emax",
        help="Max energy for displaying on convex hull"
        "(default: auto)",
        type=float,
        default=None
    )
    parser.add_argument(
        "--classic_residuals",
        help="When set, plot classic energy/energy correlation in the redisuals of the fit",
        action="store_true",
    )
    parser.set_defaults(func_derive=atat)


def _atat_read_predstr(path: Path) -> pd.DataFrame:
    """Read predstr.out.
    """
    return pd.read_csv(
        path / 'predstr.out', sep=' ', header=None,
        names=['concentration', 'energy', 'predicted energy',
               'index',
               # b = busy, e = error, u = unknown (not calculated),
               # bg/eg/ug = ground state
               'status']
    )


def _atat_read_gs(path: Path) -> pd.DataFrame:
    """Read gs.out.
    """
    return pd.read_csv(
        path / 'gs.out', sep=' ', header=None,
        names=['concentration', 'energy', 'fitted energy', 'index']
    )


def _atat_read_fit(path: Path) -> pd.DataFrame:
    """Read fit.out.
    """
    return pd.read_csv(
        path / 'fit.out', sep=' ', header=None,
        names=['concentration', 'energy',
               'fitted energy', 'energy delta',
               'weight', 'index'])


def _atat_read_clusters(path: Path) -> pd.DataFrame:
    """Read clusters.out and eci.out files and return a DataFrame with cluster data."""
    cluster_sizes = []
    cluster_diams = []
    eci_weights = []

    with open(path / 'clusters.out', 'r', encoding='utf-8') as clusters, \
         open(path / 'eci.out', 'r', encoding='utf-8') as eci:
        try:
            while True:
                next(clusters)  # Skip cluster index
                cluster_diam = float(next(clusters))
                cluster_size = int(next(clusters))
                eci_weight = float(next(eci))

                if cluster_size >= 2:
                    cluster_sizes.append(cluster_size)
                    cluster_diams.append(cluster_diam)
                    eci_weights.append(eci_weight)

                # Skip cluster basis functions
                for _ in range(cluster_size):
                    next(clusters)

                # Handle tensor format if present
                spacing = next(clusters)
                if spacing == "tensor":
                    rank = int(next(clusters))
                    for _ in range(rank):
                        next(clusters)
                    ndata = int(next(clusters))
                    for _ in range(ndata):
                        next(clusters)
                    next(clusters)  # Skip final newline

        except StopIteration:
            pass

    return pd.DataFrame({
        'cluster_size': cluster_sizes,
        'cluster_diameter': cluster_diams,
        'eci_weight': eci_weights
    })


def _atat_read_extra(
        fit: pd.DataFrame,
        path: Path,
        e0: float, e1: float,
        ref_structure_len: int) -> pd.DataFrame:
    """Read energies from PATH mirrowing ATAT folders.
    Only consider structure indices from FIT + "1" folder containing
    reference structure for energy normalization.
    Return dataframe with 'concentration', 'energy', and 'index' fields.
    The dataframe name will be set to PATH.
    Energy is calculated from XXX/energy file contants, normalized by XXX/str.out
    structure size vs. REF_STRUCTURE_LEN reference size.  The energy is then adjusted
    for convex hull according to "0" and "1" energies.
    When "0" or "1" is not available, e0 and e1 energies (presumably
    from ref_energy.out) are used.
    """
    concentrations = []
    energies = []
    indices = []
    e0_file = Path(path) / "0" / "energy"
    e1_file = Path(path) / "1" / "energy"
    if e0_file.is_file() and e1_file.is_file():
        logger.debug("Reading reference energies from %s", path)
        e0_len = len(Structure.from_file(Path(path) / "0" / "str.out"))
        e0 = float(e0_file.read_text(encoding='utf-8')) / (e0_len / ref_structure_len)
        e1_len = len(Structure.from_file(Path(path) / "1" / "str.out"))
        e1 = float(e1_file.read_text(encoding='utf-8')) / (e1_len / ref_structure_len)
    else:
        raise IOError(f"0 and 1 reference energies are not available in {path}")
    for concentration, index in alive_it(
            zip(fit['concentration'], fit['index']),
            title=f'Reading extra energies from {path}',
            total=len(fit)):
        energy_file = Path(path) / str(index) / 'energy'
        if not energy_file.is_file():
            continue
        tot_energy = float(energy_file.read_text(encoding='utf-8'))
        structure_len = len(Structure.from_file(Path(path) / str(index) / "str.out"))
        energy_per_cell = tot_energy / (structure_len / ref_structure_len)
        energy = energy_per_cell - concentration * e1 - (1 - concentration) * e0
        concentrations.append(concentration)
        indices.append(index)
        energies.append(energy)
    df = pd.DataFrame({
        'concentration': concentrations,
        'energy': energies,
        'index': indices
    })
    df.name = str(path)
    return df


def _atat_plot_clusters(
        ax: plt.Axes,
        clusters: pd.DataFrame) -> None:
    """Plot ECI fit coefficients for all the clusters."""
    ax.set_xlabel("Diameter, Å")
    ax.set_ylabel("Cluster energy, eV")
    ax.set_title("ECI vs cluster diameter")
    ax.axhline(0, color='black', linewidth=0.5)

    xticks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    xticklabels = [
        "pair", "5", "10", "15",
        "trip", "5", "10", "15",
        "quad", "5", "10", "15"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_xlim(-5, 60)
    ax.scatter(
        (clusters['cluster_size'] - 2) * 20 + clusters['cluster_diameter'],
        clusters['eci_weight'])


def _atat_plot_fitted_energies(
        ax: plt.Axes,
        predstr: pd.DataFrame,
        gs: pd.DataFrame,
        fit: pd.DataFrame,
        c_range: tuple[float, float] = (0.0, 1.0),
        erange: tuple[float, float] | None = None) -> None:
    """Plot Fitted energies at AX axis.
    """
    newgs = predstr[predstr['status'].str.contains('g')]

    ax.set_title('Fitted Energies')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    ax.set_xlim(c_range[0], c_range[1])
    if erange:
        ax.set_ylim(erange[0], erange[1])

    ax.plot(
        predstr['concentration'], predstr['predicted energy'],
        'o', label='predicted')
    ax.plot(
        fit['concentration'], fit['fitted energy'],
        'o', label='known str')
    ax.plot(
        gs['concentration'], gs['fitted energy'],
        'o-', fillstyle='none',
        label='fitted gs', color='black')
    ax.plot(
        newgs['concentration'], newgs['predicted energy'],
        's', fillstyle='none', label='predicted gs', color='red')
    ax.legend()


def _atat_plot_calc_vs_fit_energies(
        ax: plt.Axes,
        fit: pd.DataFrame,
        c_range: tuple[float, float] = (0.0, 1.0),
        erange: tuple[float, float] | None = None,
        ) -> None:
    """Plot Fitted vs Calculated energies at AX axis.
    """
    ax.set_title('Calculated and Fitted Energies')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    ax.set_xlim(c_range[0], c_range[1])
    if erange:
        ax.set_ylim(erange[0], erange[1])

    ax.plot(fit['concentration'], fit['energy'], 'P', label='calculated')
    ax.plot(fit['concentration'], fit['fitted energy'], 'o', label='fitted')
    ax.legend()


def __blue_orrd_cmap(data_min, data_max, color='blue', split_val=0.1):
    """Return colormap with COLOR for 0..SPLIT_VAL and gradient for the rest.
    Return None when data_min == data_max.
    """
    if data_max == data_min:
        return None
    n = 256  # resolution
    # Set the relative position of split_val
    frac = (split_val - data_min) / (data_max - data_min)
    n_fixed_color = int(frac * n)
    n_orrd = n - n_fixed_color
    greens = np.tile([to_rgba(color)], (n_fixed_color, 1))
    orrd = plt.get_cmap("OrRd", n_orrd)
    orrd_colors = orrd(np.linspace(0, 1, n_orrd))
    return ListedColormap(np.vstack([greens, orrd_colors]))


def _atat_plot_residuals(
        ax: plt.Axes,
        cve: str,
        fit: pd.DataFrame,
        classic_residuals: bool = False) -> None:
    """Plot fit error at AX axis.
    When CLASSIC_RESIDUALS is True, plot classic energy/energy correlation
    as residuals of the fit.
    """
    ax.set_title(f'Residuals of the fit\n{cve}')
    if classic_residuals:
        ax.set_xlabel('DFT energy per reference cell, eV')
        ax.set_ylabel('Predicted energy per reference cell, eV')
    else:
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Energy per reference cell, eV')
    displ = np.array(fit['sublattice deviation'], dtype=float)
    cmap = __blue_orrd_cmap(
        np.nanmin(displ), np.nanmax(displ),
        color=ax._get_lines.get_next_color())
    sc = ax.scatter(
        fit['energy'] if classic_residuals else fit['concentration'],
        fit['fitted energy'] if classic_residuals else fit['energy delta'],
        c=displ, cmap=cmap, norm=Normalize(np.nanmin(displ), np.nanmax(displ)),
        marker='o', label='data')
    if classic_residuals:
        ax.plot([np.min(fit['energy']), np.max(fit['energy'])],
                [np.min(fit['fitted energy']), np.max(fit['fitted energy'])],
                '-', label=None, color='black')
    if cmap is not None:
        plt.colorbar(
            sc, ax=ax, label='Sublattice deviation per displaced atom, Å')


def _atat_plot_calculated_energies(
        ax: plt.Axes,
        predstr: pd.DataFrame,
        gs: pd.DataFrame,
        fit: pd.DataFrame,
        extra: list[pd.DataFrame] | None = None,
        c_range: tuple[float, float] = (0.0, 1.0),
        erange: tuple[float, float] | None = None,
        ) -> None:
    """Plot Fitted energies at AX axis.
    """
    erred = predstr[
        predstr['status'].str.contains('e') &
        ~predstr['status'].str.contains('b')]

    ax.set_title('Calculated Energies')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    ax.set_xlim(c_range[0], c_range[1])
    if erange:
        ax.set_ylim(erange[0], erange[1])

    displ = np.array(fit['sublattice deviation'], dtype=float)
    cmap = __blue_orrd_cmap(
        np.nanmin(displ), np.nanmax(displ),
        color=ax._get_lines.get_next_color())
    sc = ax.scatter(
        fit['concentration'], fit['energy'],
        c=displ, cmap=cmap, norm=Normalize(np.nanmin(displ), np.nanmax(displ)),
        marker='P', label='known str')
    if cmap is not None:
        plt.colorbar(
            sc, ax=ax, label='Sublattice deviation per displaced atom, Å')
    if extra is not None:
        for df in extra:
            concentrations = []
            energies = []
            for index, concentration, energy in zip(
                    df['index'], df['concentration'], df['energy']):
                orig_energy = fit.loc[fit['index'] == index, 'energy'].iloc[0]
                if np.isclose(energy, orig_energy, atol=df.threshold):
                    continue
                concentrations.append(concentration)
                energies.append(energy)
            if len(concentrations) > 0:
                ax.plot(concentrations, energies,
                        's', fillstyle='none', label=df.name)
    ax.plot(
        gs['concentration'], gs['energy'],
        'o-', fillstyle='none', color='black',
        label='calculated gs')
    error_groups = {}
    for idx in erred['index']:
        error_dir = Path(f"{idx}")
        for err_file in error_dir.glob("error_*"):
            error_groups.setdefault(err_file.name, []).append(
                (erred.loc[erred['index'] == idx, 'concentration'].iloc[0],
                 erred.loc[erred['index'] == idx, 'energy'].iloc[0])
            )
    colors = ["red", "brown", "black", "magenta"]
    for idx, (err_name, points) in enumerate(error_groups.items()):
        x_vals, y_vals = zip(*points)
        ax.plot(
            x_vals, y_vals, 'x',
            color=colors[idx % len(colors)], label=err_name)
    ax.legend()


def _atat_plot_sublattice_deviation(
        ax: plt.Axes,
        gs: pd.DataFrame,
        fit: pd.DataFrame) -> None:
    """Plot Fitted energies at AX axis.
    """
    ax.set_title('Deviation from ideal sublattice')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Displacement per moved atom, Å')
    ax.plot(fit['concentration'], fit['sublattice deviation'], 'o', label='all')
    ax.plot(gs['concentration'], gs['sublattice deviation'], 'o', label='gs')
    ax.legend()


def _atat_1(
        wdir: str,
        extra_dirs: list[str] | None = None,
        extra_dirs_threshold: float = 0.001,
        cmin: float | None = None,
        cmax: float | None = None,
        erange: tuple[float, float] | None = None,
        classic_residuals: bool = False) -> None:
    """Plot ATAT output in WDIR.
    If EXTRA_DIRS is provided, also plot additional ATAT caluclation points
    from those dirs (mirrowing ATAT dir structure).  The points are only
    plotted when their energies differ by more than EXTRA_DIRS_THRESHOLD
    from the main ATAT calculation.
    When CLASSIC_RESIDUALS is True, plot classic energy/energy correlation
    as residuals of the fit.
    """
    # Ignore directories that do not contain lat.in
    if not (Path(wdir) / 'lat.in').is_file():
        return

    clusters = _atat_read_clusters(Path(wdir))
    predstr = _atat_read_predstr(Path(wdir))
    gs = _atat_read_gs(Path(wdir))
    fit = _atat_read_fit(Path(wdir))
    not_converged = False
    try:
        with open(Path(wdir) / 'maps.log', 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.strip().split('\n')
            cve = lines[-1] if lines else ''
            cve = cve.replace('Crossvalidation score', 'CV')
            # Check for fit convergence warning
            if "Among structures of known energy, true ground states differ from fitted ground states" in content:
                not_converged = True
            # Check concentration range
            conc_range = (0.0, 1.0)
            match = re.search(
                    r'Concentration range used for ground state checking:'
                    r'\s*\[(.*?),(.*?)\]',
                    content)
            if match:
                conc_range = (float(match.group(1)), float(match.group(2)))
    except Exception:
        cve = 'N/A'
        not_converged = True
        conc_range = (0.0, 1.0)

    if (cmin is not None) and (cmax is not None):
        conc_range = (cmin, cmax)
    elif cmin is not None:
        conc_range = (cmin, conc_range[1])
    elif cmax is not None:
        conc_range = (conc_range[0], cmax)
    else:
        pass

    logger.debug('Read concentration range: %s', conc_range)

    with open(Path(wdir) / "ref_energy.out", 'r', encoding='utf-8') as ref_energy:
        e0_atat = float(next(ref_energy))
        e1_atat = float(next(ref_energy))
    # Sometimes, ATAT uses smaller reference cell
    with open(Path(wdir) / "0/energy", 'r', encoding='utf-8') as e0_file:
        e0_energy = float(next(e0_file))
        ref_structure_mult = round(e0_energy / e0_atat)

    # Determine reference energies for scaling
    if conc_range == (0.0, 1.0):
        e0_index = 0
        e0 = e0_atat
        e1_index = 1
        e1 = e1_atat
    else:
        # Find nearest ground state structures to boundaries
        a, b = conc_range
        gs_near_a = gs.iloc[(gs['concentration'] - a).abs().argsort()[:1]]
        gs_near_b = gs.iloc[(gs['concentration'] - b).abs().argsort()[:1]]
        e0_index = int(gs_near_a.iloc[0]['index'])
        e0 = fit.loc[fit['index'] == e0_index, 'energy'].values[0]
        e1_index = int(gs_near_b.iloc[0]['index'])
        e1 = fit.loc[fit['index'] == e1_index, 'energy'].values[0]

    extra = []
    if extra_dirs is not None:
        ref_structure_len = len(Structure.from_file(Path('1/str.out')))
        ref_structure_len /= ref_structure_mult
        extra = [_atat_read_extra(fit, extra_dir, e0_atat, e1_atat, ref_structure_len)
                 for extra_dir in extra_dirs]

    # Re-scale energies
    if not conc_range == (0.0, 1.0):
        logger.debug(
            'Re-scaling energies for concentration range %s',
            conc_range)
        for data in [predstr, fit, gs] + extra:
            for field in ['energy', 'predicted energy', 'fitted energy']:
                if field not in data:
                    continue
                try:
                    if hasattr(data, 'name'):
                        e0_here = data.loc[data['index'] == e0_index, 'energy'].values[0]
                        e1_here = data.loc[data['index'] == e1_index, 'energy'].values[0]
                    else:
                        e0_here = e0
                        e1_here = e1
                except Exception:
                    warnings.warn(
                        f"Missing reference energy in extra folder {data.name}"
                    )
                    e0_here = e0
                    e1_here = e1
                data[field] = data.apply(
                    lambda row: row[field] - (
                        e0_here + (row['concentration'] - conc_range[0]) *
                        (e1_here - e0_here) / (conc_range[1] - conc_range[0])
                    ) if not np.isnan(row[field]) else row[field],
                    axis=1
                )

    for df in extra:
        df.threshold = extra_dirs_threshold
        filename = str(df.name).replace('/', '') + ".out"
        filename = Path(wdir)/filename
        logger.info("Saving extra energy points to %s", filename)
        df.sort_values('concentration').to_csv(
            filename, sep=' ', index=False)
        print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
              + colored("ATAT ", "magenta")
              + f"Saved extra energy points to {filename}")

    for idx in alive_it(fit['index'],
                        title="Getting sublattice deviations"):
        vaspdir = IMDGVaspDir(f"{idx}/ATAT")
        sublattice = Structure.from_file(f"{idx}/str.out")
        if not vaspdir.converged:
            displ = np.nan
        elif not IMDatat.check_volume_distortion(
                vaspdir.initial_structure, vaspdir.structure):
            displ = np.nan
            print(colored(f"{vaspdir.path}: large volume distortion (this must not happen)", "red"))
        elif (not Path(f"{idx}/str.out.old").is_file()
              and not IMDatat.check_sublattice_flip(
                  vaspdir.initial_structure, vaspdir.structure, sublattice)):
            displ = np.nan
            print(colored(f"{vaspdir.path}: sublattice flip (this must not happen)", "red"))
        elif Path(f"{idx}/sublattice_deviation").is_file():
            logger.debug(
                "Reading sublattice deviation from %s",
                f"{idx}/sublattice_deviation")
            with open(f"{idx}/sublattice_deviation", 'r', encoding='utf-8') as f:
                displ = float(f.read().strip())
        else:
            logger.debug(
                "Calculating sublattice deviation from %s",
                f"{idx}/ATAT")
            str_after_normalized = vaspdir.structure.copy()
            str_after_normalized.lattice = sublattice.lattice
            displ = structure_distance(
                str_after_normalized, sublattice,
                # Compare specie-insensitively
                match_first=True,
                match_species=False)
        fit.loc[fit['index'] == idx, 'sublattice deviation'] = displ
        gs.loc[gs['index'] == idx, 'sublattice deviation'] = displ

    logger.info("Saving sublattice deviation to %s", 'fit2.out')
    filename = Path(wdir)/"fit2.out"
    fit.sort_values('concentration').to_csv(
        filename, sep=' ', index=False)
    print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
          + colored("ATAT ", "magenta")
          + f"Saved sublattice deviations to {filename}")

    global_title = str(Path(wdir).absolute())

    # Adjust font sizes based on the provided font_size argument
    DEFAULT_FONT_SIZE = 8
    base_sz = DEFAULT_FONT_SIZE
    base_markersize = base_sz * 0.5
    plt.rcParams.update({
        'font.size': base_sz,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': base_sz,
        'axes.titlesize': base_sz * 1.2,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.2,
        'lines.markersize': base_markersize,
        'xtick.labelsize': base_sz,
        'ytick.labelsize': base_sz,
        'legend.fontsize': base_sz,
        'figure.titlesize': base_sz * 1.2,
    })

    width = 8.27*0.9  # 90% A4
    fig, axs = plt.subplots(3, 2, figsize=(width, 1.4*width))
    fig.suptitle(global_title)

    _atat_plot_fitted_energies(axs[1, 0], predstr, gs, fit, conc_range, erange)
    _atat_plot_calculated_energies(axs[0, 0], predstr, gs, fit, extra, conc_range, erange)
    _atat_plot_calc_vs_fit_energies(axs[2, 0], fit, conc_range, erange)
    _atat_plot_residuals(axs[0, 1], cve, fit, classic_residuals)
    _atat_plot_sublattice_deviation(axs[1, 1], gs, fit)
    _atat_plot_clusters(axs[2, 1], clusters)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])  # Leave space for both suptitle and warning

    # Add warning text if fit not converged
    if not_converged:
        plt.figtext(
            0.5, 0.92,  # Adjusted to avoid overlapping with suptitle
            "WARNING: True and fitted ground states differ"
            " - fit not converged!",
            ha="center",
            bbox={"facecolor": "red", "alpha": 0.3, "pad": 5})

    output_png = Path(wdir)/'atat-summary-test.png'
    plt.savefig(output_png, dpi=300)
    output_svg = Path(wdir)/'atat-summary-test.svg'
    plt.savefig(output_svg, dpi=300)

    logger.info("Saving ATAT plots to %s", output_png)
    print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
          + colored("ATAT ", "magenta")
          + f"Saved ATAT plots to {output_png} and {output_svg}")


def atat(args):
    """Create ATAT visualization.
    """
    _atat_1(
        args.dir, args.plot_extra, args.plot_extra_threshold,
        args.cmin, args.cmax,
        (args.emin, args.emax) if (args.emin and args.emax) else None,
        args.classic_residuals)
    return 0


def hull_add_args(parser):
    """Setup parser arguments for formation energy hull visualization.
    Args:
      parser: subparser
    """
    parser.help = "Plot formation energy hull from ATAT results"

    parser.add_argument(
        "--extra_dir",
        help="Extra directories with VASP outputs to include in the hull. "
        "These are read directly without recursive scanning or include/exclude filtering.",
        type=str,
        nargs="*",
        default=None)

    parser.add_argument(
        "--ion", default="Li",
        help="Working ion element (default: Li)",
        type=Element)
    parser.add_argument(
        "--max_composition",
        help="Maximum composition for the concentration axis (e.g. LiC2). "
        "If not specified, uses pure ion element as maximum.",
        type=Composition)

    parser.add_argument(
        "--dpi", default=600,
        help="Output DPI for publication quality (default: 300)",
        type=int)
    parser.add_argument(
        "--format", default="png",
        help="Output format (default: png, options: png, pdf, svg)",
        choices=["png", "pdf", "svg"])
    parser.add_argument(
        "--show_unstable", default=0.2,
        help="Show unstable entries with energy above hull less than this value (eV/atom) (default: 0.2)",
        type=float)
    parser.add_argument(
        "--font_size", default=10,
        help="Base font size for the plot (default: 10)",
        type=int)
    parser.add_argument(
        "--title", default=None,
        help="Custom title for the phase diagram plot (default: '<ion>-<matrix> Phase Diagram')",
        type=str)
    parser.add_argument(
        "--ymin",
        help="Manual y-axis minimum (in meV/atom).",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--ymax",
        help="Manual y-axis maximum (in meV/atom).",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--entropy",
        help="Adjust energies by the provided emc2 output file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--entropy_vibrational",
        help="Adjust energies by vibrational entropy (need to set --entropy to use this)",
        action="store_true"
    )
    parser.add_argument(
        "--include",
        help="Include only directories whose relative path matches this regexp. "
        "Applied after --exclude.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--exclude",
        help="Exclude directories whose relative path matches this regexp. "
        "Default: gorun_\\d+ (gorun_<number> pattern).",
        type=str,
        default=r'gorun_\d+'
    )
    parser.add_argument(
        "--pickle_structure_column",
        default="structure",
        help="Column name for ASE atoms in pickle file (default: structure)",
        type=str)
    parser.add_argument(
        "--pickle_energy_column",
        default="total_energy",
        help="Column name for total energy in pickle file (default: total_energy)",
        type=str)
    parser.add_argument(
        "--pickle_id_column",
        default="fold_id",
        help="Column name for structure ID in pickle file (default: fold_id). "
        "If the column is missing, the DataFrame index is used.",
        type=str)
    parser.add_argument(
        "--filter",
        default=None,
        help="Python expression to filter rows from the pickle DataFrame. "
        "The DataFrame is bound to the name 'df' during evaluation. "
        "The 'pd' module (pandas) is also available in the eval context. "
        "Example: --filter \"df[df['total_energy'] < -10]\"",
        type=str)
    parser.set_defaults(func_derive=hull)


def _to_subscript(text: str) -> str:
    """Convert integer characters in a composition string to LaTeX subscript form.

    Example:
        'LiC6' -> 'LiC$_{6}$'
    """
    return re.sub(r'(\d+)', r'$_{\1}$', text)


def _hull_get_entries_recursively(
        path: Path,
        include: str | None = None,
        exclude: str | None = r'gorun_\d+') -> list:
    """Scan PATH for VASP directories and return a list of ComputedStructureEntry.

    Uses IMDGVaspDir.read_vaspdirs to find VASP directories recursively.
    Directories matching EXCLUDE regexp (default: gorun_<number>) are skipped.
    When INCLUDE is set, only directories whose relative path matches the
    regexp are included (applied after exclude).

    Args:
        path: Root path to scan.
        include: Optional regexp; only dirs whose relative path matches are kept.
        exclude: Regexp; dirs whose relative path matches are skipped.
                 Default: r'gorun_\\d+'.
    """
    root = Path(path)
    exclude_re = re.compile(exclude) if exclude else None
    include_re = re.compile(include) if include else None

    def path_filter(parent_path: Path) -> bool:
        """Filter VASP directories by include/exclude regexps."""
        rel_str = os.path.relpath(str(parent_path), start=str(root))

        if exclude_re and exclude_re.search(rel_str):
            return False
        if include_re and not include_re.search(rel_str):
            return False
        return True

    vaspdirs = IMDGVaspDir.read_vaspdirs(root, path_filter=path_filter)

    entries = []
    n_skipped = 0
    for vasp_path, vaspdir in alive_it(
            vaspdirs.items(),
            total=len(vaspdirs),
            title='Reading VASP outputs'):
        try:
            if vaspdir.final_energy is None:
                n_skipped += 1
                continue
            if not vaspdir.converged:
                print(f"Skipping {vasp_path}: unconverged")
                n_skipped += 1
                continue
            entry = ComputedStructureEntry(
                vaspdir.structure, vaspdir.final_energy)
            entry.data["volume"] = vaspdir.structure.volume
            entry.data["ID"] = vasp_path
            entries.append(entry)
        except Exception as e:
            print(f"Skipping {vasp_path}: {str(e)}")
            n_skipped += 1

    print(f"Read {len(vaspdirs)} runs")
    print(f"Skipped: {n_skipped}")
    return entries


def _hull_plot_custom_phase_diagram(
        phd: PhaseDiagram,
        ax: plt.Axes,
        ion_element: str,
        matrix_element: str,
        max_conc: float = 1.0,
        show_unstable: float = 1000,
        font_size: int = 10,
        title: str | None = None,
        ymin: float | None = None,
        ymax: float | None = None) -> None:
    """Plot a custom phase diagram on AX, overriding pymatgen's hardcoded font settings."""
    energy_mult = 1000

    plotter = PDPlotter(phd, show_unstable=show_unstable)
    lines, stable_entries, unstable_entries = plotter.pd_plot_data

    # Save all entries to formation_en.txt
    data_file = 'formation_en.txt'
    gs_data_file = 'formation_en_gs.txt'
    min_data_file = 'formation_en_min.txt'
    data = [{
        'ID': str(entry.data.get("ID")),
        'Energy': entry.energy_per_atom,
        'Concentration': coords[0],
        'Formation Energy (meV/atom)': coords[1] * energy_mult,
        "Energy above hull (meV/atom)": (
            phd.get_e_above_hull(entry) * energy_mult
            if phd.get_e_above_hull(entry) is not None else None),
        'Formula': (
            "C" if np.isclose(coords[0], 0)
            else f"{ion_element}C{int((1 - coords[0]) / coords[0])}")
    } for entry, coords in unstable_entries.items()
      if phd.get_e_above_hull(entry) is not None
      and phd.get_e_above_hull(entry) < show_unstable]
    stable_data = [{
        'ID': str(entry.data.get("ID")),
        'Energy': entry.energy_per_atom,
        'Concentration': coords[0],
        'Formation Energy (meV/atom)': coords[1] * energy_mult,
        "Energy above hull (meV/atom)": (
            phd.get_e_above_hull(entry) * energy_mult
            if phd.get_e_above_hull(entry) is not None else None),
        'Formula': (
            "C" if np.isclose(coords[0], 0)
            else f"{ion_element}C{int((1 - coords[0]) / coords[0])}")
    } for coords, entry in stable_entries.items()]
    data.extend(stable_data)

    if data:
        df = pd.DataFrame(data)
        df.to_csv(data_file, index=False, sep=' ')
    print(f"All energies saved to {data_file}")
    if stable_data:
        df = pd.DataFrame(stable_data)
        df.to_csv(gs_data_file, index=False, sep=' ')
    print(f"GS energies saved to {gs_data_file}")
    if phd.all_entries:
        min_entries = []
        for _, group_iter in itertools.groupby(
                phd.all_entries,
                key=lambda e: e.composition.reduced_composition):
            group = list(group_iter)
            entry = min(group, key=lambda e: e.energy_per_atom)
            min_entries.append({
                'ID': str(entry.data.get("ID")),
                "Energy": entry.energy_per_atom,
                "Formation energy (meV/atom)": (
                    phd.get_form_energy_per_atom(entry) * energy_mult),
                "Energy above hull (meV/atom)": (
                    phd.get_e_above_hull(entry) * energy_mult),
                "Reduced formula": entry.reduced_formula,
            })
        df = pd.DataFrame(min_entries)
        df.to_csv(min_data_file, index=False, sep=' ')
        print(f'Min energies saved to {min_data_file}')

    plt.style.use('default')
    base_sz = font_size
    base_markersize = base_sz * 0.5
    edge_width = max(0.8, round(0.12 * base_sz, 2))
    plt.rcParams.update({
        'font.size': base_sz,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': base_sz,
        'axes.titlesize': base_sz * 1.2,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.2,
        'lines.markersize': base_markersize,
        'xtick.labelsize': base_sz,
        'ytick.labelsize': base_sz,
        'legend.fontsize': base_sz,
        'figure.titlesize': base_sz * 1.2,
    })

    marker_color = '#4daffa'
    for entry, coords in unstable_entries.items():
        e_above_hull = phd.get_e_above_hull(entry)
        if e_above_hull is not None and e_above_hull < show_unstable:
            ax.plot(coords[0], np.array(coords[1]) * energy_mult, 's',
                    markerfacecolor=marker_color,
                    markeredgecolor='black',
                    markeredgewidth=edge_width,
                    alpha=0.7)

    for x, y in lines:
        ax.plot(x, np.array(y) * energy_mult, 'k-', linewidth=1.2)

    gs_color = '#00af00'
    for coords in stable_entries:
        entry = stable_entries[coords]
        print(f"GS:: {entry.data}, {coords[0]}, {coords[1]}")
        ax.plot(coords[0], np.array(coords[1]) * energy_mult, 'o',
                markerfacecolor=gs_color,
                markeredgecolor='black',
                markersize=base_markersize * 1.8,
                markeredgewidth=edge_width)

    ground_state_marker = Line2D(
        [], [], marker='o', color='none',
        markerfacecolor=gs_color,
        markeredgecolor='black',
        markersize=base_markersize * 1.8)
    above_hull_marker = Line2D(
        [], [], marker='s', color='none',
        markerfacecolor=marker_color,
        markeredgecolor='black')
    ax.legend(
        handles=[ground_state_marker, above_hull_marker],
        labels=['Ground state', 'Above hull'],
        loc='best', fontsize=font_size)

    min_y = min(c[1] for c in stable_entries)
    center = (0.5, min_y / 2)

    font = FontProperties()
    font.set_size(base_sz)
    font.set_weight('bold')

    for coords in sorted(stable_entries, key=lambda x: -x[1]):
        entry = stable_entries[coords]
        if entry.composition.is_element:
            continue
        raw_label = entry.name
        label = _to_subscript(raw_label)
        offset_radius_pt = base_sz * 1.5
        vec = np.array(coords) - center
        norm_vec = np.linalg.norm(vec)
        if norm_vec != 0:
            vec = vec / norm_vec * offset_radius_pt
        else:
            vec = np.zeros_like(vec)
        valign = "bottom" if vec[1] > 0 else "top"
        if vec[0] < -0.01:
            halign = "right"
        elif vec[0] > 0.01:
            halign = "left"
        else:
            halign = "center"
        ax.annotate(
            label,
            [coords[0], coords[1] * energy_mult],
            xytext=vec,
            textcoords="offset points",
            horizontalalignment=halign,
            verticalalignment=valign,
            fontproperties=font,
            color='black'
        )

    elem_font_size = base_sz * 1.2
    elem_font = FontProperties(size=base_sz + 2, weight='bold')
    for coords in stable_entries:
        entry = stable_entries[coords]
        if entry.composition.is_element:
            elem_offset_pt = elem_font_size
            if coords[0] < 0.1:
                ax.annotate(matrix_element,
                            [coords[0], coords[1] * energy_mult],
                            xytext=(-elem_offset_pt, 0),
                            textcoords="offset points",
                            horizontalalignment="right",
                            verticalalignment="center",
                            fontproperties=elem_font)
            elif coords[0] > 0.9:
                ax.annotate(ion_element,
                            [coords[0], coords[1] * energy_mult],
                            xytext=(elem_offset_pt, 0),
                            textcoords="offset points",
                            horizontalalignment="left",
                            verticalalignment="center",
                            fontproperties=elem_font)

    ax.set_xlabel(f'{ion_element} Concentration')
    ax.set_ylabel('Formation Energy (meV/atom)')
    ax.set_title(
        title if title is not None
        else f'{ion_element}-{matrix_element} Phase Diagram',
        pad=20)
    ax.set_xlim(-0.05, max_conc + 0.05)

    all_y = [c[1] * energy_mult for c in stable_entries] + \
        [c[1] * energy_mult for _, c in unstable_entries.items()]
    y_min = min(all_y)
    y_max = max(all_y)
    if ymin is not None and ymin < y_min:
        y_min = ymin
    if ymax is not None:
        y_max = ymax
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    ionc6_concentration = 1 / 7
    plt.axvline(x=ionc6_concentration, color='black',
                linewidth=1, linestyle='--', alpha=0.7)
    plt.annotate(f'{ion_element}C$_{6}$',
                 xy=(ionc6_concentration, ax.get_ylim()[0]),
                 xytext=(base_sz * 1, base_sz * 0.5),
                 textcoords='offset points',
                 ha='left', va='bottom', font=elem_font,
                 color='black')


def _hull_get_entries_from_pickle(
        path: Path,
        structure_column: str = "structure",
        energy_column: str = "total_energy",
        id_column: str | None = "fold_id",
        filter_expr: str | None = None) -> list:
    """Read structures and energies from a pickle file and return ComputedStructureEntry list.

    The pickle file must contain a DataFrame with ASE Atoms objects in
    STRUCTURE_COLUMN, total energies in ENERGY_COLUMN, and optionally an ID
    column in ID_COLUMN.

    Args:
        path: Path to the pickle file containing the DataFrame.
        structure_column: Name of the column holding ASE Atoms objects.
        energy_column: Name of the column holding total energies.
        id_column: Optional column name for structure IDs.
                  If None or missing from the DataFrame, the index is used.
        filter_expr: Optional Python expression to filter the DataFrame.
                     The loaded DataFrame is bound to the name 'df' during
                     evaluation.  The 'pd' module (pandas) is also available
                     in the eval context.  The expression must return a
                     DataFrame (or array-like indexer).
                     Example: "df[df['total_energy'] < -10]"

    Returns:
        List of ComputedStructureEntry objects.
    """
    df = pd.read_pickle(path)
    print(f"Read database: {path}")
    print(df)
    print(f"Columns: {df.columns}")
    print(f"Using {id_column} (ids), {structure_column} (structures), {energy_column} (energies)")

    n_before = len(df)
    if filter_expr:
        df = eval(filter_expr, {"df": df, "pd": pd})
        print(f"Filter applied: {n_before} -> {len(df)} structures retained")

    entries = []
    has_id_column = id_column is not None and id_column in df.columns
    for idx, row in df.iterrows():
        atoms = row[structure_column]
        total_energy = row[energy_column]

        structure = AseAtomsAdaptor.get_structure(atoms)

        entry = ComputedStructureEntry(structure, total_energy)
        entry.data["volume"] = structure.volume
        entry.data["ID"] = str(row[id_column]) if has_id_column else str(idx)
        entries.append(entry)

    print(f"Read {len(entries)} structures from pickle file {path}")
    return entries


def hull(args):
    """Plot formation energy hull from ATAT results.
    """
    path = Path(args.dir)

    fig, ax = plt.subplots(figsize=(4.13, 3))

    temperature = 0.0
    if args.entropy:
        df = pd.read_csv(
            args.entropy, header=None, sep='\t',
            names=['T', 'mu', 'E', 'x', 'F'],
            usecols=list(range(5)))
        assert np.isclose(df['T'].min(), df['T'].max())
        temperature = float(df['T'].min())
        print(f'Adding entropy adjustments for T={temperature}')

    if path.is_file():
        entries = _hull_get_entries_from_pickle(
            path,
            structure_column=args.pickle_structure_column,
            energy_column=args.pickle_energy_column,
            id_column=args.pickle_id_column,
            filter_expr=args.filter)
    else:
        entries = _hull_get_entries_recursively(
            path, include=args.include, exclude=args.exclude)

    # Read extra directories directly (no recursive scan, no include/exclude)
    if args.extra_dir:
        for extra_path in args.extra_dir:
            try:
                extra_vaspdir = IMDGVaspDir(Path(extra_path))
                if extra_vaspdir.final_energy is None:
                    print(f"Skipping {extra_path}: no final energy")
                    continue
                if not extra_vaspdir.converged:
                    print(f"Skipping {extra_path}: unconverged")
                    continue
                extra_entry = ComputedStructureEntry(
                    extra_vaspdir.structure, extra_vaspdir.final_energy)
                extra_entry.data["volume"] = extra_vaspdir.structure.volume
                extra_entry.data["ID"] = extra_path
                entries.append(extra_entry)
                print(f"Added {extra_path}: {extra_entry.energy_per_atom} eV/atom")
            except Exception as ex:
                print(f"Skipping {extra_path}: {str(ex)}")

    # Validate that pure element entries exist for both ion and matrix.
    pure_ion_entries = [
        e for e in entries
        if e.composition.is_element and e.composition.elements[0] == args.ion]
    pure_matrix_entries = [
        e for e in entries
        if e.composition.is_element and e.composition.elements[0] != args.ion]

    if not pure_matrix_entries:
        print(
            "Error: No pure element (matrix) entry found in the data. "
            "Expected a pure element calculation that does not contain "
            f"{args.ion.symbol}.")
        print(
            "Please ensure a pure element calculation (without "
            f"{args.ion.symbol}) is included in the directory scan "
            "or provided via --extra_dir.")
        return 1
    if not pure_ion_entries:
        print(
            f"Error: No pure {args.ion.symbol} (ion) entry found "
            "in the data.")
        print(
            f"Please ensure a pure {args.ion.symbol} calculation is "
            "included in the directory scan or provided via --extra_dir.")
        return 1

    # Ensure only one distinct matrix element is present.
    matrix_elements = {e.composition.elements[0] for e in pure_matrix_entries}
    if len(matrix_elements) > 1:
        print(
            "Error: Multiple distinct matrix element candidates found: "
            f"{', '.join(e.symbol for e in sorted(matrix_elements, key=lambda x: x.symbol))}.")
        print(
            "Expected only one type of pure element that is not the ion "
            f"({args.ion.symbol}).")
        print(
            "Please ensure only one matrix element type is present in the "
            "directory scan or provided via --extra_dir.")
        return 1

    matrix_element = next(iter(matrix_elements))
    matrix_entry = min(pure_matrix_entries, key=lambda e: e.energy_per_atom)
    ion_entry = min(pure_ion_entries, key=lambda e: e.energy_per_atom)
    print(f"Using {matrix_entry.composition.reduced_formula} as matrix reference "
          f"(energy={matrix_entry.energy_per_atom:.4f} eV/atom)")
    print(f"Using {ion_entry.composition.reduced_formula} as ion reference "
          f"(energy={ion_entry.energy_per_atom:.4f} eV/atom)")

    if args.entropy:
        df['c'] = (df['x'] + 1) / 2
        for entry in entries:
            atomic_fraction = entry.composition.get_atomic_fraction(args.ion)
            c = 2 * atomic_fraction / (1 - atomic_fraction)
            closest_idx = (df['c'] - c).abs().idxmin()
            if np.abs(df.loc[closest_idx]['c'] - c) > 0.02:
                print(f'Warning: assigning entropy for concentration diff: '
                      f'{np.abs(df.loc[closest_idx]["c"] - c)}')
            ts = df.loc[closest_idx]['E'] - df.loc[closest_idx]['F']
            entry.correction = -ts / (df.loc[closest_idx]['c'] + 2) * \
                entry.composition.num_atoms

    if np.isclose(temperature, 0) or not args.entropy_vibrational:
        phd = PhaseDiagram(
            entries=entries,
            elements=[matrix_element, args.ion])
    else:
        all_entries = entries

        def _reduced_mass(structure) -> float:
            """Reduced mass as calculated via Eq. 6 in Bartel et al. (2018)."""
            reduced_comp = structure.composition.reduced_composition
            n_elems = len(reduced_comp.elements)
            elem_dict = reduced_comp.get_el_amt_dict()
            denominator = (n_elems - 1) * reduced_comp.num_atoms
            all_pairs = itertools.combinations(elem_dict.items(), 2)
            mass_sum = 0
            for pair in all_pairs:
                m_i = Composition(pair[0][0]).weight
                m_j = Composition(pair[1][0]).weight
                alpha_i = pair[0][1]
                alpha_j = pair[1][1]
                mass_sum += (alpha_i + alpha_j) * (m_i * m_j) / (m_i + m_j) * 2
            return (1 / denominator) * mass_sum

        for entry in all_entries:
            if entry.composition.reduced_composition.num_atoms == 1:
                reduced_mass = entry.composition.reduced_composition.weight
            else:
                reduced_mass = _reduced_mass(entry.structure)
            sisso_corr = (
                entry.composition.num_atoms *
                GibbsComputedStructureEntry._g_delta_sisso(
                    entry.structure.volume / len(entry.structure),
                    reduced_mass,
                    temperature
                )
            )
            print(entry.composition.get_atomic_fraction(args.ion),
                  sisso_corr / entry.composition.num_atoms)
            entry.correction += sisso_corr
        phd = PhaseDiagram(
            entries=all_entries,
            elements=[matrix_element, args.ion])

    max_conc = 1.0
    if args.max_composition:
        max_conc = args.max_composition.get_atomic_fraction(Element(args.ion))

    _hull_plot_custom_phase_diagram(
        phd, ax, str(args.ion), matrix_element.symbol,
        show_unstable=args.show_unstable,
        max_conc=max_conc,
        font_size=args.font_size,
        title=args.title,
        ymax=args.ymax,
        ymin=args.ymin)

    plt.tight_layout()

    output_file = [f'formation_en.{args.format}']
    if not args.format == "svg":
        output_file.append('formation_en.svg')
    for output in output_file:
        plt.savefig(output, dpi=args.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()

    print(f"Formation energy profile saved to {output_file}")
    return 0


def voltage_add_args(parser):
    """Setup parser arguments for voltage profile visualization.
    Args:
      parser: subparser
    """
    parser.help = "Plot voltage profile from ATAT results"

    parser.add_argument(
        "--ion", default="Li",
        help="Working ion element (default: Li)",
        type=Element)
    parser.add_argument(
        "--extra_dir",
        help="Extra directories with VASP outputs to include in the analysis. "
        "These are read directly without recursive scanning or include/exclude filtering.",
        type=str,
        nargs="*",
        default=None)
    parser.add_argument(
        "--include",
        help="Include only directories whose relative path matches this regexp. "
        "Applied after --exclude.",
        type=str,
        default=None)
    parser.add_argument(
        "--exclude",
        help="Exclude directories whose relative path matches this regexp. "
        "Default: gorun_\\d+ (gorun_<number> pattern).",
        type=str,
        default=r'gorun_\\d+')
    parser.add_argument(
        "--dpi", default=600,
        help="Output DPI (default: 600)",
        type=int)
    parser.add_argument(
        "--format", default="png",
        help="Output format (default: png, options: png, pdf, svg)",
        choices=["png", "pdf", "svg"])
    parser.add_argument(
        "--font_size", default=10,
        help="Base font size for the plot (default: 10)",
        type=int)
    parser.add_argument(
        "--pickle_structure_column",
        default="structure",
        help="Column name for ASE atoms in pickle file (default: structure)",
        type=str)
    parser.add_argument(
        "--pickle_energy_column",
        default="total_energy",
        help="Column name for total energy in pickle file (default: total_energy)",
        type=str)
    parser.add_argument(
        "--pickle_id_column",
        default="fold_id",
        help="Column name for structure ID in pickle file (default: fold_id). "
        "If the column is missing, the DataFrame index is used.",
        type=str)
    parser.add_argument(
        "--filter",
        default=None,
        help="Python expression to filter rows from the pickle DataFrame. "
        "The DataFrame is bound to the name 'df' during evaluation. "
        "The 'pd' module (pandas) is also available in the eval context. "
        "Example: --filter \"df[df['total_energy'] < -10]\"",
        type=str)
    parser.add_argument(
        "--xaxis", default="frac_x",
        help="X-axis quantity for the voltage profile plot. "
        "Options: capacity_grav (gravimetric capacity, mAh/g), "
        "capacity_vol (volumetric capacity, Ah/l), "
        "x_form (working ions per formula unit), "
        "frac_x (atomic fraction of working ion). "
        "(default: frac_x)",
        choices=["capacity_grav", "capacity_vol", "x_form", "frac_x"],
        type=str)
    parser.set_defaults(func_derive=voltage)


def voltage(args):
    """Plot voltage profile from ATAT results using pymatgen's battery analysis tools.
    """
    path = Path(args.dir)

    # Read entries: from pickle if path is a file, otherwise scan recursively
    if path.is_file():
        entries = _hull_get_entries_from_pickle(
            path,
            structure_column=args.pickle_structure_column,
            energy_column=args.pickle_energy_column,
            id_column=args.pickle_id_column,
            filter_expr=args.filter)
    else:
        entries = _hull_get_entries_recursively(
            path, include=args.include, exclude=args.exclude)

    # Add extra directories directly (no recursive scan, no include/exclude)
    if args.extra_dir:
        for extra_path in args.extra_dir:
            try:
                extra_vaspdir = IMDGVaspDir(Path(extra_path))
                if extra_vaspdir.final_energy is None:
                    print(f"Skipping {extra_path}: no final energy")
                    continue
                if not extra_vaspdir.converged:
                    print(f"Skipping {extra_path}: unconverged")
                    continue
                extra_entry = ComputedStructureEntry(
                    extra_vaspdir.structure, extra_vaspdir.final_energy)
                extra_entry.data["volume"] = extra_vaspdir.structure.volume
                extra_entry.data["ID"] = extra_path
                entries.append(extra_entry)
                print(f"Added {extra_path}: {extra_entry.energy_per_atom} eV/atom")
            except Exception as ex:
                print(f"Skipping {extra_path}: {str(ex)}")

    # Find pure working ion entry from collected data
    pure_ion_entries = [
        e for e in entries
        if e.composition.is_element and e.composition.elements[0] == args.ion]
    if not pure_ion_entries:
        raise ValueError(
            f"No pure {args.ion.symbol} entry found in the data. "
            f"Please ensure a pure {args.ion.symbol} calculation is included "
            "in the directory scan or provided via --extra_dir.")
    working_ion_entry = min(pure_ion_entries, key=lambda e: e.energy_per_atom)
    print(f"Using {working_ion_entry.composition.reduced_formula} as working ion reference "
          f"(energy={working_ion_entry.energy_per_atom:.4f} eV/atom)")

    # Create insertion electrode
    electrode = InsertionElectrode.from_entries(
        entries,
        working_ion_entry=working_ion_entry,
        strip_structures=False
    )

    # Extract voltage profile data
    plotter = VoltageProfilePlotter(xaxis=args.xaxis)
    x, voltage = plotter.get_plot_data(electrode, term_zero=False)
    capacity = []
    cap_acc = 0
    sub_electrodes = electrode.get_sub_electrodes(adjacent_only=True)
    normalization_mass = sub_electrodes[0].voltage_pairs[0].mass_charge
    for sub_electrode in sub_electrodes:
        capacity.append(cap_acc)
        cap_acc += sum(pair.mAh for pair in sub_electrode.voltage_pairs) / normalization_mass
        capacity.append(cap_acc)
    voltage_data = {'x': x, 'voltage': voltage, 'capacity': capacity}
    df = pd.DataFrame(voltage_data).sort_values("x")

    # Save data
    output_data = Path(args.dir) / 'voltage.out'
    df.to_csv(output_data, sep=' ', index=False)
    print(f"Voltage data saved to {output_data}")

    # Plot
    base_sz = args.font_size
    plt.rcParams.update({
        'font.size': base_sz,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': base_sz,
        'axes.titlesize': base_sz * 1.2,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.2,
        'xtick.labelsize': base_sz,
        'ytick.labelsize': base_sz,
        'legend.fontsize': base_sz,
        'figure.titlesize': base_sz * 1.2,
    })

    # Choose x-axis data and label based on selected xaxis
    if args.xaxis == "capacity_grav" or args.xaxis == "capacity":
        plot_x = df['capacity']
        xlabel = 'Capacity (mAh/g)'
    elif args.xaxis == "capacity_vol":
        plot_x = df['capacity']
        xlabel = 'Capacity (Ah/l)'
    elif args.xaxis == "x_form":
        plot_x = df['x']
        xlabel = f'x in {args.ion.symbol}<sub>x</sub>Host'
    elif args.xaxis == "frac_x":
        plot_x = df['x']
        xlabel = f'Atomic Fraction of {args.ion.symbol}'
    else:
        raise ValueError(f"Unknown xaxis: {args.xaxis}")

    plt.figure(figsize=(10, 6))
    plt.step(plot_x, df['voltage'], where='post', color='blue', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(f'Voltage vs. {args.ion.symbol}/{args.ion.symbol}+ (V)')
    plt.title('Voltage Profile')
    plt.grid(alpha=0.3)

    output_png = Path(args.dir) / f'voltage.{args.format}'
    output_svg = Path(args.dir) / 'voltage.svg'
    plt.tight_layout()
    plt.savefig(output_png, dpi=args.dpi)
    if not args.format == "svg":
        plt.savefig(output_svg, dpi=args.dpi)
    plt.close()

    print(colored(f"{str(Path(args.dir)).replace('./', '')}: ", attrs=['bold'])
          + colored("Voltage ", "magenta")
          + f"profile saved to {output_png} and {output_svg}")
    return 0


def visualize(args):
    """Main routine.
    """
    return args.func_derive(args)
