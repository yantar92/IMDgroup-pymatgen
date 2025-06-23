"""Visualization extension specific to IMD group.
"""
import logging
import os
from termcolor import colored
from pathlib import Path
from pymatgen.core import Structure
from alive_progress import alive_it
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgba
import numpy as np
from IMDgroup.pymatgen.core.structure import merge_structures, structure_distance
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
from IMDgroup.pymatgen.io.vasp.sets import write_selective_dynamics_summary_maybe

logger = logging.getLogger(__name__)


def add_args(parser):
    """Setup parser arguments.
    Args:
      parser: Sub-parser.
    """
    parser.help = """Visualize vasp outputs."""

    parser.add_argument(
        "dir",
        help="""Directory to read (recusrively); defaults to current dir""",
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
    parser.set_defaults(func_derive=atat)


def _atat_read_predstr(path: Path) -> pd.DataFrame:
    """Read predstr.out.
    """
    return pd.read_csv(
        path / 'predstr.out', sep=' ', header=None,
        names=['concentration', 'energy', 'predicted_energy',
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
        names=['concentration', 'energy', 'fitted_energy', 'index']
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
        fit: pd.DataFrame) -> None:
    """Plot Fitted energies at AX axis.
    """
    newgs = predstr[predstr['status'].str.contains('g')]

    ax.set_title('Fitted Energies')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    ax.set_xlim(0, 1)
    ax.plot(
        predstr['concentration'], predstr['predicted_energy'],
        'o', label='predicted', markersize=1)
    ax.plot(
        fit['concentration'], fit['fitted energy'],
        'o', label='known str')
    ax.plot(
        gs['concentration'], gs['fitted_energy'],
        'o-', fillstyle='none',
        label='fitted gs', color='black', markersize=8)
    ax.plot(
        newgs['concentration'], newgs['predicted_energy'],
        's', markersize=8, fillstyle='none', label='predicted gs', color='red')
    ax.legend()


def _atat_plot_calc_vs_fit_energies(
        ax: plt.Axes,
        fit: pd.DataFrame) -> None:
    """Plot Fitted vs Calculated energies at AX axis.
    """
    ax.set_title('Calculated and Fitted Energies')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    ax.set_xlim(0, 1)
    ax.plot(fit['concentration'], fit['energy'], 'P', label='calculated')
    ax.plot(fit['concentration'], fit['fitted energy'], 'o', label='fitted')
    ax.legend()


def __blue_orrd_cmap(data_min, data_max, color='blue', split_val=0.1):
    """Return colormap with COLOR for 0..SPLIT_VAL and gradient for the rest.
    Return None when data_min == data_max.
    """
    if data_max == data_min:
        return None
    N = 256  # resolution
    # Set the relative position of split_val
    frac = (split_val - data_min) / (data_max - data_min)
    n_fixed_color = int(frac * N)
    n_orrd = N - n_fixed_color
    greens = np.tile([to_rgba(color)], (n_fixed_color, 1))
    orrd = plt.get_cmap("OrRd", n_orrd)
    orrd_colors = orrd(np.linspace(0, 1, n_orrd))
    return ListedColormap(np.vstack([greens, orrd_colors]))


def _atat_plot_residuals(
        ax: plt.Axes,
        cve: str,
        fit: pd.DataFrame) -> None:
    """Plot fit error at AX axis.
    """
    ax.set_title(f'Residuals of the fit ({cve})')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    displ = np.array(fit['sublattice deviation'], dtype=float)
    cmap = __blue_orrd_cmap(
        displ.min(), displ.max(),
        color=ax._get_lines.get_next_color())
    sc = ax.scatter(
        fit['concentration'], fit['energy delta'],
        c=displ, cmap=cmap, norm=Normalize(displ.min(), displ.max()),
        marker='o', label='data')
    if cmap is not None:
        plt.colorbar(
            sc, ax=ax, label='Sublattice deviation per displaced atom, Å')


def _atat_plot_calculated_energies(
        ax: plt.Axes,
        predstr: pd.DataFrame,
        gs: pd.DataFrame,
        fit: pd.DataFrame) -> None:
    """Plot Fitted energies at AX axis.
    """
    erred = predstr[
        predstr['status'].str.contains('e') &
        ~predstr['status'].str.contains('b')]

    ax.set_title('Calculated Energies')
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Energy per reference cell, eV')
    ax.set_xlim(0, 1)

    displ = np.array(fit['sublattice deviation'], dtype=float)
    cmap = __blue_orrd_cmap(
        displ.min(), displ.max(),
        color=ax._get_lines.get_next_color())
    sc = ax.scatter(
        fit['concentration'], fit['energy'],
        c=displ, cmap=cmap, norm=Normalize(displ.min(), displ.max()),
        marker='P', label='known str')
    if cmap is not None:
        plt.colorbar(
            sc, ax=ax, label='Sublattice deviation per displaced atom, Å')
    ax.plot(
        gs['concentration'], gs['energy'],
        'o-', fillstyle='none', color='black',
        markersize=8, label='calculated gs')
    ax.plot(
        erred['concentration'], erred['energy'],
        'x', markersize=5, label='error', color='red')
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


def _atat_1(wdir: str) -> None:
    """Plot ATAT output in WDIR.
    """
    # Ignore directories that do not contain lat.in
    if not (Path(wdir) / 'lat.in').is_file():
        return

    clusters = _atat_read_clusters(Path(wdir))
    predstr = _atat_read_predstr(Path(wdir))
    gs = _atat_read_gs(Path(wdir))
    fit = _atat_read_fit(Path(wdir))
    try:
        with open(Path(wdir) / 'maps.log', 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            cve = lines[-1]
    except Exception:
        cve = ''

    for idx in alive_it(fit['index'],
                        title="Getting sublattice deviations"):
        vaspdir = IMDGVaspDir(f"{idx}/ATAT")
        tolerance = 0.2  # tested for graphite-Na/AA/0%
        if vaspdir.converged:
            try:
                displ = structure_distance(
                    vaspdir.initial_structure,
                    vaspdir.structure,
                    match_first=True,
                    tol=tolerance,
                    norm=True)
            except ValueError:
                # structures are too different
                displ = structure_distance(
                    vaspdir.initial_structure,
                    vaspdir.structure,
                    tol=tolerance,
                    match_first=False,
                    norm=True)
        else:
            displ = np.nan
        fit.loc[fit['index'] == idx, 'sublattice deviation'] = displ
        gs.loc[gs['index'] == idx, 'sublattice deviation'] = displ

    logger.info("Saving sublattice deviation to %s", 'fit2.out')
    fit.to_csv("fit2.out", sep=' ')
    print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
          + colored("ATAT ", "magenta")
          + "Saved sublattice deviations to fit2.out")

    global_title = str(Path(wdir).absolute())

    fig, axs = plt.subplots(2, 3, figsize=(19.2, 9.6))
    fig.suptitle(global_title, fontsize=16)

    plt.rcParams['lines.markersize'] = 3

    _atat_plot_fitted_energies(axs[0, 0], predstr, gs, fit)
    _atat_plot_calculated_energies(axs[0, 1], predstr, gs, fit)
    _atat_plot_calc_vs_fit_energies(axs[0, 2], fit)
    _atat_plot_residuals(axs[1, 0], cve, fit)
    _atat_plot_sublattice_deviation(axs[1, 1], gs, fit)
    _atat_plot_clusters(axs[1, 2], clusters)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
    output_png = 'atat-summary-test.png'
    plt.savefig(output_png, dpi=300)

    logger.info("Saving ATAT plots to %s", output_png)
    print(colored(f"{wdir.replace("./", "")}: ", attrs=['bold'])
          + colored("ATAT ", "magenta")
          + f"Saved ATAT plots to {output_png}")


def atat(args):
    """Create ATAT visualization.
    """
    for wdir, subdirs, _ in os.walk(args.dir):
        subdirs.sort()  # this will make loop go in order
        _atat_1(wdir)
    return 0


def visualize(args):
    """Main routine.
    """
    return args.func_derive(args)
