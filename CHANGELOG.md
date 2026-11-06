# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

## 2.0.0 - 2026-11-06

### Renamed to MatRiFFlow

The project was renamed from **IMDgroup-pymatgen** to **MatRiFFlow**:

-   Import package `IMDgroup` -> `matrifflow`; importing `IMDgroup` still
    works but emits a `DeprecationWarning` and redirects to the new modules.
-   CLI `imdg` -> `mrf`; the `imdg` entry point remains as a deprecated
    alias.
-   `IMDGVaspDir` -> `MRFVaspDir`, and other `IMD*` prefixes -> `MRF*`.
-   Documentation and docstrings updated; the GitHub Pages URL is now
    `yantar92.github.io/MatRiFFlow`.

### Merged gorun

`IMDgroup-gorun` was merged into this repository, providing the `imdg run`,
command.

### New features

-   Compressed VASP files (for example `.gz`) are auto-detected and read
    throughout `mrf` and `gorun`.
-   `mrf analyze`:
    -   Outputs framework symmetry.
    -   Uses a rich table and adds a programmatic output mode; `--short`
        was dropped.
-   `mrf status`: output reworked around `rich` with a live table and
    progress display.
-   `gorun` (or `mrf run`):
    -   `--local` runs VASP outside the cluster.
    -   Accepts a human-readable time limit (`--time-limit`, aliased as
        `--time`).
    -   `--mark` and `--local` are now mutually exclusive.
-   `mrf diff` uses a new, more intuitive argument order (breaking change).

### Other

-   Python version requirement bumped; the `tomli` fallback (Python < 3.11)
    was dropped.
-   Various fixes, expanded test coverage, and stricter type checking.

## 1.2.0 - 2026-10-06

-   New `imdg derive` subcommand `potcar` (override pseudopotentials with
    `ELEMENT:POTCAR` pairs, written to `INCAR.toml`).
-   `imdg status`:
    -   Reports convergence speed: mean +/- standard deviation of wall-clock
        time per electronic (SCF) cycle and per ionic step.
    -   New `--skip-running` flag.
    -   Renamed `--skip_converged` to `--skip-converged`.
-   `INCAR.toml` now records non-default metadata (the resolved POTCAR
    mapping, the k-point grid density, and the exchange-correlation
    functional) so that `gorun` and other tools can reproduce the written
    inputs.
-   `IMDDerivedInputSet` transfers `INCAR.toml` from the source directory.
-   `IMDGVaspDir` is now string-like (`__fspath__`), so instances can be
    used wherever a filesystem path is expected.
-   Dependencies: `tomli` added (Python < 3.11).

## 1.1.0 - 2026-09-11

-   New `imdg visualize` subcommands:
    -   **`hull`:** plot a formation-energy convex hull from VASP outputs or
        pickled entries.
    -   **`voltage`:** plot a voltage profile.
    -   **`atat`:** gained `--plot_extra` and `--classic_residuals` options.
-   New `imdg derive` subcommand `fix` (selective-dynamics constraints),
    plus new options: `fill --tol`, `neb_diffusion --limit`, and
    `kpoints --grid`.
-   `imdg analyze`:
    -   Grouping by INCAR is now the default (`--nogroup` disables it;
        previously enabled with `--group`).
    -   New `--short` flag.
    -   New `max_force` and `space_group` fields.
-   `imdg status`:
    -   Reports maximum residual force.
    -   Summarizes INCAR variation across runs.
-   `IMDGVaspDir` cache rewritten on top of LMDB, with cache versioning,
    age-based eviction, periodic flushing, and cached VASP logs.
-   New `IMDgroup.utils.mpl.mpl_defaults` helper for consistent matplotlib
    styling.
-   New `IMDStructure.from_structure` constructor and
    `structure_remove_duplicates` helper.
-   `InsertMoleculeTransformation` now honors the insertion limit exactly.
-   `get_neb_pairs` prioritizes shorter paths and skips very short hops.
-   Documentation: Sphinx API reference and GitHub Pages deployment.
-   Dependencies: pymatgen version bumped; `duecredit` imported directly;
    `lmdb` added.

## 1.0.0 - 2025-11-25

First tagged release.
