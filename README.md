

# IMDgroup-pymatgen

This package provides a set of add-ons and helpers for [pymatgen](https://pymatgen.org/),
tailored to research performed in the [Inverse Materials Design group](https://www.oimalyi.org/).


# Key Features & Optimizations


## Optimized for High-Throughput (Caching)

Unlike standard pymatgen tools, this library is specifically designed
to handle **huge numbers of VASP outputs** efficiently.

-   **Persistent Caching:** The core `IMDGVaspDir` class implements a
    persistent disk cache (stored in `~/.cache/imdgVASPDIRcache` or
    `XDG_CACHE_HOME`).
-   **Lazy Loading:** Parsed VASP data is cached automatically. Subsequent
    analysis of the same directories is orders of magnitude faster,
    making it feasible to analyze thousands of calculations repeatedly
    without re-parsing XML or OUTCAR files.


## Automated Health Checks & Warnings

The library proactively monitors VASP calculations for common
pitfalls, essential for reliable high-throughput workflows:

-   **VASP Warnings:** Automatically parses logs (e.g., `vasp.out`,
    `slurm*.out`) to surface execution warnings like convergence
    failures or grid errors.

-   **Structural Integrity:** Checks for physical anomalies during
    relaxation to catch problems early:
    -   **Large Displacements**: Warns if atoms move significantly more than
        expected (e.g., indicating bond breaking or phase instability).
    -   **Symmetry Changes**: Detects if the framework symmetry breaks
        during relaxation, which often indicates a failed calculation or
        unwanted phase transition.
    -   **Stress & Forces**: Monitors for unexpected hydrostatic stress or
        excessive residual forces.


## Extensions to pymatgen

While heavily based on `pymatgen`, this library introduces several key
differences:

-   `imdg analyze` vs `pmg analyze`::
    -   The `imdg analyze` command is built on the caching `IMDGVaspDir` class.
    -   It focuses on **relative changes**: instead of just reporting
        absolute values, it calculates changes between the initial and
        final structures (e.g., volume change `%vol`, lattice parameter
        changes `%a`, `%b`, `%c`, and total atomic displacement `displ`).
    -   It supports grouping runs by identical `INCAR` parameters to
        easily compare different calculation settings.
-   **Enhanced Input Sets:** New input sets like `IMDDerivedInputSet` allow deriving new
    calculations (relaxations, strains, NEB) directly from existing
    output directories, preserving context/history.


# Getting Help


## Command Line

You can get help for any command directly in the terminal:

    # General help and list of subcommands
    imdg --help
    
    # Help for a specific subcommand (e.g., arguments for analyze)
    imdg analyze --help
    imdg derive . ins --help


## Python Code

All classes and functions are documented with docstrings. You can
access them using Python's built-in `help()` function:

    from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
    help(IMDGVaspDir)


# Installation

    git clone https://git.sr.ht/~yantar92/IMDgroup-pymatgen
    cd IMDgroup-pymatgen
    pip install .


# Command Line Interface

The package installs command line tool: `imdg` - the master script for
VASP workflows.


## `imdg status`

Monitor the status of VASP calculations in the current or specified directories.

    # Check status of all runs in current directory and subdirectories
    imdg status
    
    # Only show problematic runs (warnings or failures)
    imdg status --problematic
    
    # Exclude specific directories using regex
    imdg status --exclude "test_runs"

Features:

-   Identifies running SLURM jobs.
-   Checks convergence (electronic, ionic, and multi-step sequences).
-   Parses and highlights VASP warnings (e.g., convergence failures, internal errors).
-   Displays NEB image convergence summaries.


## `imdg analyze`

Summarize key properties of VASP outputs in a tabular format. This
tool uses caching to quickly re-analyze large directory trees.

    # Analyze all runs in the current directory
    imdg analyze
    
    # Report specific fields only
    imdg analyze --fields energy e_per_atom displ
    
    # Group results by similar INCAR parameters
    imdg analyze --group

**Available Fields**:

-   `energy`, `e_per_atom`: Final reliable energy and energy per atom.
-   `%vol`: Percentage change in volume between initial and final structure.
-   `displ`: Average atomic displacement between initial and final structure.
-   `%a`, `%b`, `%c`, `%alpha`, `%beta`, `%gamma`: Percentage change in lattice parameters.
-   `total_mag`: Total magnetization.


## `imdg create`

Generate fresh VASP inputs from various sources.

    # Create input from Materials Project ID
    imdg create mp-48
    
    # Create input from a CIF/POSCAR file
    imdg create structure.cif
    
    # Create a box with a single atom
    imdg create "Li 10x10x10"


## `imdg derive`

This is the primary tool for chaining VASP calculations. It creates
new input sets derived from existing VASP directories (inputs or
outputs), allowing for complex workflows like relaxation chains,
strain application, or NEB setup.


### General usage

    # Derive a new calculation in 'new_dir' based on 'old_dir'
    imdg derive old_dir --output new_dir <subcommand> [args]


### Subcommands

-   `relax`: Set up relaxation (ISIF 2-7).
    
        imdg derive . --output relax_run relax RELAX_POS_SHAPE_VOL
-   `scf`: Set up static self-consistent field calculation.
-   `kpoints`: Change K-point density.
    
        imdg derive . --output dense_kpoints kpoints --density 5000
-   `strain`: Apply lattice strain (e.g., for elastic constants).
    
        imdg derive . --output strain_run strain --amin 0.98 --amax 1.02 --asteps 3
-   `perturb`: Perturb atomic positions (e.g., to break symmetry).
-   `supercell`: Generate a supercell.
-   `functional`: Switch DFT functional (e.g., `PBE`, `PBE+D3-BJ`, `optB88-vdW`).
-   `incar`: Modify specific INCAR tags.
    
        imdg derive . --output high_prec incar PREC:Accurate EDIFF:1e-7
-   `fix`: Apply selective dynamics constraints.
-   `ins`: Insert atoms/molecules into voids (see `pmg-insert-molecule`).
-   `fill`: Fill sites based on relaxed unique insertion points.
-   `atat`: Generate input for ATAT (Alloy Theoretic Automated Toolkit) from a VASP run.


### NEB and Diffusion

`imdg derive` includes specialized tools for Nudged Elastic Band (NEB) calculations.

    # Simple NEB between current dir and target dir
    imdg derive . neb target_dir --nimages 5
    
    # Complex diffusion analysis: find all unique paths between stable sites
    imdg derive prototype_dir neb_diffusion --diffusion_points site1_dir site2_dir site3_dir --nimages 5

The `neb_diffusion` subcommand analyzes the topology of interstitial
sites and automatically generates unique diffusion paths between them.


## `imdg diff`

Compare structures or input parameters between directories.

    # Compare structures in directories, grouping identical ones
    imdg diff structure dir1 dir2 dir3
    
    # Compare INCAR files, showing differences
    imdg diff incar dir1 dir2


## `imdg visualize`

Generate visual summaries of calculations.

    # Visualize NEB trajectory (creates .cif files)
    imdg visualize neb
    
    # Visualize ATAT cluster expansion results (convex hull, residuals)
    # Requires running inside an ATAT directory
    imdg visualize atat


## `pmg-insert-molecule`

Systematically insert a molecule or atom into a host structure at
various positions and orientations, making sure to cover all viable
positions without overlaps.

    # Insert water molecule into host.cif, stepping 0.5A grid, rotating 45 degrees
    pmg-insert-molecule water.xyz host.cif output_dir --step 0.5 --anglestep 45

There is also `imdg ins` subcommand counterpart that can directly use VASP
folder as input.


# Python API


## VASP Directory Handling

The `IMDGVaspDir` class provides a dictionary-like interface to VASP
directories with aggressive caching to handle high-throughput analysis
on file systems like Lustre.

    from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir
    
    # Initialize (data will be loaded from cache if available)
    vdir = IMDGVaspDir("path/to/vasp/calculation")
    
    # Access parsed pymatgen objects
    structure = vdir.structure
    energy = vdir.final_energy
    incar = vdir["INCAR"]
    
    # Check convergence
    if vdir.converged:
        print("Calculation converged!")
    
    # Handling NEB directories
    if vdir.nebp:
        for image_dir in vdir.neb_dirs():
            print(f"Image {image_dir.path}: {image_dir.final_energy}")


## Transformations


### Inserting Molecules

Find void space and insert species.

    from IMDgroup.pymatgen.transformations.insert_molecule import InsertMoleculeTransformation
    
    transformer = InsertMoleculeTransformation(
        molecule='Li',  # or Molecule object
        step=0.2,       # Grid spacing in Angstroms
        proximity_threshold=0.75
    )
    
    # Get list of all unique insertion structures
    inserts = transformer.all_inserts('host_structure.cif')


### Symmetry Cloning

Generate all symmetrically equivalent configurations of a structure.

    from IMDgroup.pymatgen.transformations.symmetry_clone import SymmetryCloneTransformation
    from pymatgen.core import Structure
    
    host = Structure.from_file("host.cif")
    # Structure with one interstitial site
    defect = Structure.from_file("defect.cif")
    
    # Generate all symmetrically equivalent defects
    # relative to the host symmetry
    trans = SymmetryCloneTransformation(sym_operations=host)
    clones = trans.get_all_clones(defect)


## Diffusion Analysis

The `get_neb_pairs` function automates the discovery of unique
diffusion paths in a material.

    from IMDgroup.pymatgen.diffusion.neb import get_neb_pairs
    
    # Given a list of stable interstitial sites (structures) and the host prototype
    # calculate all unique hops between them.
    pairs = get_neb_pairs(
        structures=stable_sites_list,
        prototype=host_structure,
        cutoff='auto',         # Automatically determine max hop distance
        remove_compound=True   # Remove multi-step paths if single steps exist
    )
    
    for start, end in pairs:
        print(f"Path from {start} to {end}")


# Acknowledgements

We acknowledge financial support from the National Centre for Research
and Development (NCBR) under project
WPC3/2022/50/KEYTECH/2024. Computational resources were provided by
the Polish high-performance computing infrastructure PLGrid, including
access to the LUMI supercomputer—owned by the EuroHPC Joint
Undertaking and hosted by CSC in Finland together with the LUMI
Consortium—through allocation PLL/2024/07/017633, as well as
additional resources at the PLGrid HPC centres ACK Cyfronet AGH and
WCSS under allocation PLG/2024/017498.

