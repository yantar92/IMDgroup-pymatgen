# Quickstart

Example calculation of Li insertion into graphite.
It retrieves graphite (AB) structure from Materials Project,
and calculates possible stable Li positions inside the structure.

Note that each command is independent: you can point any of them at an
existing VASP directory tree and it works, regardless whether that
tree came from `imdg`, your own scripts, or a workflow manager.

## Installation

The `imdg` and `gorun` commands come from the `IMDgroup-pymatgen`
package, which also pulls in `pymatgen` (and with it the `pmg` command
used below):

```bash
    git clone https://github.com/yantar92/IMDgroup-pymatgen
    cd IMDgroup-pymatgen
    pip install .
```

Check it worked:

```bash
    imdg --help
```

## Configuration

### Materials Project API setup (optional)

`imdg create mp-48` fetches the structure from Materials Project, which
needs a Materials Project API key. Create a free account at
[Materials Project](https://materialsproject.org) and copy your key from
the [API dashboard](https://next-gen.materialsproject.org/api):

```bash
    mkdir -p ~/.config
    pmg config --add PMG_MAPI_KEY <your key>
```

### POTCAR references (optional)

Writing POTCAR needs a pseudopotential directory
file:

```bash
    pmg config --add PMG_VASP_PSP_DIR /path/to/potentials
```

If pseudopotential directory is missing, POTCAR files will be omitted
and you will need to provide them yourself.

You can also configure pseudopotentials later.  `imdg run` will
generate POTCAR files automatically.

### Cluster setup (optional)

`imdg run` (= `gorun`) needs a config file and a VASP installation
before it can submit slurm jobs. A shortened real config for the Ares
CPU cluster, single queue:

```toml
[cluster.names]
ares = ['login01.ares.cyfronet.pl']

[ares]
queues = ['plgrid']
VASP-setup = "module load intel/2023b imkl/2023.2.0"

[ares.defaults.sbatch]
nodes = 4
time = '72:00:00'
ntasks-per-node = 48

[ares.plgrid.sbatch]
partition = 'plgrid'
```

Save it as `~/.config/gorun/gorun.toml` (see [gorun](gorun.md) for the
search path). `gorun --example-config` prints a more complete example
with more clusters and the GPU / MACE sections.

`VASP_PATH` is the top-level VASP install directory (not the binary,
not `bin/`); gorun runs `$VASP_PATH/bin/vasp_std`, `vasp_ncl`, or
`vasp_gam`:

```bash
export VASP_PATH=/path/to/vasp   # the directory containing bin/vasp_std
```

Alternatively, you can use your own way to run VASP.
Just skip the `imdg run` step and run everything yourself.

## 1. Create the starting directory

```bash
imdg create mp-48
# if you do not have API key, download POSCAR from https://legacy.materialsproject.org/materials/mp-48/
# and save it as POSCAR.mp-48_C
imdg create --output C.mp-48.hexagonal.194 ./POSCAR.mp-48_C
```

Writes a ready VASP directory (INCAR, POTCAR, KPOINTS, POSCAR) with
defaults. The directory is auto-named from the system, for example
`C.mp-48.hexagonal.194` (formula, Materials Project ID, spacegroup).

Every `imdg` command writes an `imdg.log` in the directory you run it
from. It records what the command did and any warnings. Add `-v` to also
see the log on the terminal, or `-vv` for debug detail.

## 2. Switch functional to account for van der Waals correction

```bash
imdg derive C.mp-48.hexagonal.194 --output graphite_vdw functional optB88-vdW
```

Every derived directory also gets an `IMDVaspInputSet.log`. It records
the structure, the source directory it was derived from (`directory:`),
the inherited INCAR/KPOINTS, and the mutation applied (`name:` /
`functional:`), so you can trace any result back to its inputs.

If the source directory contains VASP output, `imdg derive` will use
the final structure.

## 3. Set up relax calculation (ISIF=2)

```bash
imdg derive graphite_vdw --output graphite_vdw.relax relax RELAX_POS
```

`relax RELAX_POS` sets ISIF=2: ionic positions relax while the cell
shape and volume stay fixed. It also sets ISTART=0, NSW=500, IBRION=2
(CG), EDIFF=1e-6, EDIFFG=-0.01.

See all available options via `imdg derive . relax --help`.

## 4. Create graphite supercell

```bash
imdg derive graphite_vdw.relax supercell 2x2x1
```

This will create a VASP input with identical setup, but POSCAR altered
to create supercell in `graphite_vdw.relax.2x2x1`.

## 5. Insert Li (one-to-many)

```bash
imdg derive graphite_vdw.relax.2x2x1 --output inserts ins Li --step 0.1 --threshold 0.5
imdg derive graphite_vdw.relax --output inserts ins Li --step 0.1 --threshold 0.5 --multithread
```

Writes one directory per symmetry-inequivalent insertion site.  The
`--threshold` is a proximity cutoff expressed as a multiple of the sum
of atomic radii (default `0.75`); lowering it to `0.5` accepts Li
positions closer to the carbon framework and yields the distinct sites.

## 6. Run

`imdg run` (= `gorun`) picks the earliest-finishing queue by default,
backs up any previous run to subfolders named `gorun_<N>`, checks
convergence, and auto-generates POTCAR/vdw kernel.

To run a single directory, do it from inside that directory:

```bash
(
cd inserts/graphite_vdw.relax.2x2x1.ins.Li.0;
imdg run vasp --time-limit 12:00:00
)
```

For many directories, prepare first and submit with load balancing:

```bash
# works from inside compute node
gorun vasp --mark --dir inserts/*
gorun-all-ready.sh -n 50 # max 50 jobs running
```

`gorun-all-ready.sh` keeps at most `-n` jobs queued at once so a large
batch does not flood the scheduler.

We recommend running `gorun-all-ready.sh` inside
[tmux](https://github.com/tmux/tmux/wiki), a terminal multiplexer that
keeps the submission loop alive while you are logged off or disconnected
from the terminal.

Alternatively, submit right away

```bash
# on login node, as slurm submissions are usually disabled from compute nodes
gorun vasp --time 12:00:00 --dir inserts/*
```

```
Running gorun in inserts/graphite_vdw.relax.ins.Li.0
Regenerating POTCAR
...graphite_vdw.relax.ins.Li.0: Copied vdw_kernel.bindat from ...
Candidate time (plgrid): 2026-11-01 09:21:00
Best finish time: 2026-11-01 09:21:00
Submitted batch job 25323794
Job submitted to SLURM scheduler.
...
```

## 7. Monitor

```bash
imdg status
# or just runs with issues
imdg status --problematic
```

Collapses running, converged, and warning states across the tree and
flags suspicious structural changes (large displacement, symmetry
change, stress and force anomalies) for a human review.

```
Scanning ['.'] for VASP directories |████████████████████████████████████████| 14 in 1.0s (13.99/s)
Directories containing VASP input but not output:
  C.mp-48.hexagonal.194
  graphite_vdw
  graphite_vdw.relax
  graphite_vdw.relax.2x2x1
8 runs: 3 running, 1 converged, 4 unconverged
   WHEN  STATUS                  DIR                      ENERGY (eV)  FMAX (eV/Å)  PER-STEP                   WARN                     DETAILS
    now  converged               inserts/graphite_vdw.re    -129.2873       0.0054  SCF: 1s±0s relax: 7s±2s    -                        CPU 0:02:55 (96 cor…
                                 lax.2x2x1.ins.Li.0
    now  running                 inserts/graphite_vdw.re            -            -  SCF: 1s±0s relax: 7s±3s    -                        step:   0.6015(harm…
                                 lax.2x2x1.ins.Li.1
    now  running                 inserts/graphite_vdw.re            -            -  SCF: 4s±0s relax: 46s±12s  -                        step:   0.9466(harm…
                                 lax.2x2x1.ins.Li.2
    now  running                 inserts/graphite_vdw.re            -            -  SCF: 4s±0s relax: 26s±13s  -                        step:   0.3067(harm…
                                 lax.2x2x1.ins.Li.3
 2m ago  incomplete vasprun.xml  inserts/graphite_vdw.re            -            -  SCF: 2s±0s relax: 12s      brmix, subspacematrix    DAV:   7     0.4398…
                                 lax.ins.Li.0
 3m ago  incomplete vasprun.xml  inserts/graphite_vdw.re            -            -  SCF: 2s±0s relax: 4s       edddav, subspacematrix   |     Error EDDDAV:…
                                 lax.ins.Li.1
 2m ago  incomplete vasprun.xml  inserts/graphite_vdw.re            -            -  SCF: 9s±0s relax: 45s      brmix, edddav,           |     Error EDDDAV:…
                                 lax.ins.Li.2                                                                  subspacematrix
 2m ago  incomplete vasprun.xml  inserts/graphite_vdw.re            -            -  SCF: 9s±1s relax: 47s      brmix, subspacematrix    DAV:   5    -0.3635…
                                 lax.ins.Li.3

Warnings:
  subspacematrix [unconverged] (4):
      inserts/graphite_vdw.relax.ins.Li.0 inserts/graphite_vdw.relax.ins.Li.1 inserts/graphite_vdw.relax.ins.Li.2 inserts/graphite_vdw.relax.ins.Li.3
      WARNING: Sub-Space-Matrix is not hermitian in DAV           12
      TIP: Might happen when NCORE/NPAR is too large or atoms are too close. As long as converged, should not affect final energy
...
Details:
...
```

## 8. Adjust setup for failed runs

At the previous steps, all the non-supercell structures failed with common issue reported as

```
  subspacematrix [unconverged] (4):
      inserts/graphite_vdw.relax.ins.Li.0 inserts/graphite_vdw.relax.ins.Li.1 inserts/graphite_vdw.relax.ins.Li.2 inserts/graphite_vdw.relax.ins.Li.3
      WARNING: Sub-Space-Matrix is not hermitian in DAV           12
      TIP: Might happen when NCORE/NPAR is too large or atoms are too close. As long as converged, should not affect final energy
```

The VASP fails there because the default NCORE that works for
realistic systems with many atoms, tends to fail when there are too
few atoms (and bands).

In addition, one Li per primitive graphite cell is extremely high Li
concentration that tends to create tricky electronic densities.

Fix this via

```bash
gorun --time 12:00:00 --incar NCORE:2 ALGO:All --dir inserts/graphite_vdw.relax.ins.Li.*
# You can also copy-paste the problematic folder list right from the imdg status output
```

```
Running gorun in inserts/graphite_vdw.relax.ins.Li.0
Backing up ...inserts/graphite_vdw.relax.ins.Li.0
Changed INCAR: {'NCORE': {'INCAR1': 16, 'INCAR2': 1}, 'ALGO': {'INCAR1': 'Normal', 'INCAR2': 'All'}}
Regenerating POTCAR
...
Job submitted to SLURM scheduler.
...
```

Now, everything should converge numerically.

```bash
imdg status --exclude gorun
```



## 9. Analyze

```bash
imdg analyze
```

Tabular summary of energy, %vol, displacement, lattice changes, max
force, grouped by identical INCAR settings. Energies judged unreliable
are masked (shown as `unreliable` / `unconverged`) to avoid accidental use.

[screenshot: imdg analyze]

## 10. Diff (compare results)

```bash
imdg diff incar inserts/*
imdg diff structure inserts/*
```

The INCAR comparison shows common settings and per-group differences.
The structure comparison groups directories that converged to the same
structure (symmetry-aware) and sorts groups by energy.

[screenshot: imdg diff structure]
