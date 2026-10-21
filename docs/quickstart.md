# Quickstart: graphite (mp-48)

Example calculation of Li insertion into graphite.
It retrieves graphite (AB) structure from Materials Project,
and calculates possible stable Li positions inside the structure.

Note that each command is independent: you can point any of them at an
existing VASP directory tree and it works, regardless whether that
tree came from `imdg`, your own scripts, or a workflow manager.

## Installation

The `imdg` and `gorun` commands come from the IMDgroup-pymatgen
package, which also pulls in pymatgen (and with it the `pmg` command
used below):

    git clone https://github.com/yantar92/IMDgroup-pymatgen
    cd IMDgroup-pymatgen
    pip install .

Check it worked:

    imdg --help

## Prerequisites

`imdg create mp-48` fetches the structure from Materials Project, which
needs a Materials Project API key. Create a free account at
[Materials Project](https://materialsproject.org) and copy your key from
the [API dashboard](https://next-gen.materialsproject.org/api):

    mkdir -p ~/.config
    pmg config --add PMG_MAPI_KEY <your key>

Writing POTCAR needs a pseudopotential directory, written to the same
file:

    pmg config --add PMG_VASP_PSP_DIR /path/to/potentials

## Cluster setup (optional)

`imdg run` (= `gorun`) needs a config file and a VASP installation
before it can submit. A shortened real config for the Ares CPU
cluster, single queue:

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
with more clusters and the GPU / MACE / ATAT sections.

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
```

Writes a ready VASP directory (INCAR, POTCAR, KPOINTS, POSCAR) with
defaults. The directory is auto-named from the system, for example
`C.mp-48.hexagonal.194` (formula, Materials Project ID, spacegroup).

Every `imdg` command writes an `imdg.log` in the directory you run it
from. It records what the command did and any warnings. Add `-v` to also
see the log on the terminal, or `-vv` for debug detail.

## 2. Switch functional

```bash
imdg derive C.mp-48.hexagonal.194 --output graphite_vdw functional optB88-vdW
```

Every derived directory also gets an `IMDVaspInputSet.log`. It records
the structure, the source directory it was derived from (`directory:`),
the inherited INCAR/KPOINTS, and the mutation applied (`name:` /
`functional:`), so you can trace any result back to its inputs.

[screenshot: imdg derive functional]

## 3. Insert Li (one-to-many)

```bash
imdg derive graphite_vdw --output inserts ins Li --step 0.1 --threshold 0.5 --multithread
```

Writes one directory per symmetry-inequivalent insertion site.  The
`--threshold` is a proximity cutoff expressed as a multiple of the sum
of atomic radii (default `0.75`); lowering it to `0.5` accepts Li
positions closer to the carbon framework and yields the distinct sites.

[screenshot: imdg derive ins]

## 4. Run

`imdg run` (= `gorun`) picks the earliest-finishing queue by default,
backs up any previous run, checks convergence, and auto-generates
POTCAR/vdw kernel.

To run a single directory, do it from inside that directory:

```bash
(
cd inserts/graphite_vdw.ins.Li.0;
imdg run vasp --nodes 1 --time-limit 12:00:00
)
```

For many directories, prepare first and submit with load balancing:

```bash
gorun vasp --mark --dir inserts/*
gorun-all-ready.sh -n 5 # max 5 jobs running
```

`gorun-all-ready.sh` keeps at most `-n` jobs queued at once so a large
batch does not flood the scheduler.

We recommend running `gorun-all-ready.sh` inside
[tmux](https://github.com/tmux/tmux/wiki), a terminal multiplexer that
keeps the submission loop alive while you are logged off or disconnected
from the terminal.

<TODO: real nodes / time / queue for graphite on the target cluster>

## 5. Monitor

```bash
imdg status
imdg status --problematic
```

Collapses running / converged / warning states across the tree and flags
suspicious structural changes (large displacement, symmetry change,
stress / force anomalies) for a human review.

[screenshot: imdg status]

## 6. Analyze

```bash
imdg analyze
```

Tabular summary of energy, %vol, displacement, lattice changes, max
force, grouped by identical INCAR settings. Energies judged unreliable
are masked (shown as `unreliable` / `unconverged`) to avoid accidental use.

[screenshot: imdg analyze]

## 7. Diff (compare results)

```bash
imdg diff incar inserts/*
imdg diff structure inserts/*
```

The INCAR comparison shows common settings and per-group differences.
The structure comparison groups directories that converged to the same
structure (symmetry-aware) and sorts groups by energy.

[screenshot: imdg diff structure]

## 8. Perturb and rerun

```bash
imdg derive <DIR> --output <DIR>.perturbed perturb --distance <DIST>
gorun vasp --nodes <NODES> --time-limit <HH:MM:SS>
imdg diff structure <DIR> <DIR>.perturbed
```

Breaking symmetry before a relaxation is a one-command delta, not a
hand-edited set of files.

<TODO: real perturb --distance and which directory>

## Visualize

Visualization is independent of the loop above and usually runs on a
finished set of results:

```bash
imdg visualize . hull    --ion Li
imdg visualize . voltage --ion Li
```

See [cli](cli.md) for NEB, ATAT, hull, and voltage options.

[screenshot: imdg visualize voltage]
