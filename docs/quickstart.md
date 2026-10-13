# Quickstart: graphite (mp-48)

Example calculation of Li insertion into graphite.
It retrieves graphite structure from Materials Project,
and calculates possible stable Li positions inside the structure.

Note that each command is independent: point any of them at an
existing VASP directory tree and it works, whether that tree came from
`imdg`, your own scripts, or a workflow manager.

## Prerequisites

`imdg create mp-48` fetches the structure from Materials Project, which
needs a Materials Project API key in `~/.pmgrc.yaml`:

    pmg config --add PMG_MAPI_KEY <key>

Writing POTCAR needs a pseudopotential directory, also in `~/.pmgrc.yaml`:

    pmg config --add PMG_VASP_PSP_DIR /path/to/potentials

`gorun` regenerates POTCAR through a different path and reads
`VASP_PP_PATH` instead (see [gorun](gorun.md)).

## Cluster setup

`gorun` needs a config and two VASP paths before it can submit. A
shortened real config for the Ares CPU cluster, single queue:

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
search path, and `examples/gorun-config.example.toml` for the full
version with more clusters and the GPU / MACE / ATAT sections).

```bash
export VASP_PATH=/path/to/vasp
export VASP_PP_PATH=/path/to/potentials
```

Alternatively, you can use your own way to run VASP.
Just skip the `imdg run` step and run everything yourself.

## 1. Create the starting directory

```bash
imdg create mp-48
```

Writes a ready VASP directory (INCAR, POTCAR, KPOINTS, POSCAR) with
defaults. Don't like a default? Derive a variant instead of editing by
hand.

## 2. Switch functional

```bash
imdg derive mp-48 --output graphite_vdw functional optB88-vdW
```

[screenshot: imdg derive functional]

## 3. Insert Li (one-to-many)

```bash
imdg derive graphite_vdw --output inserts ins Li --step <STEP> --threshold 0.75
```

Writes one directory per symmetry-inequivalent insertion site.

[screenshot: imdg derive ins]

<TODO: real --step and any other insert options for graphite>

## 4. Run

```bash
imdg run vasp --nodes <NODES> --time-limit <HH:MM:SS>
```

`gorun` picks the earliest-finishing queue by default, backs up any
previous run, checks convergence, and regenerates POTCAR.

For many directories, prepare first and submit with load balancing:

```bash
gorun vasp --mark --dir inserts
gorun-all-ready.sh -n <MAX_CONCURRENT>
```

`gorun-all-ready.sh` keeps at most `-n` jobs queued at once so a large
batch does not flood the scheduler.

<TODO: real nodes / time / queue for graphite on the target cluster>

## 5. Monitor

```bash
imdg status
imdg status --problematic
```

Collapses running / converged / warning states across the tree and flags
suspicious structural changes (large displacement, symmetry change,
stress / force anomalies) for a human look.

[screenshot: imdg status]

## 6. Analyze

```bash
imdg analyze
```

Tabular summary of energy, %vol, displacement, lattice changes, max
force, grouped by identical INCAR settings. Energies judged unreliable
are masked on purpose (shown as `unreliable` / `unconverged`) rather than
silently used.

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
imdg visualize hull     . --ion Li
imdg visualize voltage  . --ion Li
```

See [cli](cli.md) for NEB, ATAT, hull, and voltage options.

[screenshot: imdg visualize voltage]
