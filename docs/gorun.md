# Job submission: `gorun`

`gorun` is a single interface for submitting Slurm jobs on the clusters
the group uses (for example Athena, Ares, Helios, LUMI). It is not VASP
only: the same command submits VASP, MACE multihead fine-tuning, ATAT
`maps`, and arbitrary GPU jobs.

Common behavior across all subcommands:

- cluster-specific module loads and environment are read from a TOML
  config;
- `--queue` picks a partition, or the scheduler selects the one that
  finishes earliest;
- previous outputs are backed up before a new run;
- convergence is checked so an already-finished job is not resubmitted
  (override with `--force`);
- `--mark` prepares a directory and writes a `gorun_ready` marker without
  submitting;
  - use `gorun-all-ready.sh` to submit all marked directories with load-balancing
- `--dir` runs the pipeline over several directories in one process.

## Configuration

gorun looks for its config in this order:

1. `--config PATH` (explicit)
2. `$IMDGroup/dist/etc/gorun.toml` (group checkout, when `IMDGroup` is set)
3. `~/.config/gorun/gorun.toml` (or `$XDG_CONFIG_HOME/gorun/gorun.toml`)

If none is found, gorun prints a short message with a copy-paste example
and exits.

The config maps hostnames to cluster definitions, queues, module loads,
and resource defaults:

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

See `examples/gorun-config.example.toml` for a more complete example.

VASP itself needs two environment variables:

```bash
export VASP_PATH=/path/to/vasp
export VASP_PP_PATH=/path/to/potentials
```

`VASP_PATH` points at the VASP installation (binaries under `bin/`);
`VASP_PP_PATH` at the pseudopotential directory gorun's ASE-based POTCAR
generation reads.

## `gorun vasp`

Submit a VASP job from the current directory. Plain `gorun` is shorthand
for `gorun vasp`.

```bash
gorun vasp --nodes 2 --time-limit 24:00:00
gorun vasp --vasp gam --nodes 1 --time-limit 1:00:00
gorun vasp --incar ISIF:3 NSW:100
gorun vasp --local
```

Before submitting, `gorun` runs safety and convergence checks, backs up
the previous run, sanitizes inputs, regenerates POTCAR, and applies any
`--incar` overrides.

### `INCAR.py`: Python around the run

If an `INCAR.py` file is present in the directory (and `--no-incar-py`
is not set), `gorun` does not invoke VASP directly. Instead it wraps the
run in a Python script that sets up an ASE `Vasp` calculator, reads the
existing `INCAR` / `POTCAR` / `KPOINTS` / `POSCAR`, and then runs your
`INCAR.py` body before and after the calculation.

Inside `INCAR.py` you have three objects available:

- `atoms` — the ASE `Atoms` from `POSCAR`;
- `calc` — the pre-configured ASE `Vasp` calculator;
- `vaspdir` — an `IMDGVaspDir` for the working directory.

Use it to attach constraints, adjust the calculator, or run custom logic
around the job:

```python
from ase.constraints import FixAtoms

atoms.set_constraint(FixAtoms(mask=[a.symbol == "Li" for a in atoms]))
energy = atoms.get_potential_energy()
```

When `atoms.get_potential_energy()` runs, the calculator invokes
`VASP_COMMAND`, which `gorun` has pointed back at
`gorun vasp --local --no-incar-py --force --no-clean --keep POSCAR` so
the actual VASP call still goes through gorun's own local runner.

Disable `INCAR.py` for one submission with `--no-incar-py`.

## `gorun mace-finetune`

**Experimental.** The MACE adapter is under active development; its flags
and `INCAR.toml` layout may change.

Submit a MACE multihead fine-tuning job (the two-stage replay protocol:
`fine_tuning_select` then `run_train`). Training hyperparameters live in
`INCAR.toml` in the working directory, analogous to `INCAR`.

```bash
gorun mace-finetune \
    --model-path 13_MACE-MATPES-PBE-0_medium.model \
    --replay-xyz matpes-pbe-replay-data.xyz \
    --data-path result.pkl \
    --e0s "{3:-0.01686124,6:-1.26246048,11:-0.22829243}"
```

Override any parameter from the CLI with `--incar`:

```bash
gorun mace-finetune --incar run_train.batch_size:4 run_train.lr:0.0005
```

A custom `RUNFILE.sh` or `RUNFILE.py` in the working directory replaces
the canned two-stage protocol when you need a different loop.

## `gorun gpu`

Submit a generic GPU job. The computation is whatever script you point
it at:

```bash
gorun gpu --script-name train.py --time-limit 48:00:00
```

`--script-name` names the script to run; `.py` files run with
`python -u`, everything else with `bash`. The script inherits the
cluster GPU environment (module loads, PyTorch flags, cache directories)
from the config's `[gpu]` → `setup`.

If you do not pass `--script-name`, `gorun gpu` falls back to a
`RUNFILE.sh` or `RUNFILE.py` in the working directory. This is a
convenience for keeping the command short when a project uses one
conventional entry file:

```bash
# uses RUNFILE.sh or RUNFILE.py in the current directory
gorun gpu --time-limit 48:00:00
```

## `gorun maps`

Submit an ATAT `maps` (cluster expansion) run, using `gorun-atat-local`
as the per-structure worker.

```bash
gorun maps --kpoints 3000 --max_strain 0.1
```

## `gorun atat-local`

The worker `pollmach` calls for each `str.out`; creates VASP inputs and
runs a validate-then-relax-then-SCF pipeline with distance, k-point,
strain, and sublattice checks.

## Batch submission with load balancing

Prepare many directories, then submit with a cap on concurrent jobs:

```bash
gorun vasp --mark --dir run1 run2 run3
gorun-all-ready.sh -n 20
```

`gorun-all-ready.sh` recursively finds `gorun_ready` markers and submits
only as many jobs as keep the running count under `-n`, so a large batch
does not flood the scheduler. It waits and re-checks the queue between
submissions.
