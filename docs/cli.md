# Command-line reference: `imdg`

`imdg` is a single entry point with subcommands. Get help at any level:

```bash
imdg --help
imdg analyze --help
imdg derive . ins --help
```

## `imdg create`

Generate a fresh VASP directory from a Materials Project id, a structure
file, or a formula.

```bash
imdg create mp-48
imdg create structure.cif
imdg create "Li 10x10x10"
```

`mp-*` ids require a Materials Project API key (`PMG_MAPI_KEY` in
`~/.config/.pmgrc.yaml`, set via `pmg config --add PMG_MAPI_KEY <key>`),
and POTCAR generation requires a pseudopotential directory
(`PMG_VASP_PSP_DIR` in `~/.config/.pmgrc.yaml`).

## `imdg derive`

Derive a new calculation from an existing VASP directory. This is the
primary tool for chaining calculations.

```bash
imdg derive <input_dir> [--output <output_dir>] <subcommand> [args]
```

Subcommands:

| Subcommand       | Purpose                                | Example                                                               |
|------------------|----------------------------------------|-----------------------------------------------------------------------|
| `functional`     | switch exchange-correlation functional | `imdg derive . functional optB88-vdW`                                 |
| `potcar`         | override pseudopotentials              | `imdg derive . potcar Li:Li_sv C:C`                                   |
| `incar`          | set or unset INCAR tags                | `imdg derive . incar PREC:Accurate EDIFF:1e-7 NELMIN:None`            |
| `kpoints`        | change k-point density or grid         | `imdg derive . kpoints --density 5000`                                |
| `relax`          | set up a relaxation (ISIF)             | `imdg derive . relax RELAX_POS_SHAPE_VOL`                             |
| `scf`            | static SCF calculation                 | `imdg derive . scf`                                                   |
| `strain`         | apply lattice strain                   | `imdg derive . strain --amin 0.98 --amax 1.02 --asteps 3`             |
| `supercell`      | make a supercell                       | `imdg derive . supercell 2x2x2`                                       |
| `perturb`        | random position perturbation           | `imdg derive . perturb --distance 0.1`                                |
| `fix`            | selective-dynamics constraints         | `imdg derive . fix "{'Li': [True, True, True]}"`                      |
| `insert` / `ins` | insert atom/molecule at sites          | `imdg derive . ins Li --step 0.2`                                     |
| `del`            | delete species                         | `imdg derive . del Li`                                                |
| `fill`           | fill all insertion sites               | `imdg derive . fill --uniq_sites dir1 dir2`                           |
| `neb`            | NEB between two runs                   | `imdg derive . neb target_dir --nimages 5`                            |
| `neb_diffusion`  | all unique diffusion paths             | `imdg derive framework_struct neb_diffusion --diffusion_points a b c` |
| `atat`           | input from ATAT str.out                | `imdg derive . atat str.out --kpoints 3000`                           |

`derive` is one-to-one (change a setting, get one directory) or
one-to-many (`ins`, `strain`, `neb_diffusion` write several
directories).

## `imdg status`

Check running status, convergence, warnings, and timing across a tree.

```bash
imdg status
imdg status --problematic
imdg status --exclude "test_runs"
```

Features:

- identifies running SLURM jobs;
- checks electronic, ionic, and multi-step convergence;
- surfaces VASP warnings and suspicious structural changes (large
  displacement, symmetry change, stress / force anomalies);
- reports convergence speed (wall-clock per electronic and ionic step).

## `imdg analyze`

Tabular summary of key properties across a tree. Grouping by identical
INCAR is on by default.

```bash
imdg analyze
imdg analyze --fields energy e_per_atom displ
imdg analyze --fields ALL
imdg analyze --nogroup
imdg analyze --format csv
imdg analyze --format json
```

Fields: `energy`, `e_per_atom`, `max_force`, `%vol`, `displ`,
`space_group`, `a`, `b`, `c`, `alpha`, `beta`, `gamma`, and the
`%`-changes of each. The table view shows a smaller set by
default. Pass `--fields ALL` to show every field.  Energies judged
unreliable (for example, from a volume/shape relaxation that shifted
the k-point grid) or unconverged are shown as `unreliable` /
`unconverged` instead of a number, so they are not silently used.

`--format` controls the output. The default `table` renders a rich
table; `csv`, `json`, and `orgtbl` bypass rich and emit all fields
(unless `--fields` narrows them) for programmatic use.

## `imdg diff`

Compare settings or structures across several directories.

```bash
imdg diff incar dir1 dir2 dir3
imdg diff structure dir1 dir2 dir3
imdg diff structure dirs --copy-to unique/
```

`incar` reports common settings and per-group differences. `structure`
groups directories that converged to the same structure (symmetry-aware)
and sorts the groups by energy; `--copy-to` copies one representative
per group.

## `imdg visualize`

Generate visual summaries.

```bash
imdg visualize neb
imdg visualize atat
imdg visualize . hull    --ion Li
imdg visualize . voltage --ion Li
```

`neb` writes a trajectory CIF for each converged NEB; `atat` summarizes
cluster-expansion fits; `hull` plots a formation-energy convex hull;
`voltage` plots a voltage profile.

## `imdg run`

Submit or run a job; forwards to `gorun` (see [gorun](gorun.md)).

```bash
imdg run vasp --nodes 2 --time-limit 24:00:00
# restart an existing run with adjusted setup
imdg run vasp --incar ALGO:All
```
