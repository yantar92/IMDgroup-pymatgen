# Why this exists

This page names the recurring manual tasks the package helps to
automate. If you recognize them, you know what to reach for.

## Automatic VASP input setup

You start from a Materials Project id, a CIF you already have, or a
formula, and you want a sane VASP directory. `imdg create` produces it;
`imdg derive` produces the variant (functional, k-point density, strain,
supercell, NEB, pseudopotential, constraints) as a new directory instead
of a hand edit.

See [imdg create](cli.md#imdg-create) and [imdg derive](cli.md#imdg-derive) for the full command reference.

## One-command VASP submission

Every cluster wants its own sbatch script, module loads, and queue
guesses. `gorun` encodes these once in a config, picks the
earliest-finishing queue, backs up the previous run, checks convergence,
and regenerates POTCAR. The same `gorun` submits MACE fine-tuning or
an arbitrary GPU-dependent scripts.

See [gorun](gorun.md) for the submission reference.

## Detect convergence and anomalies in running/finished calculations

grep-ping OUTCARs does not scale. `imdg status` collapses running /
converged / warning states across a tree and flags suspicious structural
changes (large displacement, symmetry change, stress / force anomalies)
for a human look.

See [imdg status](cli.md#imdg-status).

## Overview for many calculations

`imdg analyze` reports energy, %vol, displacement, lattice changes, and
max force across a tree, grouped by identical INCAR settings, and cached
so re-running is fast.

See [imdg analyze](cli.md#imdg-analyze).

## Comparing runs

Two energies can look fine while differing in functional, POTCAR, ENCUT,
k-points, or constraints. `imdg analyze` groups by settings, `imdg diff`
shows where INCARs or structures differ, and energies judged unreliable
are masked rather than silently used.

See [imdg analyze](cli.md#imdg-analyze) and [imdg diff](cli.md#imdg-diff).
