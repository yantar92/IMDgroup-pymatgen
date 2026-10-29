

# IMDgroup-pymatgen

Python and command-line tools for **iterative development of
computational workflows** for VASP atomistic simulations, built on top
of [pymatgen](https://pymatgen.org/).

Works with ordinary calculation directories produced manually, by scripts,
or by **any existing workflow tooling that writes standard VASP files**.
The independent commands do not require a workflow manager or prescribe
how jobs must be executed.

Also includes `gorun`, a cluster-agnostic job-submission script for VASP,
MACE fine-tuning, ATAT, and generic GPU jobs on Slurm clusters.

API documentation: <https://yantar92.github.io/IMDgroup-pymatgen/>

![img](docs/img/workflow-development.svg)

`imdg` can handle large collections of calculations (including
running) and detect anomalies or incomparable results.  It
deliberately **does not automate fixing anomalies**, leaving it up to
the user to explore the problems and identify the solutions (with the
optional help of `imdg` tooling).


## Workflow manager compatibility

`imdg` operates directly on standard VASP directory trees without a
workflow manager-specific database or import. It can complement manual
calculations, custom scripts, and existing workflow managers.

`imdg analyze` groups runs by identical settings, `imdg diff` compares
INCAR settings and structures, and `imdg analyze` provides calculation
summaries.


## High-throughput optimization

`imdg` can handle directory trees containing **thousands of
calculations**. Parsing VASP outputs (XML, OUTCAR) is costly on
supercomputer file systems, so the tools cache parsed results on disk
(with invalidation if the original data changes).

See the [Python API](https://yantar92.github.io/IMDgroup-pymatgen/python_api.html) for `IMDGVaspDir`, the class providing cached
directory inspection.


## Installation

    git clone https://github.com/yantar92/IMDgroup-pymatgen
    cd IMDgroup-pymatgen
    pip install .

This installs the `imdg` and `gorun` commands.

Note: the `lmdb` dependency (on-disk result cache) links against the
native LMDB library. On some systems &ndash; especially HPC clusters &ndash; you
must provide it before `pip install .`, for example `module load lmdb`
or the system dev package (`apt install liblmdb-dev`,
`dnf install lmdb-devel`). If the build fails while compiling `lmdb`,
load the module and re-run `pip install .`.


## Next steps

-   [Quickstart](https://yantar92.github.io/IMDgroup-pymatgen/quickstart.html): example graphite (mp-48) workflow
-   [Why this exists](https://yantar92.github.io/IMDgroup-pymatgen/concepts.html): semi-automated workflow development
-   [Command-line reference](https://yantar92.github.io/IMDgroup-pymatgen/cli.html) and [job submission](https://yantar92.github.io/IMDgroup-pymatgen/gorun.html)
-   [Python API](https://yantar92.github.io/IMDgroup-pymatgen/python_api.html)


## Contributing

Contributions are welcome.

-   Report bugs and propose features by opening an [issue](https://github.com/yantar92/IMDgroup-pymatgen/issues).
-   Ask questions and discuss design in [GitHub Discussions](https://github.com/yantar92/IMDgroup-pymatgen/discussions).
-   Submit changes as [pull requests](https://github.com/yantar92/IMDgroup-pymatgen/pulls).

You can also contact the maintainer directly by email
(`yantar92@posteo.net`) or on GitHub.

