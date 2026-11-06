---
title: 'MatRiFFlow: A toolkit for iterative development of VASP workflows'
tags:
    - workflow development
    - VASP
    - pymatgen
    - Python
authors:
    - name: Ihor Radchenko
      orcid: 0000-0002-0498-7100
      affiliation: 1
    - name: Oleksandr I. Malyi
      orcid: 0000-0002-2366-2838
      affiliation: 1
affiliations:
    - name: Centre of Excellence ENSEMBLE3 Sp. z o. o., Wolczynska Str. 133, Warsaw, 01-919, Poland
      index: 1
date: 6 November 2026
bibliography: paper.bib
---

# Summary

In the field of atomistic simulations, there is an increasing interest
in developing automated, reliable, and reproducible calculation
workflows. A number of workflow automation libraries like AiiDA
[@huber_aiida_2020; @pizzi_aiida_2016], Atomate2
[@ganose_atomate2_2025], FireWorks [@jain_fireworks_2015], pyiron
[@janssen_pyiron_2019], or PerQueue [@sjolin_perqueue_2024] have been
developed to date. These libraries typically provide a graphical
interface and/or a Python API to define computational workflows, allow
executing these workflows, manage the calculation results, and even
recover from typical calculation failures. However, developing these
automated workflows is still largely an iterative process that often
requires quick experimentation, fixing unexpected failures, and
monitoring the physical consistency of the results.

MatRiFFlow is a command-line interface and a Python library that
streamlines common tasks encountered during computational workflow
development of atomistic calculations using VASP
[@kresse_efficient_1996]. It enables quick experimentation with
elements of more complex workflows and detects anomalies in running,
failed, and (nominally) finished calculations. It also provides easy
tooling to understand and test fixes for the identified issues before
integrating them into larger workflows. MatRiFFlow is designed to work
directly with VASP outputs, remaining workflow automation-agnostic. It
can also handle high-throughput calculations, letting users monitor
thousands of calculations through efficient caching.

# Statement of need

The existing workflow automation libraries for atomistic simulations
allow implementing and executing very complex workflows
[@sjolin_perqueue_2024; @uhrin_workflows_2021]. However, designing a
reliable workflow is not an easy job. The final published workflow
recipes can be very far from their initial versions, produced over
many development iterations, so that they can handle numerical
convergence issues, or various physical edge cases that cannot be
easily predicted without running the workflow on realistic inputs.

Indeed, numerical convergence or edge cases are common enough that it
is a pretty standard practice for the workflow libraries to allow
writing custom error handlers, where all possible edge cases can be
incorporated into the workflow itself. Moreover, libraries like AiiDA
[@huber_aiida_2020; @pizzi_aiida_2016] or Custodian [@ong_python_2013;
@CustodianVaspHandlers] provide ready-to-use error handlers for the
common failure modes (e.g. VASP failing to converge). However, writing
error handlers for non-standard errors that are specific to a given
workflow still requires writing custom logic as a part of the
workflow.

MatRiFFlow aims to make it easier and faster to develop custom error
handlers. It does not replace the need to incorporate the handlers
into the existing atomistic simulation workflows, but allows fast
detection of numerical and physical anomalies in VASP calculations
(both converged and unconverged), and, once a problem is confirmed,
quick experimentation necessary to identify the new calculation
parameters or logic that can then be implemented as a proper error
handler in the workflow library of the user's choice.

# State of the field

Automatic detection of calculation anomalies is not uncommon in the
field of atomistic simulations. For example, Custodian
[@ong_python_2013; @CustodianVaspHandlers] implements a comprehensive
suite for automatic VASP error recovery for unconverged VASP runs,
including VASP warning detection, and common recipes to converge
failed calculations depending on the reported VASP warnings. The
AiiDA-VASP module [@rico_haeuselmann_aiida-vaspaiida-vasp_2026] also
provides a set of automatic error recovery tools. Pymatgen
[@ong_python_2013] provides some simple warnings for numerically
unconverged VASP simulations or VASP simulations that finish with exit
code 0 but are not converged (e.g. structure relaxation runs where the
maximum number of relaxation steps is reached without achieving
convergence thresholds).

MatRiFFlow builds on these existing ideas, extending anomaly detection
from simple VASP failures or trivial convergence checks to detecting
*potential* errors, like large structure deformations, abnormal energy
trajectories, or crystal symmetry changes. Instead of fully automated
error recovery as in Custodian or AiiDA-VASP, MatRiFFlow is designed
to flag numerical and also physical anomalies in the calculations,
including nominally converged calculations. The anomalies may or may
not be real problems, depending on the nature of the computational
workflow being developed, but they may be otherwise difficult to
identify across huge numbers of calculations that are typically
performed while running high-throughput calculations.

# Software design

MatRiFFlow is designed with two primary use cases in mind: (1) a
command-line interface (CLI) for quick experimentation directly on
supercomputers; (2) a Python API allowing users to integrate
MatRiFFlow with other popular Python-based workflow managers, like
AiiDA or PerQueue. The CLI is split into independent modules,
illustrated in \autoref{fig:overview}, that allow (1) detecting
anomalies in running and completed VASP calculations, both
automatically and manually, by visual inspection (`mrf status`, `mrf
visualize`); (2) comparing the calculation results (`mrf analyze`,
`mrf diff`); (3) quickly creating new VASP inputs – from scratch or
using existing, possibly failed, calculations (`mrf create`, `mrf
derive`); (4) running the calculations on configured supercomputers
(`mrf run`). The modules, in principle, can be used to implement
simple complete workflows, where more complex workflow libraries are
overkill. However, the primary use case is rather using appropriate
CLI modules to augment workflow development with one of the existing
workflow libraries of the user's choice.

![MatRiFFlow is a tool assisting iterative workflow
 development. Rather than replacing the existing workflow managers,
 MatRiFFlow assists with detecting anomalies in the VASP calculations,
 comparing the results, and quick experimentation with problematic
 VASP calculations.\label{fig:overview}](overview.png)

The CLI implementation is heavily inspired by Pymatgen's `pmg` CLI,
with its sub-command design. `mrf analyze` and `mrf diff` are
extensions of Pymatgen's `pmg analyze` and `pmg structure --group`,
tailored to compare VASP calculations and detect outliers. `mrf
analyze` augments Pymatgen's original idea of a table summary of
multiple calculations by indicating the INCAR setting differences,
flagging potentially unreliable energies, and reporting geometry
changes after structure relaxation. `mrf diff` allows finding
calculations with abnormally high (or low) energies. `mrf derive`
utilizes ideas from the VASPKIT pre-processing CLI
[@wang_vaspkit_2021], extending VASPKIT's idea of transforming POSCAR
structures and k-point grids to complete VASP input or output
transformations, including applying selective dynamics constraints,
altering INCAR settings and pseudopotentials, building nudged elastic
band (NEB) paths, or automatically searching for potential atom or
molecule insertion sites (the full list of options can be found via
`mrf derive . --help`).

`mrf status` is one of the central CLI modules that is designed to
detect anomalies in running and finished VASP calculations. It is
inspired by automatic VASP warning detection as implemented in
Custodian, but extends it with Slurm [@klusacek_architecture_2023]
warnings and errors, and physical calculation consistency checks, like
detecting structure symmetry changes during relaxation, large atomic
displacements, non-monotonic energy trajectories, etc. All the
problems are reported in bulk, grouping VASP outputs with similar
anomalies together for easier inspection (copy-pasteable into the
command line for one-liners). A real-world example with `mrf status`
detecting symmetry changes in graphite-Na system is illustrated in
\autoref{fig:example}.

![ An example of `mrf status` detecting anomalies in converged VASP
 calculations of Na intercalation into expanded graphite. (a)
 Illustration of `mrf status` output. Despite all the VASP
 calculations numerically converging, some of the geometry
 optimizations yield crystal symmetry changes in the host
 graphite. (b) Visual comparison of the identified structures clearly
 showing that the initial AB graphite stacking is lost during the
 geometry optimization. This is not by itself an error, but was not
 intentional in this particular workflow.\label{fig:example}](example.png)

`mrf run` is an optional module that can be used to submit or
re-submit VASP calculations in the current directory, or in
bulk. While not strictly necessary for the users of workflow
libraries, the module is handy for quick experimentation in
conjunction with `mrf create` and `mrf derive`, as it takes care of
supercomputer-specific VASP setup, creating standard files like
`POTCAR` or `vdw_kernel.bindat`, backing up old failed VASP
calculations, and avoiding erroneous submissions of already-converged
or running calculations.  All the CLI modules are built on top of the
Python API that is based on the base Pymatgen classes. The two central
concepts used in the API are (1) a cached VASP directory container,
based on the `PMGDir` class from Pymatgen; (2) a "derived" VASP input
set, based on `VaspInputSet` from Pymatgen.

The VASP directory container (`MRFVaspDir`) is an abstraction allowing
fast access to thousands of VASP calculations without having to
re-parse potentially large VASP output files repeatedly, which can be
a major performance bottleneck on high-performance computing (HPC)
file systems like Lustre [@chaimov_scaling_2016]. As an example,
running `mrf status` on 2906 VASP outputs – a process that includes
parsing `vasprun.xml` and `OUTCAR` files – takes 39 minutes of
wall-clock time on uncached directories, but only 19 seconds after the
directories are cached. After adding a single calculation, `mrf
status` completes in 25 seconds on the updated set of 2907
calculations.

The `MRFVaspDir` container uses Lightning Memory-Mapped Database
(LMDB) [@Chu2011MDB] to store parsed (with Pymatgen) representations
of individual VASP output files. The database is managed
automatically, invalidating the cache when the underlying VASP files
are modified, keeping the original files as the source of truth
without having to watch for a stale cache. The parsed file objects are
serialized and de-serialized with Python's pickle module, allowing
fast loading of the data from the database. The LMDB is used to
minimize input-output on the file system and allow parallel access –
LMDB reduces file system access frequency that can be critical on
supercomputers, especially on the login nodes. This design differs
from Pymatgen's Borg module [@PymatgenBorg] that uses JSON
serialization, needs manual database management, and stores only a
fraction of the parsed data. MatRiFFlow's VASP directory container
does not require its users to know the implementation details, and
provides full access to all the data transparently, using the API
directly compatible with `PMGDir` from Pymatgen.

`MRFVaspDir` is also the basis of anomaly detection, augmenting
Pymatgen's default VASP parsers with additional consistency
verification. All the calculation anomalies are recorded, and
additionally reported as Python warnings, so that Python API users
automatically benefit from anomaly detection, as long as they use the
MatRiFFlow Python API as a replacement for Pymatgen defaults, even
without directly running the CLI tools.

Beyond accessing the existing VASP outputs, MatRiFFlow builds on
Pymatgen's notion of a VASP input set, extending it to allow
additional consistency validations of the VASP input parameters
(e.g. abnormal k-point grid densities or energy cutoffs) and utilizing
`MRFVaspDir` to quickly read and modify the existing VASP calculations
while retaining the ability to capture anomalies.

# Research impact statement

MatRiFFlow has been previously used (under the name IMDG-pymatgen) to
develop computational workflows for calculating diffusion barriers in
expanded graphite [@wang_lattice_2026] and polyethylene oxide (PEO)
[@du_interchain_2026]. It has also been used in conjunction with the
Alloy Theoretic Automated Toolkit (ATAT) [@van_de_walle_alloy_2002]
workflow to study hard carbon anodes [@radchenko_rational_2026]. More
recently, MatRiFFlow has also been used to implement a custom workflow
for studying Na storage inside carbon nanotubes
[@radchenko_cnt-pore-filling-dft_nodate].

# AI usage disclosure

The early versions of MatRiFFlow (formerly IMDG-pymatgen) were fully
developed without LLM use. Later versions made use of DeepSeek V4-Pro
and earlier versions as part of the development workflow and for code
generation. All LLM-generated code was reviewed and applied
manually. DeepSeek V4-Pro has been used to draft the initial version
of the online documentation, followed by manual editing and
verification. ChatGPT (OpenAI GPT-6) assisted with literature
verification and provided editorial suggestions during manuscript
preparation. The authors retain full responsibility for the MatRiFFlow
design, implementation, and the accuracy of the documentation.

# Acknowledgements

MatRiFFlow development has been funded by the National Centre for
Research and Development, Poland under the project
WPC3/2022/50/KEYTECH/2024. Institutional and infrastructural support
for the ENSEMBLE3 Centre of Excellence was provided through the
"Centre of ExcelleNce for nanophotonicS, advancEd Materials and novel
crystal growth-Based technoLogiEs - ENSEMBLE3" project which is
carried out within the 2.1 International Research Agendas programme of
the Foundation for Polish Science co-financed by the European Union
under the European Funds for Smart Economy 2021-2027 (FENG) and the
European Union Horizon 2020 research and innovation program Teaming
for Excellence (GA. No. 857543). The software was created as part of
the project of the Minister of Science and Higher Education "Support
for the activities of Centers of Excellence established in Poland
under the Horizon 2020 program" under contract
No. MEiN/2023/DIR/3797. The authors would also like to acknowledge the
Polish high-performance computing infrastructure, PLGrid, for access
to computational resources under project number PLG/2026/019890, and
the LUMI supercomputer, owned by the EuroHPC Joint Undertaking and
hosted by CSC (Finland) and the LUMI consortium, through project
number PLL/2025/09/018974.

# References
