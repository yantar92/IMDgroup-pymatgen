IMDgroup-pymatgen
==================

Pymatgen add-ons for the `Inverse Materials Design group <https://www.oimalyi.org/>`_.

This package provides VASP workflow tools, caching infrastructure for
high-throughput calculations, structure transformations, and diffusion
analysis -- built on top of pymatgen.

Installation
------------

.. code-block:: bash

   git clone https://git.sr.ht/~yantar92/IMDgroup-pymatgen
   cd IMDgroup-pymatgen
   pip install .

Quick start
-----------

.. code-block:: python

   from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

   vdir = IMDGVaspDir("path/to/vasp/calculation")
   print(vdir.final_energy)
   print(vdir.structure)

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api
