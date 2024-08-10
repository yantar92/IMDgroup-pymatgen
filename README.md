

# IMDgroup-pymatgen

This is a set of add-ons and helpers for [pymatgen](https://pymatgen.org/), that are tailored to
research that is performed in [Inverse Materials Design group](https://www.oimalyi.org/).


# Installation

    # Download the package and source dependencies
    git clone https://git.sr.ht/~yantar92/IMDgroup-pymatgen
    
    # Activate virtual environment
    pip -m venv .venv
    . .venv/bin/activate
    
    # Install into the environment
    pip install IMDgroup-pymatgen


# Features

-   Generate all possible unique insertion sites of an atom or molecule
    into a given structure.
-   VASP input set optimized for cellulose relaxation.
    See Yadav, Malyi 2024 Cellulose (doi: [10.1007/s10570-024-05754-7](https://doi.org/10.1007/s10570-024-05754-7))


# Usage


## Loading VASP input sets from group publications

From Python

    from IMDgroup.pymatgen.io.vasp.sets import IMDRelaxCellulose
    
    # Create VASP input file generator, using default Cellulose ibeta
    # structure from Yadav, Malyi 2024 Cellulose
    input = IMDRelaxCellulose(structure='ibeta')
    
    # Write VASP input files into "test" directory
    input.write_input(output_dir='test', potcar_spec=True)


## Inserting molecule into a given structure

From command line

    # Generate all possible inserts of water into cellulose_ibeta and
    # write VASP input files into output_dir
    pmg-insert-molecule water.xyz cellulose_ibeta.cif output_dir

From Python

    from IMDgroup.pymatgen.transformations.insert_molecule\
        import InsertMoleculeTransformation
    from IMDgroup.pymatgen.io.vasp.sets import IMDRelaxCellulose
    import pymatgen.io.vasp.sets as vaspset
    import numpy as np
    
    transformer = InsertMoleculeTransformation(
        'water.xyz', step=2.0,
        anglestep=np.radians(90),
        matcher=None)
    
    structures = transformer.all_inserts('cellulose_ibeta.cif')
    
    vaspset.batch_write_input(
        structures,
        vasp_input_set=IMDRelaxCellulose,
        output_dir='output_dir',
        potcar_spec=True)


# TODO Citing


# Contributing

We welcome contributions in all forms. If you want to contribute,
please fork this repository, make changes and send us a pull request!

