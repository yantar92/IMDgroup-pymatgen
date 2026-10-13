# Python API

The Python layer mirrors the CLI and adds programmatic access to parsed
VASP data with persistent caching.

## VASP directory handling: `IMDGVaspDir`

`IMDGVaspDir` gives dictionary-like access to a VASP directory with a
persistent disk cache, so repeated analysis of large trees does not
re-parse XML or OUTCAR files every time.

The cache is optimized for supercomputer file system, avoiding
excessive IO.

```python
from IMDgroup.pymatgen.io.vasp.vaspdir import IMDGVaspDir

vdir = IMDGVaspDir("path/to/vasp/calculation")

structure = vdir.final_structure
energy = vdir.final_energy
incar = vdir["INCAR"]

if vdir.converged:
    print("Calculation converged")

# NEB directories
if vdir.nebp:
    for image_dir in vdir.neb_dirs():
        print(f"Image {image_dir.path}: {image_dir.final_energy}")
```

`vdir.final_energy_reliable` returns a float only when the energy is
judged usable, otherwise `"unreliable"` (for example, a volume/shape
relaxation) or `"unconverged"`.

The cache is change-aware: it stores a file signature and re-parses only
directories whose relevant files changed.

## ATAT structure files

```python
from IMDgroup.pymatgen.core.structure import IMDStructure

s = IMDStructure.from_file("str.out")  # vacancies become "X"
```

## Transformations

### Inserting molecules

```python
from IMDgroup.pymatgen.transformations.insert_molecule import InsertMoleculeTransformation

transformer = InsertMoleculeTransformation(
    molecule="Li",
    step=0.2,
    proximity_threshold=0.75,
)

inserts = transformer.all_inserts("host_structure.cif")
```

### Symmetry cloning

Given a `host` structure and a `defect` structure, generate every
symmetry-equivalent placement of an interstitial under the host's
space group.

```python
from IMDgroup.pymatgen.transformations.symmetry_clone import SymmetryCloneTransformation
from pymatgen.core import Structure

host = Structure.from_file("host.cif")
defect = Structure.from_file("host_with_interstitial.cif")

trans = SymmetryCloneTransformation(sym_operations=host)
clones = trans.get_all_clones(host_with_interstitial)
```

## Diffusion analysis

```python
from IMDgroup.pymatgen.diffusion.neb import get_neb_pairs

pairs = get_neb_pairs(
    structures=stable_sites_list,
    prototype=host_structure,
    cutoff="auto",
    remove_compound=True,
)

for start, end in pairs:
    print(f"Path from {start} to {end}")
```
