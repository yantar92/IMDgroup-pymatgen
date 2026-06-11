"""Sphinx configuration for IMDgroup-pymatgen.

Generates API documentation from docstrings using automodule
with explicitly listed modules.
"""

import os
import sys
from importlib.metadata import version as get_version

# -- Path setup --------------------------------------------------------------
# Point to src/ so Sphinx can import the package
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "IMDgroup-pymatgen"
copyright = "2024-2025, Inverse Materials Design Group"
author = "Ihor Radchenko"
release = get_version("IMDgroup-pymatgen")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "autodocsumm",
    "myst_parser",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autoclass_content = "both"

# Autosummary settings
autosummary_generate = False
autosummary_imported_members = False

# Intersphinx links to pymatgen and standard library
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pymatgen": ("https://pymatgen.org", None),
}

# Templates
templates_path = ["_templates"]

# Source suffix
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Add module name before class/func names
add_module_names = False

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# Quiet warnings about undocumented members (the package has many)
suppress_warnings = ["autodoc.undoc"]

