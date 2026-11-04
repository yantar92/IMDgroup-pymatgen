"""Deprecated compatibility package for the MatRiFFlow rename.

``IMDgroup`` was the import package name before the project was renamed to
MatRiFFlow (import package ``matrifflow``).  Importing anything under
``IMDgroup`` emits a :class:`DeprecationWarning` and resolves to the
corresponding ``matrifflow`` module, so existing scripts keep working
during the transition.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_LEGACY = "IMDgroup"
_TARGET = "matrifflow"

# The 8 CLI subcommand modules were renamed ``imdg*`` -> ``mrf*`` during the
# MatRiFFlow rename.  The prefix-only mapping above cannot recover those, so
# map their basenames explicitly.  Everything else shares its basename.
_CLI_REMAP = {
    "imdg": "mrf",
    "imdg_create": "mrf_create",
    "imdg_derive": "mrf_derive",
    "imdg_diff": "mrf_diff",
    "imdg_analyze": "mrf_analyze",
    "imdg_status": "mrf_status",
    "imdg_visualize": "mrf_visualize",
    "imdg_run": "mrf_run",
}

warnings.warn(
    "IMDgroup is deprecated; use matrifflow instead.",
    DeprecationWarning,
    stacklevel=2,
)


class _AliasLoader(importlib.abc.Loader):
    """Loader that returns the already-imported real module.

    Reusing the real module object (rather than re-executing its source)
    means ``IMDgroup.X`` *is* ``matrifflow.X``, so ``isinstance`` checks,
    pickling, and pymatgen's MSONable registry keep working across the two
    import paths.
    """

    def __init__(self, real_name: str):
        self._real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self._real_name)

    def exec_module(self, module):
        # Already executed by create_module()'s import_module() call.
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _LEGACY or not fullname.startswith(_LEGACY + "."):
            return None
        real_name = _TARGET + fullname[len(_LEGACY):]
        pkg, _, leaf = real_name.rpartition(".")
        if leaf in _CLI_REMAP:
            real_name = f"{pkg}.{_CLI_REMAP[leaf]}"
        real_spec = importlib.util.find_spec(real_name)
        if real_spec is None:
            return None
        spec = importlib.util.spec_from_loader(fullname, _AliasLoader(real_name))
        if real_spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(real_spec.submodule_search_locations)
        return spec


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
