"""Shared test infrastructure and conventions.

Testing principle: test IMDgroup code, not pymatgen.

IMDgroup-pymatgen is a thin layer over pymatgen.  When one of our
classes delegates to a pymatgen method, we do not re-test the upstream
behaviour.  Instead we test only what is ours:

- the values we compute and pass to pymatgen,
- the transformations and decisions we make around the pymatgen call,
- our own data structures, file I/O, and CLI dispatch.

This keeps the suite small and avoids coupling tests to pymatgen
internals that may move.

Where pymatgen already provides the canonical fixture or helper, reuse
it instead of maintaining a copy:

- ``pymatgen.util.testing.MatSciTest`` provides ``assert_msonable``,
  ``assert_str_content_equal``, ``serialize_with_pickle``, and an
  autouse temporary-directory fixture.  Subclass it in test classes
  that need those helpers.  Note: ``PymatgenTest`` is deprecated; use
  ``MatSciTest``.
- ``pymatgen.util.structures`` ships curated example structures.  Load
  them with ``MatSciTest.get_structure(name)`` instead of committing our
  own structure fixtures.
"""
