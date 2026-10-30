# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of IMDgroup-pymatgen package
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Common routines used across the sub-libraries."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def short_path(path) -> str:
    """Return the final path component for compact log messages.

    gorun working directories live under long cluster mount points
    (e.g. ``/net/storage/.../generations.Li/.../relax.2``).  Repeating
    the full path on every line drowns out the useful information, so
    per-step messages only show the final component.  The working
    directory is announced once at the start of a run.
    """
    name = Path(path).name
    return name if name else str(path)


def configure_progress_output():
    """Send ``alive_progress`` output to stderr.

    ``alive_progress`` writes progress bars and their final receipt to
    stdout by default.  For CLI tools, stdout is reserved for actual
    results, so redirect progress to stderr.  This is a process-wide
    setting; call it once at each CLI entry point.
    """
    import sys
    from alive_progress import config_handler
    # alive_progress 3.1.5 monkey-patches ``set_global`` onto the
    # ``config_handler`` callable.  Pyright only sees the function type
    # and cannot resolve the attached attribute.
    config_handler.set_global(file=sys.stderr)  # pyright: ignore[reportFunctionMemberAccess]


def suppress_spglib_warnings():
    """Silence spglib's raw C-level warnings.

    spglib prints internal retry diagnostics directly to stderr when
    symmetry detection needs several attempts (e.g. ``"spglib:
    ssm_get_exact_positions failed."``, ``"spglib: No point group was
    found"``).  They are harmless and duplicate the structured warnings
    surfaced by IMDgroup's own diagnostics, but they interleave with
    rich tables and corrupt the display.

    spglib reads ``SPGLIB_WARNING`` on every call (see its
    ``warning_enabled``), so setting it at runtime is enough.  We use
    ``setdefault`` so a user who has already chosen a value keeps
    control; only the absent-key case is forced to ``OFF``.
    """
    import os
    os.environ.setdefault("SPGLIB_WARNING", "OFF")


def groupby_cmp(lst, cmp_eq, title_function=None):
    """Group elements of a list by pairwise equality.

    Each item is compared only against the first member of every existing
    group, so the number of ``cmp_eq`` calls is O(n * g), where ``g`` is
    the number of groups produced, instead of O(n^2) over group members.

    This is exact when ``cmp_eq`` is an equivalence relation (reflexive,
    symmetric, and transitive).  A non-transitive ``cmp_eq`` still works,
    but it is only compared against the first member of each group, so
    items that would match a later member but not the representative may
    be split into separate groups.

    Args:
        lst: List of elements to group.
        cmp_eq: Function that takes two elements and returns True if they
            should belong to the same group.  Ideally an equivalence
            relation; see above for the behavior when it is not
            transitive.
        title_function: Optional function that takes an element and returns
            a string used in log messages.  When None, "??" is logged.

    Returns:
        list[list]: List of grouped lists.  Each sublist contains elements
        that are pairwise equivalent according to cmp_eq.
    """
    groups = []
    for item in lst:
        title = title_function(item) if title_function is not None else "??"
        for group in groups:
            if cmp_eq(item, group[0]):
                logger.debug('Appending %s to existing group', title)
                group.append(item)
                break
        else:
            logger.debug('Creating a new group for %s', title)
            groups.append([item])
    return groups
