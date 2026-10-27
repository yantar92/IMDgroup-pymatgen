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


"""Build the Sphinx documentation and fail on warnings.

Documentation is deployed by a separate workflow, so a broken docstring,
a dangling cross-reference, or an unsupported theme option would
otherwise only be noticed after the change is merged.  This test runs
the same build locally with ``-W``, which turns Sphinx warnings into
errors.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent / "docs"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sphinx") is None,
    reason="sphinx is not installed (pip install .[docs])",
)

# Intersphinx downloads the python and pymatgen inventories.  Offline the
# download fails and Sphinx emits a warning that says nothing about our
# sources, so that one warning is ignored.
_INVENTORY_WARNING = "failed to reach any of the inventories"


def _warning_blocks(stderr: str) -> list[str]:
    """Group Sphinx stderr into warning blocks, continuation lines included.

    A warning starts with ``WARNING:`` or with a source location followed
    by ``WARNING:``; the message itself may continue on further lines.
    """
    blocks: list[str] = []
    for line in stderr.splitlines():
        if "WARNING:" in line:
            blocks.append(line)
        elif blocks:
            blocks[-1] = f"{blocks[-1]}\n{line}"
    return blocks


def test_docs_build_without_warnings(tmp_path: Path) -> None:
    """``sphinx-build -W`` on docs/ must report no warnings."""
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx",
            "-b", "html",
            "-W", "--keep-going", "-q",
            str(DOCS_DIR), str(tmp_path / "html"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    blocks = _warning_blocks(result.stderr)
    unexpected = [
        block for block in blocks if _INVENTORY_WARNING not in block
    ]
    assert not unexpected, "\n\n".join(unexpected)
    # A non-zero exit is acceptable only when every warning was the
    # unreachable-intersphinx-inventory one above.
    assert result.returncode == 0 or (blocks and not unexpected), result.stderr
