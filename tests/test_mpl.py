# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of MatRiFFlow package
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


"""Tests for matrifflow.utils.mpl."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from matrifflow.utils.mpl import A4_WIDTH, mpl_defaults


def test_a4_width_constant() -> None:
    """A4_WIDTH is the full A4 width in inches (8.26 in)."""
    assert A4_WIDTH == pytest.approx(8.26, abs=0.01)


def test_defaults_sets_figure_size() -> None:
    """Figure size is (width, width * ratio)."""
    mpl_defaults(width=5.0, ratio=0.5)
    assert plt.rcParams["figure.figsize"] == [5.0, 2.5]


def test_defaults_sets_dpi() -> None:
    """Screen and savefig DPI are set independently."""
    mpl_defaults(dpi=200, savefig_dpi=300)
    assert plt.rcParams["figure.dpi"] == 200
    assert plt.rcParams["savefig.dpi"] == 300


def test_defaults_sets_font_sizes() -> None:
    """Base font size drives the derived label/tick sizes."""
    mpl_defaults(font_size=12)
    assert plt.rcParams["font.size"] == 12
    assert plt.rcParams["axes.labelsize"] == 12
    assert plt.rcParams["axes.titlesize"] == pytest.approx(12 * 4 / 3)
    assert plt.rcParams["legend.fontsize"] == pytest.approx(12 / 1.2)
    assert plt.rcParams["xtick.labelsize"] == pytest.approx(12 / 1.2)
    assert plt.rcParams["ytick.labelsize"] == pytest.approx(12 / 1.2)


def test_defaults_enables_publication_settings() -> None:
    """Constrained layout and editable-text fonts are switched on."""
    mpl_defaults()
    assert plt.rcParams["figure.constrained_layout.use"] is True
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    assert plt.rcParams["mathtext.default"] == "regular"


def test_defaults_sets_prop_cycle() -> None:
    """The custom cycler combines 10 colors with 4 linestyles (40 entries)."""
    mpl_defaults()
    cycle = plt.rcParams["axes.prop_cycle"]
    assert len(cycle) == 40
    assert {"linestyle", "color"} <= set(cycle.keys)
