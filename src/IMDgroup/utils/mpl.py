# MIT License
#
# Copyright (c) 2024-2025 Inverse Materials Design Group
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

"""Helpers for matplotlib plotting."""
import matplotlib.pyplot as plt
from cycler import cycler

A4_WIDTH = 4.13 * 2  # full A4 width in inches


def mpl_defaults(
        *,
        font_size: float = 12,
        width: float = A4_WIDTH / 2,
        ratio: float = 0.75,
        dpi: int = 300,
        savefig_dpi: int = 600,
) -> None:
    """Configure matplotlib rcParams for publication-quality figures.

    Applies consistent settings: editable PDF text, direction-in
    ticks, minor ticks, constrained layout, and a custom
    color/linestyle cycler.  Scripts that need further customization
    (e.g. a seaborn base style) should call this function *after* any
    style sheet import so these values take precedence.

    Args:
        font_size: Base font size in pt.
        width: Figure width in inches (default A4_WIDTH/2 = single-column).
        ratio: Height / width ratio (default 0.75 = 3:4).
        dpi: Screen DPI for interactive display.
        savefig_dpi: DPI for saved figures.
    """
    height = width * ratio

    # Custom color + linestyle cycler (IMDgroup-specific).
    colors = plt.cm.tab20c(range(0, 20, 2))
    dashes = ['-', '--', '-.', ':']
    custom_cycler = cycler(linestyle=dashes) * cycler(color=colors)
    plt.rc('axes', prop_cycle=custom_cycler)

    plt.rcParams.update({
        'figure.figsize': (width, height),
        'figure.dpi': dpi,
        'savefig.dpi': savefig_dpi,

        # Use constrained layout so that the axes rectangle within the
        # figure is determined by the layout engine rather than by
        # ad-hoc subplot-parameter defaults.  When all figures include
        # the same set of text elements (title, xlabel, ylabel -- even
        # if some are empty strings), constrained_layout produces
        # identical axes sizes across figures, keeping the data area
        # consistent.  Without this, adding or removing a title
        # silently shrinks or expands the axes box while the total
        # figure dimensions stay fixed, which confuses visual
        # comparison and panel alignment in publications.
        #
        # IMPORTANT: saving with bbox_inches='tight' or calling
        # fig.tight_layout() defeats this consistency.  Both
        # operations recompute the layout and can change the saved
        # figure dimensions.  Save with only the filename and dpi.
        'figure.constrained_layout.use': True,

        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size * 4 / 3,
        'legend.fontsize': font_size / 1.2,
        'xtick.labelsize': font_size / 1.2,
        'ytick.labelsize': font_size / 1.2,

        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,

        'lines.linewidth': 1.2,
        'lines.markersize': 3.5,

        'pdf.fonttype': 42,          # editable text in Illustrator
        'ps.fonttype': 42,
        'mathtext.default': 'regular',
    })
