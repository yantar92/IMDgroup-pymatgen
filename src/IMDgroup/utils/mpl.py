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

"""Helpers for matplotlib plotting.
"""
import matplotlib.pyplot as plt
from cycler import cycler

A4_WIDTH = 4.13 * 2


def mpl_defaults(font_size=8, width=A4_WIDTH, ratio=1.5) -> None:
    """Setup sensible global matplotlib defaults for publication-style plots.
    """
    # Adjust font sizes based on the provided font_size argument
    base_sz = font_size
    base_markersize = base_sz * 0.5
    height = width / ratio

    # More unique plot styles
    colors = plt.cm.tab20c(range(0, 20, 2))
    dashes = ['-', '--', '-.', ':']
    custom_cycler = (cycler(linestyle=dashes) * cycler(color=colors))
    plt.rc('axes', prop_cycle=custom_cycler)

    plt.rcParams.update({
        'figure.figsize': (width, height),
        'font.size': base_sz,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'axes.prop_cycle': custom_cycler,
        'axes.labelsize': base_sz,
        'axes.titlesize': base_sz * 1.2,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.2,
        'lines.markersize': base_markersize,
        'xtick.labelsize': base_sz,
        'ytick.labelsize': base_sz,
        'legend.fontsize': base_sz,
        'figure.titlesize': base_sz * 1.2,
    })
