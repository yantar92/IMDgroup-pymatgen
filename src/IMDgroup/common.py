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


"""Common routines used across the sub-libraries.
"""
import logging

logger = logging.getLogger(__name__)


def groupby_cmp(lst, cmp_eq, title_function=None):
    """Group elements of a list by pairwise comparison.

    Groups consecutive elements using a caller-supplied equality function.

    Args:
        lst: List of elements to group.
        cmp_eq: Function that takes two elements and returns True if they
            should belong to the same group.
        title_function: Optional function that takes an element and returns
            a string used in log messages.  When None, "??" is logged.

    Returns:
        list[list]: List of grouped lists.  Each sublist contains elements
        that are pairwise equivalent according to cmp_eq.
    """
    groups = []

    def _add_to_groups(item):
        """Add item to groups, creating a new group if needed.

        Mutates the enclosing ``groups`` list by side effect.

        Args:
            item: Element to place into an existing or new group.
        """
        in_group = False
        for group in groups:
            for group_item in group:
                if cmp_eq(item, group_item):
                    logger.debug(
                        'Appending %s to existing group',
                        title_function(item) if title_function is not None
                        else "??"
                    )
                    group.append(item)
                    in_group = True
                    break
            if in_group:
                break
        if not in_group:
            logger.debug(
                'Creating a new group for %s',
                title_function(item) if title_function is not None
                else "??"
            )
            # pylint: disable=modified-iterating-list
            groups.append([item])

    for item in lst:
        _add_to_groups(item)

    return groups
