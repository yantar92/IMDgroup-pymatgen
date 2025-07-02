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
    """Group elements of list LST comparing them using CMP_EQ function.
    CMP_EQ should take two elements of the list as arguments and
    return True iff they are equal.
    Optional TITLE_FUNCTION argument is a function used to generate
    item name in the logs.
    Return a list of grouped lists.
    """
    groups = []

    def _add_to_groups(item):
        """Add ITEM to groups.
        If ITEM is not the same with all items in groups,
        create a separate group.
        Modifies "groups" by side effect.
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
