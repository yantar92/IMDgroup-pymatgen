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


"""Tests for IMDgroup.common."""

from __future__ import annotations

from IMDgroup.common import groupby_cmp, short_path


def test_groupby_cmp_groups_equivalent() -> None:
    """Equivalent items are grouped together regardless of position."""
    assert groupby_cmp([1, 2, 3, 1, 2], lambda a, b: a == b) == \
        [[1, 1], [2, 2], [3]]


def test_groupby_cmp_all_distinct() -> None:
    """Items that never compare equal each get their own group."""
    assert groupby_cmp([1, 2, 3], lambda a, b: a == b) == [[1], [2], [3]]


def test_groupby_cmp_empty() -> None:
    """An empty input yields no groups."""
    assert groupby_cmp([], lambda a, b: a == b) == []


def test_groupby_cmp_custom_equality() -> None:
    """The caller-supplied comparison decides group membership."""
    result = groupby_cmp([1, 2, 3, 4], lambda a, b: a % 2 == b % 2)
    assert result == [[1, 3], [2, 4]]


def test_groupby_cmp_title_function() -> None:
    """title_function is invoked once per item with the item itself."""
    calls: list = []
    groupby_cmp([1, 2], lambda a, b: a == b, title_function=calls.append)
    assert calls == [1, 2]


def test_short_path_returns_basename() -> None:
    """The final path component is returned for compact logging."""
    assert short_path("/net/storage/.../relax.2") == "relax.2"


def test_short_path_empty_name_falls_back() -> None:
    """A path whose final component is empty returns the full path."""
    assert short_path(".") == "."
