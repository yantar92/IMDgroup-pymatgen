"""Tests for IMDgroup.common."""

from __future__ import annotations

from IMDgroup.common import groupby_cmp


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
