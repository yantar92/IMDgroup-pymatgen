"""Tests for IMDgroup.pymatgen.io.vasp.diagnostics.

These records and containers are our own data structures, so the
MSONable round-trip contract is exercised here directly (see the
testing principle in tests/conftest.py).
"""

from __future__ import annotations

import pytest
from pymatgen.util.testing import MatSciTest

from IMDgroup.pymatgen.io.vasp.diagnostics import (
    VaspWarning,
    VaspWarningRecord,
    VaspWarnings,
)


class TestVaspWarningRecord(MatSciTest):
    def test_merge_accumulates_count_and_takes_newer(self) -> None:
        """message/tips/metadata take the newer value; count accumulates.

        source is the exception: it keeps the first non-empty value.
        """
        left = VaspWarningRecord(
            name="x", message="old", tips=["a"], count=2,
            source="s1", metadata={"k": 1},
        )
        right = VaspWarningRecord(
            name="x", message="new", tips=["b"], count=3,
            source="s2", metadata={"k": 2, "j": 3},
        )
        merged = left.merge(right)
        assert merged.name == "x"
        assert merged.message == "new"
        assert merged.tips == ["b"]
        assert merged.count == 5
        assert merged.source == "s2"
        assert merged.metadata == {"k": 2, "j": 3}

    def test_merge_empty_tips_keeps_original(self) -> None:
        """A record with no tips keeps the original tips on merge."""
        left = VaspWarningRecord(name="x", message="m", tips=["a"])
        right = VaspWarningRecord(name="x", message="m2")
        assert left.merge(right).tips == ["a"]

    def test_merge_mismatched_names_raises(self) -> None:
        """Merging records with different names is rejected."""
        with pytest.raises(ValueError):
            VaspWarningRecord(name="a", message="m").merge(
                VaspWarningRecord(name="b", message="m"))

    def test_msonable(self) -> None:
        record = VaspWarningRecord(
            name="x", message="m", tips=["t"], count=2,
            source="s", metadata={"k": 1},
        )
        self.assert_msonable(record)


class TestVaspWarnings(MatSciTest):
    def test_add_merges_same_name(self) -> None:
        """Adding two records with the same name merges them."""
        warnings = VaspWarnings()
        warnings.add(VaspWarningRecord(name="x", message="m", count=1))
        warnings.add(VaspWarningRecord(name="x", message="m2", count=1))
        assert len(warnings) == 1
        assert warnings["x"].count == 2
        assert warnings["x"].message == "m2"

    def test_overwrite_replaces(self) -> None:
        """overwrite replaces a record instead of merging it."""
        warnings = VaspWarnings()
        warnings.add(VaspWarningRecord(name="x", message="m", count=1))
        warnings.overwrite(VaspWarningRecord(name="x", message="m3", count=7))
        assert warnings["x"].count == 7
        assert warnings["x"].message == "m3"

    def test_iter_names_len_has(self) -> None:
        """The container exposes iteration, names, length, and membership."""
        warnings = VaspWarnings()
        warnings.add(VaspWarningRecord(name="x", message="m"))
        warnings.add(VaspWarningRecord(name="y", message="m"))
        assert list(warnings) == ["x", "y"]
        assert warnings.names() == {"x", "y"}
        assert warnings.has("x")
        assert not warnings.has("z")

    def test_init_from_records(self) -> None:
        """A constructor list is folded through add(), merging duplicates."""
        warnings = VaspWarnings([
            VaspWarningRecord(name="x", message="m"),
            VaspWarningRecord(name="x", message="m2"),
        ])
        assert len(warnings) == 1
        assert warnings["x"].count == 2

    def test_emit(self) -> None:
        """emit warns with the VaspWarning category per record."""
        warnings = VaspWarnings([VaspWarningRecord(name="x", message="hello")])
        with pytest.warns(VaspWarning, match="hello"):
            warnings.emit()

    def test_msonable(self) -> None:
        warnings = VaspWarnings([
            VaspWarningRecord(
                name="x", message="m", tips=["t"], count=1,
                source="s", metadata={"k": 1},
            )
        ])
        self.assert_msonable(warnings)
