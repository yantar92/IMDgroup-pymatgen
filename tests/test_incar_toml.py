"""Tests for IMDgroup.pymatgen.io.vasp.incar_toml.

This module is IMDgroup's own INCAR.toml reader/writer; it does not
delegate to pymatgen, so its full behaviour is tested here (see the
testing principle in tests/conftest.py).
"""

from __future__ import annotations

from pathlib import Path

from IMDgroup.pymatgen.io.vasp.incar_toml import (
    _format_toml,
    _toml_format_value,
    read_incar_toml,
    write_incar_toml,
)


# --- _toml_format_value -----------------------------------------------------


def test_toml_format_value_bool() -> None:
    """Booleans become unquoted true/false literals."""
    assert _toml_format_value(True) == "true"
    assert _toml_format_value(False) == "false"


def test_toml_format_value_numbers() -> None:
    """Ints and floats are written unquoted."""
    assert _toml_format_value(1) == "1"
    assert _toml_format_value(1.5) == "1.5"


def test_toml_format_value_string_escapes_quotes_and_backslashes() -> None:
    """Strings are quoted and escape backslashes plus double quotes."""
    assert _toml_format_value("plain") == '"plain"'
    assert _toml_format_value('say "hi"') == '"say \\"hi\\""'
    assert _toml_format_value("back\\slash") == '"back\\\\slash"'


def test_toml_format_value_nonscalar_falls_back_to_repr() -> None:
    """Non-scalar values are stringified and quoted as a fallback."""
    assert _toml_format_value([1, 2]) == '"[1, 2]"'
    assert _toml_format_value(None) == '"None"'


# --- _format_toml -----------------------------------------------------------


def test_format_toml_nested_dict_becomes_section() -> None:
    """Dict values become [section] tables; scalars stay top-level."""
    rendered = _format_toml({"m": 3, "sec": {"a": 2, "z": 1}})
    assert rendered == "m = 3\n\n[sec]\na = 2\nz = 1"


# --- read_incar_toml --------------------------------------------------------


def test_read_incar_toml_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing file yields an empty dict."""
    assert read_incar_toml(tmp_path / "missing.toml") == {}


def test_read_incar_toml_valid_file(tmp_path: Path) -> None:
    """A valid TOML file is parsed with native value types."""
    path = tmp_path / "INCAR.toml"
    path.write_text('key = "value"\nnumber = 3\nflag = true\n')
    assert read_incar_toml(path) == {"key": "value", "number": 3, "flag": True}


# --- write_incar_toml -------------------------------------------------------


def test_write_incar_toml_bootstrap_writes_all_non_none(tmp_path: Path) -> None:
    """Bootstrap writes every non-None value and skips None."""
    path = tmp_path / "INCAR.toml"
    write_incar_toml({"a": 1, "b": "x", "c": None}, path=path)
    assert path.read_text() == 'a = 1\nb = "x"\n'


def test_write_incar_toml_bootstrap_always_include_writes_none(
        tmp_path: Path) -> None:
    """Bootstrap writes None values for keys in always_include."""
    path = tmp_path / "INCAR.toml"
    write_incar_toml({"a": None, "b": 2}, always_include={"a"}, path=path)
    assert path.read_text() == 'a = "None"\nb = 2\n'


def test_write_incar_toml_update_skips_defaults(tmp_path: Path) -> None:
    """Update mode drops owned keys equal to their default."""
    path = tmp_path / "INCAR.toml"
    path.write_text("a = 1\n")
    write_incar_toml(
        {"a": 1, "b": 3},
        raw_existing={"a": 1},
        defaults={"a": 1, "b": 2},
        path=path,
    )
    assert path.read_text() == "b = 3\n"


def test_write_incar_toml_update_preserves_foreign_keys(tmp_path: Path) -> None:
    """Update mode keeps keys absent from defaults as user-owned."""
    path = tmp_path / "INCAR.toml"
    path.write_text('user = "keep"\n')
    write_incar_toml(
        {"a": 2},
        raw_existing={"user": "keep"},
        defaults={"a": 1},
        path=path,
    )
    assert path.read_text() == 'a = 2\nuser = "keep"\n'


def test_write_incar_toml_update_preserve_defaults(tmp_path: Path) -> None:
    """preserve_defaults keeps owned-default keys already present."""
    path = tmp_path / "INCAR.toml"
    path.write_text("a = 1\n")
    write_incar_toml(
        {"a": 1, "b": 3},
        raw_existing={"a": 1},
        defaults={"a": 1, "b": 2},
        path=path,
        preserve_defaults=True,
    )
    assert path.read_text() == "a = 1\nb = 3\n"


def test_write_incar_toml_update_noop_when_unchanged(tmp_path: Path) -> None:
    """When merged content matches the file, no write or backup occurs."""
    path = tmp_path / "INCAR.toml"
    path.write_text("b = 3\n")
    write_incar_toml(
        {"b": 3},
        raw_existing={"b": 3},
        defaults={"b": 2},
        path=path,
    )
    assert not (tmp_path / "INCAR.toml.old").exists()
    assert path.read_text() == "b = 3\n"


def test_write_incar_toml_update_backs_up_existing(tmp_path: Path) -> None:
    """A real write backs up the previous file to INCAR.toml.old."""
    path = tmp_path / "INCAR.toml"
    path.write_text("a = 1\n")
    write_incar_toml(
        {"a": 2},
        raw_existing={"a": 1},
        defaults={"a": 1},
        path=path,
    )
    backup = tmp_path / "INCAR.toml.old"
    assert backup.exists()
    assert backup.read_text() == "a = 1\n"
