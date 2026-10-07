"""Tests for IMDgroup.pymatgen.io.vasp.inputs."""

from __future__ import annotations

import pytest

from IMDgroup.pymatgen.io.vasp.inputs import Incar
import IMDgroup.pymatgen.io.vasp.inputs as inputs


# --- Incar.image_dir_names --------------------------------------------------


def test_image_dir_names_includes_ends() -> None:
    """IMAGES=N yields N+2 image directories, ends included."""
    assert Incar({"IMAGES": 3}).image_dir_names() == \
        ["00", "01", "02", "03", "04"]


def test_image_dir_names_excludes_ends() -> None:
    """include_ends=False drops the first and last directories."""
    assert Incar({"IMAGES": 3}).image_dir_names(include_ends=False) == \
        ["01", "02", "03"]


def test_image_dir_names_single_image() -> None:
    """IMAGES=1 produces three directories."""
    assert Incar({"IMAGES": 1}).image_dir_names() == ["00", "01", "02"]


def test_image_dir_names_unset_returns_none() -> None:
    """Without IMAGES the method returns None."""
    assert Incar({}).image_dir_names() is None


# --- Incar.get_recipe -------------------------------------------------------


def test_get_recipe_pbe() -> None:
    """The PBE functional maps to its GGA tag."""
    assert Incar.get_recipe("functional", "pbe") == {"GGA": "PE"}


def test_get_recipe_case_insensitive() -> None:
    """Functional names are matched case-insensitively."""
    assert Incar.get_recipe("functional", "PBE") == {"GGA": "PE"}
    assert Incar.get_recipe("functional", "vdW-DF")["GGA"] == "RE"


def test_get_recipe_defaults_are_none() -> None:
    """__defaults maps every key to None, the marker for removing it."""
    defaults = Incar.get_recipe("functional", "__defaults")
    assert defaults
    assert all(value is None for value in defaults.values())


def test_get_recipe_vdw_df() -> None:
    """A vdW functional returns its full parameter block."""
    recipe = Incar.get_recipe("functional", "vdw-df")
    assert recipe["GGA"] == "RE"
    assert recipe["LUSE_VDW"] is True
    assert recipe["AGGAC"] == 0.0


def test_get_recipe_unknown_functional_raises() -> None:
    """An unsupported functional name raises KeyError."""
    with pytest.raises(KeyError, match="unsupported functional"):
        Incar.get_recipe("functional", "nope")


def test_get_recipe_unknown_functional_message_lists_functionals() -> None:
    """The KeyError message lists functional names, not meta keys."""
    with pytest.raises(KeyError) as exc_info:
        Incar.get_recipe("functional", "nope")
    message = str(exc_info.value)
    assert "pbe" in message
    assert "__defaults" not in message
    assert "PMG-PARENT" not in message


def test_get_recipe_unknown_setup_raises() -> None:
    """An unsupported setup raises ValueError."""
    with pytest.raises(ValueError, match="Unknown setup"):
        Incar.get_recipe("bogus", "pbe")


# --- Incar.group_incars -----------------------------------------------------


def test_group_incars_groups_ignoring_fields() -> None:
    """INCARs differing only in ignored fields share a group."""
    incars = [
        Incar({"SYSTEM": "a", "ENCUT": 400, "NELM": 60}),
        Incar({"SYSTEM": "b", "ENCUT": 400, "NELM": 120}),
    ]
    _, groups = Incar.group_incars(incars)
    assert len(groups) == 1
    assert [incar["SYSTEM"] for incar in groups[0]] == ["a", "b"]


def test_group_incars_splits_on_material_field() -> None:
    """A difference in a non-ignored field splits groups."""
    incars = [
        Incar({"SYSTEM": "a", "ENCUT": 400}),
        Incar({"SYSTEM": "b", "ENCUT": 400}),
        Incar({"SYSTEM": "c", "ENCUT": 500}),
    ]
    _, groups = Incar.group_incars(incars)
    assert len(groups) == 2
    assert [incar["SYSTEM"] for incar in groups[0]] == ["a", "b"]
    assert [incar["SYSTEM"] for incar in groups[1]] == ["c"]


def test_group_incars_common_incar() -> None:
    """common_incar holds fields shared across groups minus ignored ones."""
    incars = [
        Incar({"SYSTEM": "a", "ENCUT": 400, "EDIFF": 1e-5, "ISIF": 2}),
        Incar({"SYSTEM": "b", "ENCUT": 400, "EDIFF": 1e-5, "ISIF": 2}),
        Incar({"SYSTEM": "c", "ENCUT": 400, "EDIFF": 1e-5, "ISIF": 3}),
    ]
    common, _ = Incar.group_incars(incars)
    assert dict(common) == {"ENCUT": 400, "EDIFF": 1e-5}


# --- _load_yaml_config ------------------------------------------------------


def test_load_yaml_config_functionals() -> None:
    """The functionals config loads with its expected functional keys."""
    config = inputs._load_yaml_config("functionals")
    assert config["pbe"] == {"GGA": "PE"}
    assert "vdw-df" in config


def test_load_yaml_config_parent_merge(tmp_path, monkeypatch) -> None:
    """PARENT configs are merged: child scalars win, parent nested wins."""
    monkeypatch.setattr(inputs, "MODULE_DIR", str(tmp_path))
    (tmp_path / "parent.yaml").write_text(
        "a: 1\nb: 2\nnested:\n  x: 1\n  y: 2\n")
    (tmp_path / "child.yaml").write_text(
        "PARENT: parent\nb: 3\nnested:\n  y: 4\n")
    config = inputs._load_yaml_config("child")
    assert config == {
        "PARENT": "parent",
        "a": 1,
        "b": 3,
        "nested": {"x": 1, "y": 2},
    }
