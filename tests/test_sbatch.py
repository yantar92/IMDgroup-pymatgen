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


"""Tests for IMDgroup.gorun.sbatch."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import IMDgroup.gorun.sbatch as sbatch_mod

from IMDgroup.gorun.sbatch import (
    current_server,
    get_default_job_name,
    get_sbatch_args,
    get_user_sbatch_args,
    load_config,
    parse_time_limit,
    resolve_server,
)


def _args(nodes=None, time=None) -> SimpleNamespace:
    """Build a minimal script-args namespace."""
    return SimpleNamespace(number_of_nodes=nodes, time_limit=time)


def _config_args(config_path=None) -> SimpleNamespace:
    """Build a minimal args namespace for config resolution."""
    return SimpleNamespace(config=config_path)


# --- load_config ------------------------------------------------------------


def test_load_config_returns_none_when_missing(monkeypatch, tmp_path) -> None:
    """No config file on the search path returns None."""
    monkeypatch.setattr(
        sbatch_mod, "_candidate_config_paths", lambda: [str(tmp_path / "nope.toml")]
    )
    assert load_config(_config_args()) is None


def test_load_config_explicit_missing_raises(monkeypatch, tmp_path) -> None:
    """An explicit --config that is missing raises FileNotFoundError."""
    missing = str(tmp_path / "missing.toml")
    with pytest.raises(FileNotFoundError):
        load_config(_config_args(config_path=missing))


def test_load_config_reads_file(monkeypatch, tmp_path) -> None:
    """A found config file is parsed and returned."""
    cfg = tmp_path / "gorun.toml"
    cfg.write_text("[cluster.names]\ntestserver = ['testhost']\n")
    monkeypatch.setattr(sbatch_mod, "_candidate_config_paths", lambda: [str(cfg)])
    assert load_config(_config_args()) == {
        "cluster": {"names": {"testserver": ["testhost"]}}
    }


def test_read_example_config_returns_bundled_toml() -> None:
    """The bundled example config is readable and has the cluster table."""
    content = sbatch_mod.read_example_config()
    assert "[cluster.names]" in content


# --- _candidate_config_paths / missing_config_message ------------------------


def test_candidate_config_paths_with_imdgroup(monkeypatch) -> None:
    """An IMDGroup checkout path takes precedence over the XDG location."""
    monkeypatch.setenv("IMDGroup", "/opt/imdgroup")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg")
    paths = sbatch_mod._candidate_config_paths()
    assert paths[0] == "/opt/imdgroup/dist/etc/gorun.toml"
    assert paths[1] == "/xdg/gorun/gorun.toml"


def test_candidate_config_paths_with_xdg(monkeypatch) -> None:
    """XDG_CONFIG_HOME overrides the ~/.config default."""
    monkeypatch.delenv("IMDGroup", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg")
    paths = sbatch_mod._candidate_config_paths()
    assert paths == ["/xdg/gorun/gorun.toml"]


def test_candidate_config_paths_defaults_to_home(monkeypatch) -> None:
    """Without IMDGroup or XDG_CONFIG_HOME, ~/.config is used."""
    monkeypatch.delenv("IMDGroup", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    paths = sbatch_mod._candidate_config_paths()
    assert paths == [os.path.join(os.path.expanduser("~"), ".config",
                                  "gorun", "gorun.toml")]


def test_missing_config_message_includes_guidance(monkeypatch) -> None:
    """The public message lists searched paths and setup guidance."""
    monkeypatch.setattr(
        sbatch_mod, "_candidate_config_paths", lambda: ["/a", "/b"])
    msg = sbatch_mod.missing_config_message()
    assert "No gorun configuration found" in msg
    assert "/a" in msg and "/b" in msg
    assert "gorun --example-config" in msg


# --- resolve_server ---------------------------------------------------------


def test_resolve_server_matching(monkeypatch, fake_config) -> None:
    """A matching host returns the server name."""
    monkeypatch.setenv("CLUSTER_NAME", "testhost")
    assert resolve_server(fake_config) == "testserver"


def test_resolve_server_unmatched_host(monkeypatch, fake_config) -> None:
    """A host absent from the names table returns None."""
    monkeypatch.setenv("CLUSTER_NAME", "unknownhost")
    assert resolve_server(fake_config) is None


def test_resolve_server_no_cluster_names(monkeypatch) -> None:
    """A config without [cluster.names] returns None."""
    assert resolve_server({"testserver": {}}) is None


def test_resolve_server_missing_section(monkeypatch) -> None:
    """A mapped name with no section in the config returns None."""
    cfg = {"cluster": {"names": {"other": ["host"]}}}
    monkeypatch.setenv("CLUSTER_NAME", "host")
    assert resolve_server(cfg) is None


# --- parse_time_limit -------------------------------------------------------


def test_parse_time_limit_human_readable() -> None:
    """Human-readable durations normalize to canonical HH:MM:SS."""
    assert parse_time_limit("1h") == "01:00:00"
    assert parse_time_limit("1d") == "24:00:00"
    assert parse_time_limit("30m") == "00:30:00"
    assert parse_time_limit("1h30m") == "01:30:00"
    assert parse_time_limit("1.5h") == "01:30:00"


def test_parse_time_limit_slurm_formats_pass_through() -> None:
    """Slurm colon formats normalize; bare minutes and days-hours pass through."""
    assert parse_time_limit("48:00:00") == "48:00:00"
    assert parse_time_limit("120") == "120"
    assert parse_time_limit("2-12") == "2-12"


def test_parse_time_limit_invalid_duration_raises() -> None:
    """A duration-like value that fails to parse raises ValueError."""
    with pytest.raises(ValueError, match="1h30"):
        parse_time_limit("1h30")


# --- get_user_sbatch_args ---------------------------------------------------


def test_get_user_sbatch_args_empty() -> None:
    """No explicit nodes/time yields an empty dict."""
    assert get_user_sbatch_args(_args()) == {}


def test_get_user_sbatch_args_nodes_only() -> None:
    """An explicit node count maps to a string ``nodes`` key."""
    assert get_user_sbatch_args(_args(nodes=2)) == {"nodes": "2"}


def test_get_user_sbatch_args_time_only() -> None:
    """An explicit time limit maps to a string ``time`` key."""
    assert get_user_sbatch_args(_args(time="02:00:00")) == {"time": "02:00:00"}


def test_get_user_sbatch_args_normalizes_human_time() -> None:
    """A human-readable time limit is normalized in the programmatic path."""
    assert get_user_sbatch_args(_args(time="1h")) == {"time": "01:00:00"}


def test_get_user_sbatch_args_both() -> None:
    """Both values are mapped to their sbatch keys."""
    assert get_user_sbatch_args(_args(nodes=2, time="02:00:00")) == {
        "nodes": "2",
        "time": "02:00:00",
    }


# --- get_default_job_name ---------------------------------------------------


def test_get_default_job_name_falls_back_to_directory(monkeypatch, tmp_path: Path) -> None:
    """Without an INCAR the job name is the directory basename."""
    monkeypatch.chdir(tmp_path)
    assert get_default_job_name() == {"job-name": os.path.basename(str(tmp_path))}


def test_get_default_job_name_reads_system_from_incar(monkeypatch, tmp_path: Path) -> None:
    """A real INCAR supplies the SYSTEM value as the job name."""
    (tmp_path / "INCAR").write_text("SYSTEM = mytest\n")
    monkeypatch.chdir(tmp_path)
    assert get_default_job_name() == {"job-name": "mytest"}


# --- get_sbatch_args --------------------------------------------------------


def test_get_sbatch_args_precedence(monkeypatch, fake_config) -> None:
    """Later sources override earlier ones in the merge chain."""
    monkeypatch.setattr(sbatch_mod, "get_default_job_name", lambda: {"job-name": "myname"})
    script_args = _args(nodes=3, time="02:00:00")
    result = get_sbatch_args(
        script_args, fake_config, "testserver", "cpu", adapter_name="vasp"
    )
    assert result == {
        "account": "default",   # server_defaults
        "nodes": "3",           # user args override adapter defaults
        "time": "02:00:00",     # user args override queue
        "job-name": "myname",   # job name
        "partition": "cpu",     # always the queue
    }


def test_get_sbatch_args_no_adapter_defaults(monkeypatch, fake_config) -> None:
    """Without an adapter name, only server defaults and queue apply."""
    monkeypatch.setattr(sbatch_mod, "get_default_job_name", lambda: {"job-name": "myname"})
    result = get_sbatch_args(
        _args(), fake_config, "testserver", "cpu", adapter_name=None
    )
    assert result == {
        "account": "default",
        "time": "01:00:00",
        "job-name": "myname",
        "partition": "cpu",
    }


# --- current_server ---------------------------------------------------------


def test_current_server_uses_cluster_name_override(monkeypatch, fake_config) -> None:
    """CLUSTER_NAME maps through the cluster names table to a server."""
    monkeypatch.setenv("CLUSTER_NAME", "testhost")
    assert current_server(fake_config) == "testserver"


def test_current_server_unknown_host_returns_uname(monkeypatch, fake_config) -> None:
    """A host absent from the names table is returned unchanged."""
    monkeypatch.setenv("CLUSTER_NAME", "unknownhost")
    assert current_server(fake_config) == "unknownhost"


def test_current_server_uname_fallback(monkeypatch, fake_config) -> None:
    """Without CLUSTER_NAME, uname -n is consulted and matched."""
    monkeypatch.delenv("CLUSTER_NAME", raising=False)
    monkeypatch.setattr(sbatch_mod, "barf_if_no_cmd", lambda cmd: None)
    monkeypatch.setattr(
        sbatch_mod.subprocess,
        "check_output",
        lambda cmd, **kwargs: b"testhost\n",
    )
    assert current_server(fake_config) == "testserver"
