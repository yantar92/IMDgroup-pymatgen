# MIT License
#
# Copyright (c) 2024-2026 Inverse Materials Design Group
#
# Author: Ihor Radchenko <yantar92@posteo.net>
#
# This file is a part of MatRiFFlow package
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


"""Tests for matrifflow.gorun.cleanVASP."""

from __future__ import annotations

import gzip
import warnings
from pathlib import Path

import pytest
from pymatgen.io.vasp.inputs import BadIncarWarning

import matrifflow.gorun.cleanVASP as clean_mod

from matrifflow.gorun.cleanVASP import (
    check_incar,
    clean_vasp_input,
    clean_vasp_inputs,
    clear_useless_vasp_files,
    clear_vasp_outputs,
    contcar_to_poscar,
    directory_contains_vasp_outputp,
    directory_converged_p,
    generate_potcar,
    mdp,
    nebp,
    potcar_symbols_map,
    prepare_vasp_dir,
    put_vdw_kernel,
)


# --- directory_contains_vasp_outputp ----------------------------------------


def test_directory_contains_vasp_outputp_empty_outcar_is_false(tmp_path: Path) -> None:
    """A zero-size OUTCAR is not considered output."""
    (tmp_path / "OUTCAR").touch()
    assert directory_contains_vasp_outputp(str(tmp_path)) is False


def test_directory_contains_vasp_outputp_nonempty_outcar_is_true(tmp_path: Path) -> None:
    """A non-empty OUTCAR marks the directory as containing output."""
    (tmp_path / "OUTCAR").write_text("data")
    assert directory_contains_vasp_outputp(str(tmp_path)) is True


def test_directory_contains_vasp_outputp_neb_recursion(
    monkeypatch, tmp_path: Path
) -> None:
    """With a NEB layout, output in a numbered subdir counts."""
    (tmp_path / "INCAR").write_text("IMAGES = 3\n")
    sub = tmp_path / "00"
    sub.mkdir()
    (sub / "OUTCAR").write_text("data")
    assert directory_contains_vasp_outputp(str(tmp_path)) is True


def test_directory_contains_vasp_outputp_compressed_outcar(tmp_path: Path) -> None:
    """A non-empty gzipped OUTCAR is treated as output."""
    with gzip.open(tmp_path / "OUTCAR.gz", "wb") as f_out:
        f_out.write(b"data")
    assert directory_contains_vasp_outputp(str(tmp_path)) is True


def test_directory_contains_vasp_outputp_compressed_vasprun(tmp_path: Path) -> None:
    """A gzipped vasprun.xml counts even without an OUTCAR."""
    with gzip.open(tmp_path / "vasprun.xml.gz", "wb") as f_out:
        f_out.write(b"<modeling/>")
    assert directory_contains_vasp_outputp(str(tmp_path)) is True


# --- nebp / mdp -------------------------------------------------------------


def test_nebp_true_with_images(tmp_path: Path) -> None:
    """An INCAR with IMAGES marks a NEB run."""
    (tmp_path / "INCAR").write_text("IMAGES = 3\n")
    assert nebp(str(tmp_path)) is True


def test_nebp_false_without_images(tmp_path: Path) -> None:
    """An INCAR without IMAGES is not a NEB run."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    assert nebp(str(tmp_path)) is False


def test_nebp_false_when_incar_missing(tmp_path: Path) -> None:
    """No INCAR means not a NEB run."""
    assert nebp(str(tmp_path)) is False


def test_mdp_true_with_ibrion_zero(tmp_path: Path) -> None:
    """IBRION=0 marks an MD run."""
    (tmp_path / "INCAR").write_text("IBRION = 0\n")
    assert mdp(str(tmp_path)) is True


def test_mdp_false_with_ibrion_nonzero(tmp_path: Path) -> None:
    """A nonzero IBRION is not an MD run."""
    (tmp_path / "INCAR").write_text("IBRION = 2\n")
    assert mdp(str(tmp_path)) is False


# --- clear_useless_vasp_files / clear_vasp_outputs --------------------------


def test_clear_useless_vasp_files_removes_only_empty(tmp_path: Path) -> None:
    """Only zero-size cleanup targets are removed."""
    (tmp_path / "CHG").touch()
    (tmp_path / "WAVECAR").write_text("data")
    clear_useless_vasp_files(str(tmp_path))
    assert not (tmp_path / "CHG").exists()
    assert (tmp_path / "WAVECAR").exists()


def test_clear_vasp_outputs_removes_known_outputs(tmp_path: Path) -> None:
    """All known VASP output files are removed; unrelated files stay."""
    for name in ("CHG", "OUTCAR", "vasprun.xml", "XDATCAR"):
        (tmp_path / name).touch()
    (tmp_path / "POSCAR").touch()
    clear_vasp_outputs(str(tmp_path))
    for name in ("CHG", "OUTCAR", "vasprun.xml", "XDATCAR"):
        assert not (tmp_path / name).exists()
    assert (tmp_path / "POSCAR").exists()


def test_clear_vasp_outputs_removes_compressed_variants(tmp_path: Path) -> None:
    """Compressed OUTCAR/vasprun.xml are removed too; POSCAR stays."""
    with gzip.open(tmp_path / "OUTCAR.gz", "wb") as f_out:
        f_out.write(b"data")
    with gzip.open(tmp_path / "vasprun.xml.gz", "wb") as f_out:
        f_out.write(b"<modeling/>")
    (tmp_path / "POSCAR").touch()
    clear_vasp_outputs(str(tmp_path))
    assert not (tmp_path / "OUTCAR.gz").exists()
    assert not (tmp_path / "vasprun.xml.gz").exists()
    assert (tmp_path / "POSCAR").exists()


# --- contcar_to_poscar ------------------------------------------------------


def test_contcar_to_poscar_copies(tmp_path: Path) -> None:
    """A non-empty CONTCAR is copied over POSCAR."""
    (tmp_path / "CONTCAR").write_text("relaxed")
    (tmp_path / "POSCAR").write_text("initial")
    contcar_to_poscar(str(tmp_path))
    assert (tmp_path / "POSCAR").read_text() == "relaxed"


def test_contcar_to_poscar_skips_when_absent(tmp_path: Path) -> None:
    """Without a CONTCAR, POSCAR is left untouched."""
    (tmp_path / "POSCAR").write_text("initial")
    contcar_to_poscar(str(tmp_path))
    assert (tmp_path / "POSCAR").read_text() == "initial"


# --- clean_vasp_input / clean_vasp_inputs -----------------------------------


def test_clean_vasp_input_normalizes_crlf_and_bom(tmp_path: Path) -> None:
    """CRLF line endings and a UTF-8 BOM are stripped."""
    p = tmp_path / "INCAR"
    p.write_bytes(b"\xEF\xBB\xBFENCUT = 400\r\nEDIFF = 1e-6\r\n")
    clean_vasp_input(str(p))
    assert p.read_text(encoding="utf-8") == "ENCUT = 400\nEDIFF = 1e-6\n"


def test_clean_vasp_input_removes_tab_only_lines(tmp_path: Path) -> None:
    """Lines containing only tabs are removed."""
    p = tmp_path / "INCAR"
    p.write_text("ENCUT = 400\n\t\t\nEDIFF = 1e-6\n")
    clean_vasp_input(str(p))
    assert p.read_text() == "ENCUT = 400\n\nEDIFF = 1e-6\n"


def test_clean_vasp_inputs_loops_over_input_files(tmp_path: Path) -> None:
    """POSCAR/INCAR/KPOINTS are each cleaned; absent files are skipped."""
    (tmp_path / "POSCAR").write_bytes(b"\xEF\xBB\xBFline\r\n")
    (tmp_path / "INCAR").write_text("ENCUT = 400\n")
    clean_vasp_inputs(str(tmp_path))
    assert (tmp_path / "POSCAR").read_text() == "line\n"
    assert (tmp_path / "INCAR").read_text() == "ENCUT = 400\n"


# --- check_incar ------------------------------------------------------------


def test_check_incar_valid(tmp_path: Path) -> None:
    """A valid INCAR passes without warning."""
    (tmp_path / "INCAR").write_text("ENCUT = 400\nEDIFF = 1e-6\n")
    check_incar(str(tmp_path))


def test_check_incar_invalid_value_warns(tmp_path: Path) -> None:
    """A recognized tag with an invalid value raises BadIncarWarning."""
    (tmp_path / "INCAR").write_text("ISPIN = 3\n")
    with pytest.warns(BadIncarWarning):
        check_incar(str(tmp_path))


def test_check_incar_unknown_tag_warns(tmp_path: Path) -> None:
    """An unrecognized tag raises BadIncarWarning."""
    (tmp_path / "INCAR").write_text("FOOBAR = 1\n")
    with pytest.warns(BadIncarWarning):
        check_incar(str(tmp_path))


# --- potcar_symbols_map -----------------------------------------------------


def test_potcar_symbols_map_defaults(tmp_path: Path) -> None:
    """Without INCAR.toml, the recommended full names are returned."""
    symbols = potcar_symbols_map(str(tmp_path))
    assert symbols["W"] == "W_sv"
    assert symbols["Li"] == "Li_sv"


def test_potcar_symbols_map_from_toml(tmp_path: Path) -> None:
    """[POTCAR] overrides map element symbols to full POTCAR names."""
    (tmp_path / "INCAR.toml").write_text(
        '[POTCAR]\nW = "W_sv"\nLi = "Li_sv"\nC = "C"\n'
    )
    symbols = potcar_symbols_map(str(tmp_path))
    assert symbols["W"] == "W_sv"
    assert symbols["Li"] == "Li_sv"
    assert symbols["C"] == "C"


def test_potcar_symbols_map_override_default_w(tmp_path: Path) -> None:
    """A user override for W wins over the built-in fix."""
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nW = "W_pv"\n')
    symbols = potcar_symbols_map(str(tmp_path))
    assert symbols["W"] == "W_pv"


def test_potcar_symbols_map_non_table_raises(tmp_path: Path) -> None:
    """A non-table POTCAR value raises ValueError."""
    (tmp_path / "INCAR.toml").write_text('POTCAR = "W_sv"\n')
    with pytest.raises(ValueError):
        potcar_symbols_map(str(tmp_path))


def test_potcar_symbols_map_bad_prefix_raises(tmp_path: Path) -> None:
    """A value that is not a valid potential name raises ValueError."""
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nW = "Wv_sv"\n')
    with pytest.raises(ValueError):
        potcar_symbols_map(str(tmp_path))


def test_potcar_symbols_map_wrong_element_raises(tmp_path: Path) -> None:
    """A value not starting with the element symbol raises ValueError."""
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nW = "Li_sv"\n')
    with pytest.raises(ValueError):
        potcar_symbols_map(str(tmp_path))


def test_potcar_symbols_map_non_string_value_raises(tmp_path: Path) -> None:
    """A non-string value in [POTCAR] raises ValueError."""
    (tmp_path / "INCAR.toml").write_text('[POTCAR]\nW = 123\n')
    with pytest.raises(ValueError):
        potcar_symbols_map(str(tmp_path))


# --- directory_contains_vasp_outputp: NEB recursion branches ----------------


def test_directory_contains_vasp_outputp_neb_no_numeric_subdir(tmp_path: Path) -> None:
    """A NEB dir whose subdirs are all non-numeric has no output."""
    (tmp_path / "INCAR").write_text("IMAGES = 3\n")
    (tmp_path / "foo").mkdir()
    assert directory_contains_vasp_outputp(str(tmp_path)) is False


def test_directory_contains_vasp_outputp_neb_subdir_no_output(tmp_path: Path) -> None:
    """A numeric NEB subdir without output recurses and finds nothing."""
    (tmp_path / "INCAR").write_text("IMAGES = 3\n")
    (tmp_path / "00").mkdir()
    assert directory_contains_vasp_outputp(str(tmp_path)) is False


# --- directory_converged_p --------------------------------------------------


def test_directory_converged_p_no_output(tmp_path: Path) -> None:
    """Without VASP output, the directory is not converged."""
    assert directory_converged_p(str(tmp_path)) is False


@pytest.mark.parametrize(
    ("electronic", "ionic", "manual", "expected"),
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_directory_converged_p_flags(
    monkeypatch, tmp_path: Path, electronic, ionic, manual, expected
) -> None:
    """directory_converged_p ANDs the three convergence flags."""
    monkeypatch.setattr(
        clean_mod, "directory_contains_vasp_outputp", lambda path: True
    )

    class _FakeVaspDir:
        def __init__(self, path):
            self.converged_electronic = electronic
            self.converged_ionic = ionic
            self.converged_manual = manual

    monkeypatch.setattr(clean_mod, "MRFVaspDir", _FakeVaspDir)
    assert directory_converged_p(str(tmp_path)) is expected


# --- mdp / check_incar: missing INCAR ---------------------------------------


def test_mdp_false_when_incar_missing(tmp_path: Path) -> None:
    """No INCAR means not an MD run."""
    assert mdp(str(tmp_path)) is False


def test_check_incar_missing_incar(tmp_path: Path) -> None:
    """No INCAR: check_incar is a no-op."""
    check_incar(str(tmp_path))


# --- put_vdw_kernel ---------------------------------------------------------


def test_put_vdw_kernel_copies_from_vasp_path(monkeypatch, tmp_path: Path) -> None:
    """A kernel at $VASP_PATH/vdw_kernel.bindat is copied into PATH."""
    vasp = tmp_path / "vasp"
    vasp.mkdir()
    (vasp / "vdw_kernel.bindat").write_text("kernel")
    monkeypatch.setenv("VASP_PATH", str(vasp))
    put_vdw_kernel(str(tmp_path))
    assert (tmp_path / "vdw_kernel.bindat").read_text() == "kernel"


def test_put_vdw_kernel_falls_back_to_testsuite(monkeypatch, tmp_path: Path) -> None:
    """A missing top-level kernel falls back to the testsuite path."""
    vasp = tmp_path / "vasp"
    suite = vasp / "testsuite" / "tests" / "CuC_vdW"
    suite.mkdir(parents=True)
    (suite / "vdw_kernel.bindat").write_text("kernel2")
    monkeypatch.setenv("VASP_PATH", str(vasp))
    put_vdw_kernel(str(tmp_path))
    assert (tmp_path / "vdw_kernel.bindat").read_text() == "kernel2"


def test_put_vdw_kernel_warns_when_missing(monkeypatch, tmp_path: Path) -> None:
    """No kernel anywhere emits a warning and copies nothing."""
    vasp = tmp_path / "vasp"
    vasp.mkdir()
    monkeypatch.setenv("VASP_PATH", str(vasp))
    with pytest.warns(UserWarning, match="Cannot find vdw_kernel.bindat"):
        put_vdw_kernel(str(tmp_path))
    assert not (tmp_path / "vdw_kernel.bindat").exists()


# --- generate_potcar --------------------------------------------------------

# generate_potcar wraps pymatgen's Poscar/Potcar.  We stand in for those so
# the tests exercise our own logic (POSCAR selection, symbol mapping, the
# keep-existing/update decisions) without needing a real POTCAR library.


class _FakePoscar:
    """Stand-in for pymatgen Poscar reporting a single Li site."""

    site_symbols = ["Li"]

    def __init__(self, site_symbols):
        self.site_symbols = site_symbols

    @classmethod
    def from_file(cls, path, check_for_potcar=False):
        return cls(cls.site_symbols)


class _FakePotcar:
    """Stand-in for pymatgen Potcar using a parseable SYMS header."""

    pad_size = 50

    def __init__(self, symbols, functional=None):
        self.symbols = list(symbols)
        self.functional = functional

    @property
    def _header(self):
        return "SYMS=" + ",".join(self.symbols) + "\n"

    def write_file(self, path):
        Path(path).write_text(self._header + "x" * self.pad_size)

    @classmethod
    def from_file(cls, path):
        line = Path(path).read_text().splitlines()[0]
        syms = line.split("=", 1)[1].split(",")
        return cls(syms)


def _install_potcar_fakes(
    monkeypatch, *, symbol_map=None, is_neb=False, pad_size=50
) -> None:
    """Replace nebp/potcar_symbols_map/Poscar/Potcar with deterministic fakes."""
    monkeypatch.setattr(clean_mod, "nebp", lambda path: is_neb)
    monkeypatch.setattr(clean_mod, "potcar_symbols_map", lambda path: symbol_map or {})
    monkeypatch.setattr(clean_mod, "POTCAR_FUNCTIONAL_DEFAULT", "PBE")
    monkeypatch.setattr(clean_mod, "Poscar", _FakePoscar)
    monkeypatch.setattr(_FakePotcar, "pad_size", pad_size)
    monkeypatch.setattr(clean_mod, "Potcar", _FakePotcar)


def test_generate_potcar_no_poscar(tmp_path: Path) -> None:
    """No POSCAR anywhere: generate_potcar returns without writing anything."""
    generate_potcar(str(tmp_path))
    assert not (tmp_path / "POTCAR").exists()


def test_generate_potcar_neb_fallback(monkeypatch, tmp_path: Path) -> None:
    """A NEB layout uses 00/POSCAR when there is no top-level POSCAR."""
    (tmp_path / "00").mkdir()
    (tmp_path / "00" / "POSCAR").write_text("neb poscar")

    seen = []

    class _RecordingPoscar(_FakePoscar):
        @classmethod
        def from_file(cls, path, check_for_potcar=False):
            seen.append(str(path))
            return cls(cls.site_symbols)

    _install_potcar_fakes(monkeypatch, symbol_map={"Li": "Li_sv"}, is_neb=True)
    monkeypatch.setattr(clean_mod, "Poscar", _RecordingPoscar)

    generate_potcar(str(tmp_path))
    assert seen == [str(tmp_path / "00" / "POSCAR")]
    assert (tmp_path / "POTCAR").exists()


def test_generate_potcar_keep_existing(monkeypatch, tmp_path: Path) -> None:
    """keep_existing=True leaves an existing POTCAR untouched."""
    (tmp_path / "POSCAR").write_text("poscar")
    (tmp_path / "POTCAR").write_text("existing")
    monkeypatch.setattr(clean_mod, "nebp", lambda path: False)

    class _BoomPoscar:
        @classmethod
        def from_file(cls, *args, **kwargs):
            raise AssertionError("Poscar.from_file should not be called")

    monkeypatch.setattr(clean_mod, "Poscar", _BoomPoscar)

    generate_potcar(str(tmp_path), keep_existing=True)
    assert (tmp_path / "POTCAR").read_text() == "existing"


def test_generate_potcar_fresh(monkeypatch, tmp_path: Path) -> None:
    """Without an existing POTCAR, the mapped symbol is written fresh."""
    (tmp_path / "POSCAR").write_text("poscar")
    _install_potcar_fakes(monkeypatch, symbol_map={"Li": "Li_sv"})
    generate_potcar(str(tmp_path))
    assert (tmp_path / "POTCAR").read_text().startswith("SYMS=Li_sv\n")
    assert not (tmp_path / "POTCAR.old").exists()


def test_generate_potcar_uses_element_when_no_mapping(monkeypatch, tmp_path: Path) -> None:
    """An element absent from the symbol map is written as-is."""
    (tmp_path / "POSCAR").write_text("poscar")
    _install_potcar_fakes(monkeypatch, symbol_map={"W": "W_sv"})
    generate_potcar(str(tmp_path))
    assert (tmp_path / "POTCAR").read_text().startswith("SYMS=Li\n")


def test_generate_potcar_update_changed_symbols(monkeypatch, tmp_path: Path) -> None:
    """A POTCAR whose symbols change is updated and warns."""
    (tmp_path / "POSCAR").write_text("poscar")
    (tmp_path / "POTCAR").write_text("SYMS=Li\n" + "x" * 50)
    _install_potcar_fakes(monkeypatch, symbol_map={"Li": "Li_sv"})
    with pytest.warns(UserWarning, match="Changed pseudopotentials"):
        generate_potcar(str(tmp_path))
    assert (tmp_path / "POTCAR.old").exists()


def test_generate_potcar_unchanged(monkeypatch, tmp_path: Path) -> None:
    """Identical symbols and size remove the .old backup without warning."""
    (tmp_path / "POSCAR").write_text("poscar")
    (tmp_path / "POTCAR").write_text("SYMS=Li_sv\n" + "x" * 50)
    _install_potcar_fakes(monkeypatch, symbol_map={"Li": "Li_sv"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate_potcar(str(tmp_path))
    assert not (tmp_path / "POTCAR.old").exists()


def test_generate_potcar_update_same_symbols(monkeypatch, tmp_path: Path) -> None:
    """A size change with identical symbols keeps the backup, no warning."""
    (tmp_path / "POSCAR").write_text("poscar")
    (tmp_path / "POTCAR").write_text("SYMS=Li_sv\n" + "x" * 10)
    _install_potcar_fakes(monkeypatch, symbol_map={"Li": "Li_sv"}, pad_size=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate_potcar(str(tmp_path))
    assert (tmp_path / "POTCAR.old").exists()


# --- prepare_vasp_dir -------------------------------------------------------


def test_prepare_vasp_dir_calls_sequence(monkeypatch, tmp_path: Path) -> None:
    """Each preparation step runs in order, with generate_potcar first."""
    calls: list[str] = []
    for fn in (
        "check_incar",
        "contcar_to_poscar",
        "clean_vasp_inputs",
        "put_vdw_kernel",
        "clear_vasp_outputs",
    ):
        monkeypatch.setattr(clean_mod, fn, lambda path, fn=fn: calls.append(fn))
    generated = []
    monkeypatch.setattr(
        clean_mod, "generate_potcar", lambda path, keep: generated.append(keep)
    )
    prepare_vasp_dir(str(tmp_path))
    assert calls == [
        "check_incar",
        "contcar_to_poscar",
        "clean_vasp_inputs",
        "put_vdw_kernel",
        "clear_vasp_outputs",
    ]
    assert generated == [False]


def test_prepare_vasp_dir_keep_flags(monkeypatch, tmp_path: Path) -> None:
    """keep_poscar skips contcar_to_poscar; keep_potcar is forwarded."""
    calls: list[str] = []
    for fn in (
        "check_incar",
        "contcar_to_poscar",
        "clean_vasp_inputs",
        "put_vdw_kernel",
        "clear_vasp_outputs",
    ):
        monkeypatch.setattr(clean_mod, fn, lambda path, fn=fn: calls.append(fn))
    generated = []
    monkeypatch.setattr(
        clean_mod, "generate_potcar", lambda path, keep: generated.append(keep)
    )
    prepare_vasp_dir(str(tmp_path), keep_potcar=True, keep_poscar=True)
    assert "contcar_to_poscar" not in calls
    assert generated == [True]
