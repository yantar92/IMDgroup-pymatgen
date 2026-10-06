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

"""INCAR.toml: shared parameter-file I/O between pymatgen and gorun.

The file is the metadata analogue of VASP's ``INCAR``: it records
parameter choices that are not fully captured by the VASP input files
themselves (for example the POTCAR release and the k-point grid
density recipe).  ``gorun`` reads the ``[POTCAR]`` section to
reproduce pseudopotentials; this module provides the matching reader
and writer.

The file uses TOML.  Both the reader and writer work on the nested
dictionary shape produced by :func:`tomllib.load`: TOML ``[section]``
tables are dict values keyed by the section name, and top-level
scalar pairs are plain key-value entries.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

logger = logging.getLogger(__name__)


INCAR_TOML_PATH = Path('INCAR.toml')

_MISSING = object()


def read_incar_toml(path: Path = INCAR_TOML_PATH) -> dict[str, object]:
    """Read *path* and return its nested TOML content.

    Returns:
        dict[str, object]: The parsed content (``[section]`` tables as
        dict values), or an empty dict when the file does not exist.
    """
    if not path.is_file():
        return {}
    with open(path, 'rb') as fh:
        return tomllib.load(fh)


def _toml_format_value(value: object) -> str:
    """Format a scalar Python value for TOML output."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    return f'"{value}"'


def _merge_toml(
        values: dict[str, object],
        raw_existing: dict[str, object],
        defaults: dict[str, object],
        *,
        is_bootstrap: bool,
        always_include: set[str],
        preserve_defaults: bool) -> dict[str, object]:
    """Merge owned values over foreign keys, sparse against defaults.

    Recurses into dict-valued entries (TOML ``[section]`` tables).  A
    key is "owned" when it appears in *defaults* at the same nesting
    level; owned keys are written only when their value differs from
    the default (or when bootstrapping).  Keys present in
    *raw_existing* but not owned are preserved verbatim.

    Args:
        values: The resolved parameter values.
        raw_existing: Previously-read content, used for foreign-key
            preservation.
        defaults: Hardcoded defaults (same nested shape).
        is_bootstrap: When True, write all owned values (to surface the
            full parameter set).
        always_include: Top-level keys written even when equal to the
            default or ``None``.
        preserve_defaults: When True, keep owned keys whose value equals
            the default but were already present in *raw_existing*.

    Returns:
        dict[str, object]: The merged nested content.
    """
    merged: dict[str, object] = {}
    for key in sorted(set(values) | set(raw_existing)):
        value = values.get(key, _MISSING)
        raw_val = raw_existing.get(key, _MISSING)
        default = defaults.get(key, _MISSING)

        # Recurse into sections.
        if isinstance(value, dict):
            sub_raw = raw_val if isinstance(raw_val, dict) else {}
            sub_defaults = default if isinstance(default, dict) else {}
            sub_merged = _merge_toml(
                value, sub_raw, sub_defaults,
                is_bootstrap=is_bootstrap,
                always_include=always_include,
                preserve_defaults=preserve_defaults,
            )
            if sub_merged:
                merged[key] = sub_merged
            continue

        if value is _MISSING:
            # Present in raw_existing but not in values: only foreign
            # keys (absent from defaults) survive.
            if default is _MISSING and raw_val is not _MISSING:
                merged[key] = raw_val
            continue

        if key in always_include:
            merged[key] = value
        elif value is None:
            continue
        elif default is not _MISSING:
            # Owned: write when non-default, when bootstrapping, or when
            # explicitly preserving an already-present key.
            if is_bootstrap or value != default:
                merged[key] = value
            elif preserve_defaults and raw_val is not _MISSING:
                merged[key] = value
        else:
            # No default defined: write the value as-is.
            merged[key] = value

    return merged


def _format_toml(values: dict[str, object]) -> str:
    """Render a nested dict as TOML text.

    Dict values become ``[section]`` tables; scalars become top-level
    pairs.  One level of section nesting is supported.
    """
    lines: list[str] = []
    for key in sorted(values):
        val = values[key]
        if isinstance(val, dict):
            if lines:
                lines.append('')
            lines.append(f'[{key}]')
            for sub_key in sorted(val):
                lines.append(f'{sub_key} = {_toml_format_value(val[sub_key])}')
        else:
            lines.append(f'{key} = {_toml_format_value(val)}')
    return '\n'.join(lines)


def write_incar_toml(
        values: dict[str, object],
        *,
        raw_existing: dict[str, object] | None = None,
        defaults: dict[str, object] | None = None,
        always_include: set[str] | None = None,
        path: Path = INCAR_TOML_PATH,
        bootstrap: bool | None = None,
        preserve_defaults: bool = False) -> None:
    """Write ``INCAR.toml`` (at *path*) with a sparse nested representation.

    *values*, *defaults*, and *raw_existing* use the same nested shape
    as :func:`tomllib.load`: ``[section]`` tables are dict values.

    In *bootstrap* mode (no file exists yet), every owned default is
    written so the user can see all available parameters.  In *update*
    mode, owned keys equal to their default are omitted (unless
    *preserve_defaults* is set).  Keys present in *raw_existing* but
    absent from *defaults* are preserved verbatim.

    When the resulting content matches the file on disk, no write
    occurs.  When a write does occur, the existing file is backed up to
    ``INCAR.toml.old`` next to *path*.

    Args:
        values: Resolved parameter values (nested; sections are dicts).
        raw_existing: Previously-read ``INCAR.toml`` content, used to
            preserve non-owned keys.  ``None`` or empty means no file
            exists yet (bootstrap mode).
        defaults: Hardcoded defaults (nested).  Keys present here are
            owned; only non-default values are written in update mode.
        always_include: Top-level keys written even when equal to the
            default or ``None``.
        path: Destination file.
        bootstrap: Force bootstrap (``True``) or update (``False``)
            behaviour.  ``None`` (default) auto-detects from file
            existence.
        preserve_defaults: When True, keep owned keys whose value equals
            the default but were already present in *raw_existing*.
    """
    if raw_existing is None:
        raw_existing = {}
    if defaults is None:
        defaults = {}
    if always_include is None:
        always_include = set()
    if bootstrap is None:
        is_bootstrap = not path.is_file()
    else:
        is_bootstrap = bootstrap

    merged = _merge_toml(
        values, raw_existing, defaults,
        is_bootstrap=is_bootstrap,
        always_include=always_include,
        preserve_defaults=preserve_defaults,
    )

    if not is_bootstrap and merged == raw_existing:
        return  # nothing changed

    if path.is_file():
        backup = path.with_name(f'{path.name}.old')
        if backup.is_file():
            backup.unlink()
        shutil.copy2(path, backup)
        logger.info('Backed up %s -> %s', path.name, backup.name)

    content = _format_toml(merged)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content + ('\n' if content else ''))
    label = 'Bootstrapped' if is_bootstrap else 'Updated'
    logger.info('%s %s', label, path.name)
