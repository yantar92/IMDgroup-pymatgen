# MIT License
#
# Copyright (c) 2026-2026 Inverse Materials Design Group
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


"""Structured warning records shared by VASP output consumers.

Three layers make up the warning system:

- :class:`VaspWarning` - the Python ``Warning`` category emitted via
  ``warnings.warn``.  This is the *notification* channel.
- :class:`VaspWarningRecord` - one structured, name-keyed diagnostic.
  This is the *data* channel used for programmatic decisions (e.g.
  re-run VASP with different parameters).
- :class:`VaspWarnings` - an insertion-ordered collection of records
  keyed by name.

Every producer (:class:`Vasplog`, :class:`Outcar`, :class:`Vasprun`,
and :class:`IMDGVaspDir`) populates a :class:`VaspWarnings` container.
The container is what downstream code should inspect; ``warnings.warn``
remains as a side channel so existing notification timing is preserved.
"""

import warnings
from dataclasses import dataclass, field
from typing import Iterator

from monty.json import MSONable


class VaspWarning(Warning):
    """Warning category emitted for VASP run problems.

    Emitting this category is the notification channel; the structured
    records are collected in :class:`VaspWarnings`.
    """


@dataclass(frozen=True)
class VaspWarningRecord(MSONable):
    """One structured warning record.

    Attributes:
        name: Stable machine-readable key used for branching (e.g.
            ``"force_convergence"``, ``"time_limit"``).
        message: Human-readable description.
        tips: Optional remediation hints.
        count: Number of occurrences.  Accumulated for repeating log
            warnings; ``1`` for single-shot derived checks.
        source: File path or producer name that produced the record.
        metadata: Structured key/value data for decisions (e.g.
            ``{"max_force": 0.12}``).
    """

    name: str
    message: str
    tips: list[str] = field(default_factory=list)
    count: int = 1
    source: str | None = None
    metadata: dict = field(default_factory=dict)

    def merge(self, other: "VaspWarningRecord") -> "VaspWarningRecord":
        """Combine ``other`` into this record.

        ``message``, ``tips``, ``source`` and ``metadata`` take the
        newer value; ``count`` accumulates.
        """
        if self.name != other.name:
            raise ValueError(
                f"Cannot merge {self.name!r} with {other.name!r}")
        return VaspWarningRecord(
            name=self.name,
            message=other.message,
            tips=other.tips or self.tips,
            count=self.count + other.count,
            source=self.source or other.source,
            metadata={**self.metadata, **other.metadata},
        )

    def as_dict(self) -> dict:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "name": self.name,
            "message": self.message,
            "tips": list(self.tips),
            "count": self.count,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VaspWarningRecord":
        d = dict(d)
        d.pop("@module", None)
        d.pop("@class", None)
        return cls(**d)


class VaspWarnings(MSONable):
    """Name-keyed, insertion-ordered collection of warning records.

    ``add`` accumulates (for repeating log warnings, where the same
    pattern appears many times); ``overwrite`` replaces (for single-shot
    derived checks, so re-running a check is idempotent).
    """

    def __init__(self, records: list[VaspWarningRecord] | None = None) -> None:
        self._records: dict[str, VaspWarningRecord] = {}
        if records:
            for record in records:
                self.add(record)

    def add(self, record: VaspWarningRecord) -> None:
        """Insert a record, merging with an existing one of the same name."""
        if record.name in self._records:
            self._records[record.name] = self._records[record.name].merge(record)
        else:
            self._records[record.name] = record

    def overwrite(self, record: VaspWarningRecord) -> None:
        """Insert or replace a record (idempotent for derived checks)."""
        self._records[record.name] = record

    def __getitem__(self, name: str) -> VaspWarningRecord:
        return self._records[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def items(self):
        return self._records.items()

    def values(self):
        return self._records.values()

    def names(self) -> set[str]:
        return set(self._records)

    def has(self, name: str) -> bool:
        return name in self._records

    def emit(self, category: type[Warning] = VaspWarning) -> None:
        """Emit each record through ``warnings.warn``."""
        for record in self._records.values():
            warnings.warn(record.message, category)

    def as_dict(self) -> dict:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "records": [r.as_dict() for r in self._records.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VaspWarnings":
        return cls([
            VaspWarningRecord.from_dict(rd) for rd in d["records"]
        ])
