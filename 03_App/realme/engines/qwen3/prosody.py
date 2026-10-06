"""Small, deterministic prosody helpers for the local Qwen3-TTS worker.

These helpers do not change authored words.  They only divide longer text into
manageable delivery units and apply very small, punctuation-aware timing
differences before the units are joined again.
"""

from __future__ import annotations

import re
from typing import Optional

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?。！？；;])\s+|(?<=[。！？；;])(?=\S)|\n{2,}"
)


_HARD_STOP_CHARS = ".!?\u3002\uff01\uff1f"
_CLAUSE_STOP_CHARS = ",:;\u3001\uff0c\uff1a\uff1b\u2014\u2013"


def _punctuation_units(text: str) -> list[tuple[str, bool]]:
    """Split at sentence/clause punctuation without discarding the boundary."""

    units: list[tuple[str, bool]] = []
    start = 0
    for index, character in enumerate(text):
        if character not in _HARD_STOP_CHARS + _CLAUSE_STOP_CHARS:
            continue
        next_character = text[index + 1] if index + 1 < len(text) else ""
        if (
            next_character
            and not next_character.isspace()
            and character in _CLAUSE_STOP_CHARS
            and not ("\u3000" <= next_character <= "\u9fff")
        ):
            # Chinese punctuation is a boundary without an intervening space;
            # English commas/colons normally need whitespace before splitting.
            continue
        boundary = index + 1
        if next_character and next_character.isspace():
            boundary += len(next_character)
        candidate = text[start:boundary].strip()
        if candidate:
            units.append((candidate, character in _HARD_STOP_CHARS))
        start = boundary
    remainder = text[start:].strip()
    if remainder:
        units.append((remainder, False))
    return units


def split_delivery_units(text: str, *, mode: str = "natural") -> list[str]:
    """Return sentence/clause units while keeping hard sentence boundaries.

    ``stable`` preserves the old one-request behavior. ``natural`` and
    ``expressive`` split likely clauses and only merge clauses within the same
    sentence when a long passage would otherwise create too many model calls.
    Sentences are never merged together merely to meet the soft unit target.
    """

    cleaned = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not cleaned:
        return []
    if mode == "stable":
        return [cleaned]
    maximum = 16 if mode == "expressive" else 12
    raw_units = _punctuation_units(cleaned)
    if len(raw_units) <= maximum:
        return [unit for unit, _hard_boundary in raw_units]

    # Reduce only clause-level units. A hard sentence boundary closes a group,
    # so a long passage may exceed the soft target rather than run sentences
    # together in one model request.
    grouped: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    for unit in raw_units:
        current.append(unit)
        if unit[1]:
            grouped.append(current)
            current = []
    if current:
        grouped.append(current)

    while sum(len(group) for group in grouped) > maximum:
        candidates = [group for group in grouped if len(group) > 1]
        if not candidates:
            break
        target = max(candidates, key=len)
        first, second = target[0], target[1]
        target[:2] = [(f"{first[0]} {second[0]}".strip(), second[1])]
    return [unit for group in grouped for unit, _hard_boundary in group]


def delivery_speed_multiplier(unit: str, index: int, total: int, *, mode: str = "natural") -> float:
    """Return a deliberately small speed change for one delivery unit."""

    if mode == "stable" or total <= 1:
        return 1.0
    # Keep the rhythm clearly alive without turning it into an artificial
    # metronome. These small alternating changes are audible across clauses
    # while remaining close to the reference delivery.
    multiplier = 0.975 if index % 2 else 1.025
    if "?" in unit or "？" in unit:
        multiplier -= 0.015
    elif "!" in unit or "！" in unit:
        multiplier += 0.015
    if len(unit) > 180:
        multiplier -= 0.01
    if mode == "expressive":
        multiplier = 1.0 + (multiplier - 1.0) * 1.35
    return max(0.94, min(1.06, round(multiplier, 3)))


def _legacy_delivery_gap_ms(
    unit: str,
    *,
    mode: str = "natural",
    pause_scale: float = 1.0,
) -> int:
    """Return a short, punctuation-aware inter-unit pause."""

    pause_scale = max(0.85, min(1.25, float(pause_scale)))
    if mode == "stable":
        return 0
    if "?" in unit or "？" in unit:
        base = 145 if mode == "natural" else 165
        return round(base * pause_scale)
    if "!" in unit or "！" in unit:
        base = 120 if mode == "natural" else 140
        return round(base * pause_scale)
    base = 95 if mode == "natural" else 115
    return round(base * pause_scale)


def delivery_gap_ms(
    unit: str,
    *,
    mode: str = "natural",
    pause_scale: float = 1.0,
    reference_pause_seconds: Optional[float] = None,
    reference_longest_pause_seconds: Optional[float] = None,
    breath_pause_candidates: int = 0,
) -> int:
    """Return a bounded pause that preserves the sample's spacing.

    Sentence and clause units are rendered separately by the worker. This
    join-time silence is deliberately longer than the old 95 ms default and
    is lifted by the retained sample's measured pause/breath candidates,
    without copying one unusually long recording pause everywhere.
    """

    pause_scale = max(0.85, min(1.25, float(pause_scale)))
    if mode == "stable":
        return 0
    cleaned = unit.strip()
    hard_boundary = cleaned.endswith(tuple(_HARD_STOP_CHARS))
    clause_boundary = cleaned.endswith(tuple(_CLAUSE_STOP_CHARS))
    if "?" in cleaned or "\uff1f" in cleaned:
        base = 220 if mode == "natural" else 255
    elif "!" in cleaned or "\uff01" in cleaned:
        base = 195 if mode == "natural" else 225
    elif hard_boundary:
        base = 185 if mode == "natural" else 215
    elif clause_boundary:
        base = 125 if mode == "natural" else 150
    else:
        base = 110 if mode == "natural" else 125

    if reference_pause_seconds is not None and reference_pause_seconds > 0:
        observed_limit = 700 if hard_boundary else 420
        observed_factor = 0.9 if hard_boundary else 0.6
        base = max(
            base,
            min(observed_limit, round(reference_pause_seconds * 1000 * observed_factor)),
        )
    if hard_boundary and reference_longest_pause_seconds is not None and reference_longest_pause_seconds > 0:
        base = max(base, min(700, round(reference_longest_pause_seconds * 1000 * 0.35)))
    if hard_boundary and breath_pause_candidates > 0:
        base += 25
    return round(base * pause_scale)
