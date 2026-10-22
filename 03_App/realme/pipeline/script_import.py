"""
One script file for a whole deck.

Adopted from real_voice_qwen3. Writing narration in your own editor and
importing it beats typing into text boxes, and the strict 1..N validation is
the part that matters: a script whose slide numbers skip or repeat would map
silently onto the wrong slides, so it is refused instead.

Accepted labels, one per line:

    [[slide 1]]              [[slide 1: Study design]]
    ## Slide 1              ## Slide 1 - Study design
"""
from __future__ import annotations
import re
from dataclasses import dataclass


class ScriptImportError(ValueError):
    """The file cannot be mapped safely onto slides."""


@dataclass(frozen=True)
class Section:
    slide_number: int
    title: str
    script: str


@dataclass(frozen=True)
class ImportedScript:
    marker_format: str
    sections: tuple[Section, ...]
    warnings: tuple[str, ...]


_BRACKET = re.compile(
    r"^\s*\[\[\s*slide\s+(?P<number>\d+)"
    r"(?:\s*[:|\-–—]\s*(?P<title>.*?))?\s*\]\]\s*$", re.IGNORECASE)
_MARKDOWN = re.compile(
    r"^\s*#{1,6}\s*slide\s+(?P<number>\d+)"
    r"(?:\s*[:|\-–—]\s*(?P<title>.*?))?\s*#*\s*$", re.IGNORECASE)


def _marker(line: str):
    for fmt, pattern in (("bracket", _BRACKET), ("markdown", _MARKDOWN)):
        m = pattern.match(line)
        if m:
            return int(m.group("number")), (m.group("title") or "").strip().rstrip("#").strip(), fmt
    return None


def parse_slide_script(source: str) -> ImportedScript:
    text = source.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    found: list[tuple[int, str, str, list[str]]] = []
    preamble: list[str] = []
    warnings: list[str] = []

    for line in text.split("\n"):
        hit = _marker(line)
        if hit:
            number, title, fmt = hit
            found.append((number, title, fmt, []))
        elif found:
            found[-1][3].append(line)
        elif line.strip():
            preamble.append(line)

    if not found:
        raise ScriptImportError(
            "No slide labels found. Mark each slide with a line like "
            "'[[slide 1: Title]]' or '## Slide 1'.")
    if preamble:
        warnings.append(f"{len(preamble)} line(s) before the first slide label "
                        f"were ignored.")

    numbers = [n for n, _, _, _ in found]
    expected = list(range(1, len(found) + 1))
    if sorted(numbers) != expected:
        missing = sorted(set(expected) - set(numbers))
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        detail = []
        if missing:
            detail.append(f"missing slide(s) {missing}")
        if dupes:
            detail.append(f"repeated slide(s) {dupes}")
        raise ScriptImportError(
            "Slide labels must form a complete 1.."
            f"{len(found)} sequence; found {numbers}"
            + (" — " + ", ".join(detail) if detail else "") +
            ". Refusing to guess which slide each section belongs to.")

    sections = []
    for number, title, _, body in sorted(found, key=lambda f: f[0]):
        script = "\n".join(body).strip()
        if not script:
            warnings.append(f"Slide {number} has a label but no narration.")
        sections.append(Section(number, title, script))
    return ImportedScript(marker_format=found[0][2],
                          sections=tuple(sections), warnings=tuple(warnings))


def to_manifest(imported: ImportedScript, project_id: str, title: str = ""):
    """Turn an imported script into a Manifest the render pipeline accepts."""
    from realme.core.schema import Manifest, Segment, Prosody
    return Manifest(
        project_id=project_id,
        title=title or (imported.sections[0].title if imported.sections else project_id),
        script_source="imported-script",
        segments=[Segment(segment_id=i + 1, slide_index=s.slide_number - 1,
                          spoken_text=s.script or "(no narration)",
                          prosody=Prosody())
                  for i, s in enumerate(imported.sections)])
