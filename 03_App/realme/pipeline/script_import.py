"""
One script file for a whole deck.

Adapted from real_voice_qwen3. Narration can be written in any editor and
imported. Slide numbers are strictly validated as 1..N: a script whose slide
numbers skip or repeat would map silently onto the wrong slides, so it is
refused instead.

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


def to_script_text(manifest, titles=()) -> tuple[str, tuple[str, ...]]:
    """A lecture's narration as one notes file, in the format
    `parse_slide_script` reads: `[[slide N: Title]]` then that slide's
    narration, for every slide 1..N. Importing the result gives the same
    narration slide by slide. Delivery markers ([[pause]] and the rest) are
    kept as written; cue words are not part of the format and are left out.

    `manifest` is a Manifest, or its JSON as a dict (`_work/manifest.json`).
    `titles` are the slides' headings, by slide, where known.
    Returns the text and any warnings."""
    if isinstance(manifest, dict):
        from realme.core.schema import Manifest
        manifest = Manifest.model_validate(manifest)
    by_slide: dict[int, list[str]] = {}
    for s in sorted(manifest.segments, key=lambda s: (s.slide_index, s.segment_id)):
        by_slide.setdefault(s.slide_index, []).append(s.spoken_text.strip())
    n = max(by_slide) + 1 if by_slide else 0
    titles = list(titles or ())
    out: list[str] = []
    warnings: list[str] = []
    for i in range(n):
        title = titles[i] if i < len(titles) else ""
        title = " ".join(str(title or "").replace("]]", "]").split())[:80]
        out.append(f"[[slide {i + 1}: {title}]]" if title else f"[[slide {i + 1}]]")
        out.append("")
        body = "\n\n".join(t for t in by_slide.get(i, []) if t)
        if not body:
            warnings.append(f"Slide {i + 1} has no narration.")
        for line in body.split("\n"):
            if _marker(line):
                warnings.append(f"Slide {i + 1}: the line '{line.strip()[:40]}' "
                                f"reads as a slide label; edit it before importing.")
        out.append(body)
        out.append("")
    return "\n".join(out).rstrip() + "\n", tuple(warnings)


def looks_like_json(raw: str, filename: str = "") -> bool:
    """A RealMe narration file (`manifest.json`) rather than a notes file."""
    return filename.lower().endswith(".json") or raw.lstrip("﻿ \t\r\n").startswith("{")


def manifest_from_json(raw: str, project_id: str):
    """A lecture's narration in RealMe's own format (`_work/manifest.json`,
    as saved by the Studio and carried in a lecture's .zip), made ready for
    a new lecture: narration, cue words, pace and pauses kept; what belongs
    to its old renders (measured lengths, captions, slide pictures, revision
    marks) cleared. The same strictness as a notes file: every slide 1..N
    present. Returns the Manifest and the number of slides."""
    import json
    from pydantic import ValidationError
    from realme.core.schema import Manifest
    try:
        data = json.loads(raw.lstrip("﻿"))
    except ValueError as e:
        raise ScriptImportError(f"This .json file could not be read: {e}")
    if not isinstance(data, dict) or "segments" not in data:
        raise ScriptImportError("This .json file is not a RealMe lecture's "
                                "narration (it has no 'segments').")
    data = {**data, "project_id": project_id or data.get("project_id") or "imported"}
    data.setdefault("title", data["project_id"])
    try:
        m = Manifest.model_validate(data)
    except ValidationError as e:
        raise ScriptImportError("This .json file is not a RealMe lecture's narration: "
                                + "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: "
                                            f"{err['msg']}" for err in e.errors()[:4]))
    if m.mode != "lecture":
        raise ScriptImportError(f"This .json file is a {m.mode}, not a lecture.")
    if not m.segments:
        raise ScriptImportError("This .json file has no narration.")
    slides = sorted({s.slide_index for s in m.segments})
    if slides != list(range(len(slides))):
        missing = sorted(set(range(slides[-1] + 1)) - set(slides))
        raise ScriptImportError(
            f"The narration skips slide(s) {[i + 1 for i in missing]}. Refusing "
            f"to guess which slide each section belongs to.")
    segs = []
    for i, s in enumerate(sorted(m.segments, key=lambda s: (s.slide_index, s.segment_id))):
        segs.append(s.model_copy(update={
            "segment_id": i + 1, "measured_audio_s": None, "spoken_cues": [],
            "slide_digest": "", "revision": "", "rerender": True}))
    return m.model_copy(update={"segments": segs}), len(slides)
