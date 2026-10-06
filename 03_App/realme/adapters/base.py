"""
Adapter contracts.

Rule enforced across every adapter in RealMe v2:
  An adapter either does the real thing, or raises. It NEVER silently
  substitutes fake output for real output. Preview/placeholder behaviour is
  opt-in, is a distinct adapter class, and always stamps its output as
  placeholder so it can never be mistaken for a finished asset.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path


class AdapterUnavailable(RuntimeError):
    """Raised when an adapter's dependency, key, or model is missing."""


class BaseTTS(ABC):
    #: Human-readable identity recorded in output provenance.
    name: str = "base"
    #: True if this adapter reproduces the instructor's cloned voice.
    is_voice_clone: bool = False
    #: True if output is a stand-in, not a deliverable.
    is_placeholder: bool = False
    #: Languages this adapter can be asked for per utterance, as 2-letter
    #: codes. Empty means it speaks whatever it was built for and a
    #: [[fr-FR]] block cannot be honoured -- which is worth saying rather
    #: than rendering French text with an English voice and no warning.
    speaks_languages: tuple = ()
    #: How much text this engine does WELL in one call, in characters.
    #: None means "use the pipeline's default" -- which is the right answer
    #: for every local autoregressive engine here, because they drift past
    #: roughly twenty seconds of continuous generation and the chunking in
    #: `pipeline/speak.py` exists to bound that.
    #:
    #: A hosted engine may do better with a whole paragraph, and its API
    #: limit is far higher than its QUALITY limit. Those are two different
    #: numbers and only one of them matters here: this is the measured one,
    #: from `probe_gemini_tts.py` or its equivalent, never the documented
    #: ceiling.
    max_chars_per_call: int | None = None
    #: Which inline pronunciation syntax this engine understands, if any.
    #: One of "espeak", "indextts", "moss", "voxcpm", or None for engines
    #: with no phoneme interface (they get a respelling instead).
    phoneme_syntax: str | None = None

    def check(self) -> None:
        """Could this engine run? Raises AdapterUnavailable if not.

        Separate from `preflight` because they are different questions and
        one of them is expensive. `preflight` means "be ready to speak", and
        for the C++ engine that is a 2 GB model load taking tens of seconds.
        A status page only ever wanted "is the library there and are the
        weights converted" -- and, by asking `preflight`, paid the full load
        on every page refresh and then threw the loaded model away.

        The default is the same question, because for most adapters getting
        ready costs nothing.
        """
        self.preflight()

    def last_rush(self) -> float | None:
        """How much faster than its own norm the LAST take came out, or None.

        1.12 means "that one was 12% faster than this voice usually is". The
        default is None: most engines have no view on this, and an engine that
        does not measure must not be guessed at.

        It exists because of the one failure a length check cannot see. A
        hurried rendering contains every word -- nothing is missing, nothing
        is repeated -- so the only thing that notices is a person listening to
        the finished lecture. `pipeline/speak.py` asks this after each fresh
        synthesis and re-records the chunk in smaller pieces when the answer
        is yes.
        """
        return None

    def voice_fingerprint(self) -> str:
        """
        Everything that changes the AUDIO for identical text.

        The render ledger keys on this. v2 keyed on the adapter NAME alone,
        which meant switching to a different cloned voice inside the same
        provider silently reused the old audio -- the cache would serve you a
        lecture in the wrong voice and report a clean run.
        """
        return self.name

    @abstractmethod
    def preflight(self) -> None:
        """Raise AdapterUnavailable if this adapter cannot run right now."""

    @abstractmethod
    def synthesize(self, text: str, out_wav: Path, voice: str | None = None,
                   pace: float = 1.0) -> Path:
        """Synthesize `text` to a real, decodable WAV at `out_wav`."""


class BaseScriptWriter(ABC):
    name: str = "base"
    is_placeholder: bool = False

    @abstractmethod
    def preflight(self) -> None: ...

    @abstractmethod
    def write_lecture(self, slide_images: list[Path], notes: list[str],
                      style: str, course_context: str) -> "object":
        """Return a Manifest."""

    @abstractmethod
    def write_dialogue(self, source_text: str, topic: str, mode: str,
                       turns: int) -> "object":
        """Return a DialogueScript."""
