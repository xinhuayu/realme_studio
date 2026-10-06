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


class QuotaExceeded(AdapterUnavailable):
    """The account's own limit, not a fault in the request.

    Separate from `AdapterUnavailable` because the response is different in
    kind: nothing is wrong with the text, the voice or the key, and retrying
    the same question sooner makes it worse -- each attempt counts against the
    very quota that is exhausted. The old code retried a 429 four times with
    exponential backoff, which spent four more of a hundred daily requests to
    learn the same thing twice.

    Carries what a person needs in order to act: how long until it clears, and
    what the limit was.
    """

    def __init__(self, message: str, *, retry_after_s: float | None = None,
                 limit: int | None = None, period: str = "", tier: str = ""):
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.limit = limit
        self.period = period          # "day" or "minute", when Google says
        self.tier = tier

    @property
    def per_day(self) -> bool:
        return self.period == "day" or (self.retry_after_s or 0) > 600


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
    #: True when this engine has no rate control of its own and `pace` is
    #: realised by stretching the finished audio.
    #:
    #: It matters because of WHERE the stretching happens. Doing it inside
    #: `synthesize` bakes it into the file the ledger caches, so changing a
    #: pace afterwards means paying the engine again for every utterance in
    #: the lecture -- which is what happened: a pace correction was corrected
    #: and a whole guide video had to be re-rendered to apply it. An engine
    #: that declares this returns its own audio and leaves the stretching to
    #: the caller, which caches the stretched copy separately: the paid take
    #: survives a pace change, and only the free ffmpeg step is redone.
    #:
    #: False is right for an engine with a real rate parameter (piper's
    #: `speed`): there the pace changes the synthesis, so it cannot be moved
    #: downstream and it belongs in the paid key.
    retimes_after: bool = False

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

    def budget_note(self, calls: int) -> str:
        """What to say before a render that may run out of requests, or "".

        Default empty: a local engine has no quota to run out of, and an
        engine that charges per call knows its own limits. Asked by the
        pipeline so that the warning can be engine-specific without the
        pipeline knowing which engine it has.
        """
        return ""

    def timing_factor(self, pace: float = 1.0) -> float:
        """What a caller must still apply to this engine's output, as a speed.

        1.0 for an engine that has already done whatever `pace` asked for.
        See `retimes_after`.
        """
        return 1.0

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
