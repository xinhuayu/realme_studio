"""
Stance-aware pacing, added by WRAPPING an adapter rather than changing one.

Every adapter in this project already takes a per-utterance `pace` multiplier,
and piper already applies it on top of its configured speed:

    scale = 1.0 / (self.speed * (pace or 1.0))

That is exactly the relationship this needs. `self.speed` is the measured
dialogue pace -- set by `realme voice pace`, or the calibrated fallback in
`casting.DIALOGUE_SPEED` -- and a stance must MODIFY that, never replace it.
A stance that set an absolute speed would silently discard a measurement
someone took on purpose.

So there is nothing to change in the adapters. This is a delegating proxy: it
answers every attribute from the adapter it wraps, and intercepts exactly one
call to multiply in the stance's pace. Delete this file and the adapters are
untouched, because they never knew about it.

The fingerprint is the one place the proxy must NOT be transparent. The render
ledger keys cached audio on `voice_fingerprint()`, and if a stance-paced voice
reported the same fingerprint as a flat one, an experiment would be served the
flat audio it rendered an hour ago and would measure nothing at all.
"""
from __future__ import annotations
from realme.expressive import stance as ST


class StancePaced:
    """An adapter that speeds or slows each utterance by what the turn is doing.

    Usage is deliberately explicit -- the stance is set before each call rather
    than inferred -- because the thing being tested is whether a KNOWN stance
    produces an audible difference, and an adapter that guessed would make the
    result impossible to attribute.
    """

    def __init__(self, inner, *, enabled: bool = True, strength: float = 1.0):
        #: `strength` scales every multiplier toward 1.0, so an experiment can
        #: sweep the whole mapping with one number instead of editing a table.
        #: 0.0 reproduces the unwrapped adapter exactly, which is what makes
        #: the control arm of a comparison trustworthy.
        self._inner = inner
        self._enabled = bool(enabled)
        self._strength = float(strength)
        self._stance = ST.NEUTRAL
        self.applied: list[tuple[str, float]] = []   # what it actually did

    # -- the one thing it changes ------------------------------------------
    def set_stance(self, raw: str | None) -> tuple[str, bool]:
        st, known = ST.normalize(raw)
        self._stance = st
        return st, known

    def pace_now(self) -> float:
        if not self._enabled:
            return 1.0
        m = ST.pace_for(self._stance)
        return 1.0 + (m - 1.0) * self._strength

    def synthesize(self, text, out_wav, voice=None, pace=1.0, **kw):
        mult = self.pace_now()
        self.applied.append((self._stance, mult))
        return self._inner.synthesize(text, out_wav, voice=voice,
                                      pace=(pace or 1.0) * mult, **kw)

    # -- the one thing it must not hide -------------------------------------
    def voice_fingerprint(self) -> str:
        base = self._inner.voice_fingerprint()
        if not self._enabled or self._strength == 0.0:
            return base
        return f"{base}|stance={self._stance}@{self._strength:g}"

    # -- everything else is the adapter it wraps ----------------------------
    def __getattr__(self, name):
        # Only reached for attributes this class does not define, so the
        # overrides above always win and everything else -- name, model,
        # is_voice_clone, reference_wav, speaks_languages -- is the real
        # adapter's, including attributes added to it later.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return (f"StancePaced({self._inner!r}, enabled={self._enabled}, "
                f"strength={self._strength:g})")


class StanceVoiced:
    """An adapter that speaks each turn in the register matching its stance.

    The engine already takes a per-utterance reference -- `synthesize(...,
    voice=<wav>)` -- so switching registers needs nothing added to it. This
    chooses which one, from the stance the script writer already assigns.

    Wrapping rather than editing, for the same reason as `StancePaced`: the
    adapter never learns that registers exist, and deleting this file leaves
    it as it was. The two wrappers compose -- pace one, voice the other --
    because each intercepts a different part of the same call.

    A stance with no register recorded falls back to the neutral one, and says
    so once per stance rather than once per turn: a dialogue of forty turns
    should not print the same line forty times.
    """

    def __init__(self, inner, registers: dict, *, enabled: bool = True,
                 neutral: str = "explaining", log=None):
        self._inner = inner
        self._registers = {k: v for k, v in (registers or {}).items() if v}
        self._enabled = bool(enabled) and bool(self._registers)
        self._neutral = neutral
        self._log = log
        self._stance = None
        self._warned: set = set()
        self.used: dict = {}          # register -> how many turns

    #: Which register speaks which stance. A stance with no register of its own
    #: borrows the nearest one that shares its direction rather than falling
    #: straight to neutral: a summary is closer to conceding than to pressing.
    STANCE_REGISTER = {
        "press": "pressing", "question": "wondering", "qualify": "conceding",
        "concede": "conceding", "agree": "pressing", "summarise": "conceding",
        "open": "explaining", "close": "conceding", "neutral": "explaining",
    }

    def set_stance(self, raw):
        st, known = ST.normalize(raw)
        self._stance = st
        return st, known

    def register_now(self) -> str:
        if not self._enabled:
            return ""
        want = self.STANCE_REGISTER.get(self._stance or "", self._neutral)
        if want in self._registers:
            return want
        if want not in self._warned:
            self._warned.add(want)
            if self._log:
                self._log(f"    no '{want}' register recorded; "
                          f"'{self._stance}' turns use '{self._neutral}'")
        return self._neutral if self._neutral in self._registers else ""

    def synthesize(self, text, out_wav, voice=None, **kw):
        reg = self.register_now()
        if reg:
            self.used[reg] = self.used.get(reg, 0) + 1
        return self._inner.synthesize(text, out_wav,
                                      voice=(self._registers.get(reg) or voice),
                                      **kw)

    def voice_fingerprint(self) -> str:
        base = self._inner.voice_fingerprint()
        reg = self.register_now()
        return f"{base}|reg={reg}" if reg else base

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return (f"StanceVoiced({self._inner!r}, "
                f"registers={sorted(self._registers)})")


class ExpressionShifted:
    """An adapter whose speaker embedding is nudged along an emotion direction.

    The continuous version of `StanceVoiced`. That one switches between the
    four references that were actually recorded; this one moves between them,
    and past them, by handing the engine `x(neutral) + alpha * tau(stance)`
    instead of a file. Four recordings become a continuum.

    Wrapping rather than editing, as with the other two, but with one
    difference that cannot be wrapped away: the engine has to accept a vector.
    `set_expression` on the adapter is the whole of that change, and it is
    inert until called -- an adapter that never hears from this class produces
    exactly the bytes it produced before this file existed.

    An adapter that has no `set_expression` at all is not silently tolerated.
    It would run, sound completely ordinary, and be mistaken for a null result
    about emotion vectors when it was really a null result about this wrapper.
    So it says so, once, loudly.
    """

    #: Which direction each stance leans on. The same table as StanceVoiced's,
    #: minus the registers that are only ever the origin: 'explaining' is
    #: alpha = 0, which is to say no shift at all.
    STANCE_DIRECTION = {
        "press": "pressing", "agree": "pressing",
        "question": "wondering",
        "qualify": "conceding", "concede": "conceding",
        "summarise": "conceding", "close": "conceding",
        "open": "", "neutral": "",
    }

    #: How far along each direction, per direction.
    #:
    #: Not one number, because the registers are not equally far from neutral
    #: to begin with and the ear did not like them at the same strength. These
    #: are where the sweep's listening landed; `alpha=` overrides the lot, and
    #: `alphas=` overrides them one at a time.
    ALPHA = {"pressing": 0.85, "conceding": 0.85, "wondering": 0.85}

    def __init__(self, inner, tauset, *, alpha: float | None = None,
                 alphas: dict | None = None,
                 renorm: bool = True, enabled: bool = True, log=None):
        # Two wrappers cannot both decide what the voice is.
        #
        # StanceVoiced passes `voice=<register>.wav`, which makes the engine
        # encode THAT clip -- and then this wrapper replaces the result
        # outright. The register would be silently discarded, after paying for
        # the speaker encoder to produce it, and the fingerprint would claim
        # both were in force. alpha = 1 already IS the register, so stacking
        # them is not even a thing anyone would want.
        if isinstance(inner, StanceVoiced) or hasattr(inner, "_registers"):
            raise ValueError(
                "a register-switching voice cannot also be shifted by an "
                "emotion direction: the register it chooses would be encoded "
                "and then thrown away. Pick one -- alpha = 1 along a direction "
                "is that register.")
        self._inner = inner
        self._tau = tauset
        self._alphas = dict(self.ALPHA)
        if alphas:
            self._alphas.update({k: float(v) for k, v in alphas.items()})
        if alpha is not None:
            self._alphas = {k: float(alpha) for k in self._alphas}
        self._alpha = float(alpha) if alpha is not None else 0.0
        self._renorm = bool(renorm)
        self._enabled = bool(enabled) and tauset is not None
        self._log = log
        self._stance = None
        self._supported = hasattr(inner, "set_expression")
        self._capped: set = set()
        self.applied: list = []        # (stance, direction, alpha) per turn
        if self._enabled and not self._supported and log:
            log(f"    {type(inner).__name__} takes a reference file but not a "
                f"vector, so emotion directions cannot be applied to it. "
                f"Turns will render in the ordinary voice.")

    def set_stance(self, raw):
        from realme.expressive import stance as _ST
        st, known = _ST.normalize(raw)
        self._stance = st
        return st, known

    def direction_now(self) -> str:
        if not self._enabled or not self._supported:
            return ""
        want = self.STANCE_DIRECTION.get(self._stance or "", "")
        return want if want in getattr(self._tau, "directions", {}) else ""

    def alpha_for(self, direction: str) -> float:
        return float(self._alphas.get(direction, 0.0))

    def alpha_now(self) -> float:
        """The strength for this turn, never past what the takes support.

        Clamped to the measured ceiling rather than to a constant: a direction
        built from two takes that barely differ cannot carry alpha = 1, and the
        set knows that about itself.
        """
        d = self.direction_now()
        if not d:
            return 0.0
        want = self.alpha_for(d)
        try:
            cap = self._tau.usable_alpha(d, renorm=self._renorm)
        except Exception:
            cap = abs(want)
        a = min(abs(want), cap) * (1.0 if want >= 0 else -1.0)
        if cap < abs(want) and d not in self._capped:
            self._capped.add(d)
            if self._log:
                self._log(f"    alpha held at {cap:.2f} for '{d}' (asked "
                          f"{want:+.2f}): past that the embedding leaves the "
                          f"range your own takes covered")
        return a

    def synthesize(self, text, out_wav, voice=None, **kw):
        d, a = self.direction_now(), self.alpha_now()
        self.applied.append((self._stance, d, round(a, 3)))
        if self._supported:
            # A turn with no direction still gets a vector: the ORIGIN of this
            # space, x(explaining).
            #
            # Leaving it None looked like the obvious no-op and was not one.
            # The adapter then falls back to its own `reference_wav` -- the
            # enrolled clip, a different recording on a different day from the
            # register take every direction is measured from. So the
            # unshifted turns of a dialogue were spoken by a slightly
            # different voice than the shifted ones, and in a comparison
            # against the register switcher (which uses the neutral TAKE for
            # those turns) the two arms were not the same condition even at
            # alpha = 1, where the arithmetic says they must be.
            #
            # A control that is quietly a different voice is the fault this
            # project keeps rediscovering. Anchor it explicitly.
            vec = (self._tau.vector({d: 1.0}, a, renorm=self._renorm)
                   if d and a else list(self._tau.base))
            self._inner.set_expression(vec, alpha=a)
        try:
            return self._inner.synthesize(text, out_wav, voice=voice, **kw)
        finally:
            if self._supported:
                self._inner.set_expression(None)

    def voice_fingerprint(self) -> str:
        """Never transparent while this wrapper is in force.

        The render ledger keys cached audio on this string. A shifted turn that
        reported the flat fingerprint would be served the flat take rendered an
        hour ago, and the experiment would measure nothing -- the same trap
        StancePaced documents, one layer along.
        """
        base = self._inner.voice_fingerprint()
        if not (self._enabled and self._supported):
            return base
        d, a = self.direction_now(), self.alpha_now()
        # Even an unshifted turn is keyed apart, because it is no longer the
        # adapter's own reference clip speaking -- it is the neutral register.
        tag = f"{d}@{a:g}" if (d and a) else "base"
        return f"{base}|tau={tag}{'' if self._renorm else ':raw'}"

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return (f"ExpressionShifted({self._inner!r}, "
                f"alphas={self._alphas}, renorm={self._renorm})")
