"""
Stance-aware pacing, added by WRAPPING an adapter rather than changing one.

Every adapter takes a per-utterance `pace` multiplier, and piper applies it
on top of its configured speed:

    scale = 1.0 / (self.speed * (pace or 1.0))

`self.speed` is the measured dialogue pace -- set by `realme voice pace`, or
the calibrated fallback in `casting.DIALOGUE_SPEED` -- and a stance must
MODIFY that, never replace it, so the measurement is kept.

The adapters need no change. This is a delegating proxy: it answers every
attribute from the adapter it wraps, and intercepts exactly one call to
multiply in the stance's pace.

The fingerprint is the one place the proxy must NOT be transparent. The render
ledger keys cached audio on `voice_fingerprint()`, and if a stance-paced voice
reported the same fingerprint as a flat one, an experiment would be served
cached flat audio.
"""
from __future__ import annotations
from realme.expressive import stance as ST


class StancePaced:
    """An adapter that speeds or slows each utterance by what the turn is doing.

    Usage is deliberately explicit -- the stance is set before each call rather
    than inferred -- because the thing being tested is whether a KNOWN stance
    produces an audible difference, and an adapter that guessed would make the
    result hard to attribute.
    """

    def __init__(self, inner, *, enabled: bool = True, strength: float = 1.0):
        #: `strength` scales every multiplier toward 1.0, so an experiment can
        #: sweep the whole mapping with one number instead of editing a table.
        #: 0.0 reproduces the unwrapped adapter exactly, for the control arm
        #: of a comparison.
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

    The engine takes a per-utterance reference -- `synthesize(...,
    voice=<wav>)` -- so switching registers needs nothing added to it. This
    chooses which one, from the stance the script writer assigns.

    Wrapping rather than editing, as with `StancePaced`: the adapter does not
    know that registers exist. The two wrappers compose -- pace one, voice
    the other -- because each intercepts a different part of the same call.

    A stance with no register recorded falls back to the neutral one, and says
    so once per stance rather than once per turn.
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
    instead of a file.

    Wrapping rather than editing, as with the other two, except that the
    engine has to accept a vector. `set_expression` on the adapter provides
    that, and it is inert until called: an adapter that is never wrapped by
    this class produces identical output.

    An adapter that has no `set_expression` would run and sound ordinary,
    which could be mistaken for a null result about emotion vectors. This is
    reported once.
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
    #: Per direction, because the registers are not equally far from neutral.
    #: Values chosen by listening to sweeps; `alpha=` overrides the lot, and
    #: `alphas=` overrides them one at a time.
    ALPHA = {"pressing": 0.85, "conceding": 0.85, "wondering": 0.85}

    def __init__(self, inner, tauset, *, alpha: float | None = None,
                 alphas: dict | None = None,
                 renorm: bool = True, enabled: bool = True, log=None):
        # Two wrappers cannot both decide what the voice is.
        #
        # StanceVoiced passes `voice=<register>.wav`, which makes the engine
        # encode THAT clip -- and then this wrapper replaces the result
        # outright. The register would be silently discarded and the
        # fingerprint would claim both were in force. alpha = 1 along a
        # direction already IS the register.
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
        built from two takes that barely differ cannot carry alpha = 1.
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
            # None is not a no-op: the adapter would fall back to its own
            # `reference_wav`, the enrolled clip, which is a different
            # recording from the register take every direction is measured
            # from. Unshifted turns would then use a slightly different voice,
            # and a comparison against the register switcher (which uses the
            # neutral TAKE for those turns) would not match even at alpha = 1.
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
        reported the flat fingerprint would be served the cached flat take,
        as described for StancePaced.
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
