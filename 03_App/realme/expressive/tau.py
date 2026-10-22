"""
Emotion as a DIRECTION in speaker-embedding space -- EXPLORATORY.

Four recordings of the same passage, delivered four ways, produce four speaker
embeddings. Subtract one from another and what is left is the delivery, because
everything else -- the speaker, the room, the microphone, the words -- was held
constant on purpose and cancels:

    tau(pressing) = x(pressing) - x(explaining)

That vector is not a register. It is the DIFFERENCE between registers, which is
a much more useful object: registers are four points, and a difference can be
scaled. `x(explaining) + 0.5 * tau(pressing)` is a delivery half as insistent
as the one recorded, and nobody recorded it. Negative alpha goes the other way,
past neutral, into a withdrawal that was never performed at all. Two taus added
give a blend. This is the whole method, and it needs no training: the model is
untouched, and what changes is one vector handed to it at synthesis time.

WHAT MAKES IT WORK IS WHAT WAS HELD CONSTANT. Everything in this file assumes
the four takes differ ONLY in delivery. `enrollment.registers` is what enforces
that -- one sitting, one shared gain correction, the same words -- and the
distinctness table there is what proves the deliveries actually differed. A tau
built from takes recorded on different days is a microphone vector, and it will
behave like an emotion vector right up until it does not.

THREE THINGS ARE MEASURED HERE RATHER THAN ASSUMED.

1. WHETHER THE DIRECTIONS ARE REAL. Two embeddings of the same person reading
   the same words are close together by construction. If cos(x_reg, x_base) is
   ~1.0 the tau is rounding error with a name, and scaling rounding error just
   amplifies noise. `extract` reports the cosine for every register and says so
   when one is too close to call.

2. HOW FAR ALPHA MAY GO. The honest ceiling is not a constant somebody liked
   the look of; it comes from his own voice. The four real takes span a range
   of embeddings that are all, audibly, him. The smallest cosine among those
   pairs is therefore a measured floor for "still the same speaker", and a
   synthetic embedding that falls below it has left the region his own
   deliveries occupy. `usable_alpha` solves for where that happens.

3. WHETHER THE NORM MATTERS. Adding alpha*tau changes the vector's length as
   well as its direction, and a decoder can be sensitive to length in ways that
   have nothing to do with expression -- a longer vector reading as a different
   person rather than a more insistent one. The default is to rescale back to
   the base norm, keeping the direction and discarding the length change, but
   `renorm=False` exists precisely so the two can be COMPARED by listening
   instead of argued about. Neither is asserted to be right here.

Nothing in the lecture, narration or dialogue pipelines imports this. The
adapter change that goes with it (`set_expression`) is inert until something
sets one, and the byte-for-byte identity of the untouched path is a test.
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, asdict, field
from pathlib import Path

#: Below this, two embeddings are so close that their difference is dominated
#: by whatever the encoder does differently on two takes of the same thing.
#: Not a physical constant -- a threshold for "worth believing", the same role
#: MEANINGFUL_LU plays for loudness one level up.
TOO_CLOSE_COS = 0.995

#: Where to stop, when the four takes give no floor of their own (only one
#: register recorded, say). Deliberately timid: the measured floor is the
#: honest answer and this is what stands in for it while it is missing.
FALLBACK_MAX_ALPHA = 1.0


@dataclass
class Direction:
    """One register's difference from the neutral one."""
    name: str
    values: list = field(default_factory=list)
    cosine: float = 0.0        # cos(x_register, x_base): how far apart at all
    norm_ratio: float = 0.0    # |tau| / |x_base|
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TauSet:
    """Every direction extracted from one sitting, plus what bounds them."""
    engine: str = ""           # the fingerprint of the encoder that made these
    base_name: str = ""
    base: list = field(default_factory=list)
    directions: dict = field(default_factory=dict)   # name -> Direction
    floor_cos: float = 0.0     # smallest cosine among the REAL takes
    floor_between: str = ""    # which pair produced it
    session: str = ""
    dim: int = 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["directions"] = {k: (v.as_dict() if isinstance(v, Direction) else v)
                           for k, v in self.directions.items()}
        return d

    # -- the two questions a caller actually asks ---------------------------
    def vector(self, weights: dict, alpha: float = 1.0,
               renorm: bool = True) -> list:
        """The embedding for a blend of directions at a given strength."""
        return blend(self.base, {k: self.directions[k].values
                                 for k in weights if k in self.directions},
                     weights, alpha, renorm=renorm)

    def usable_alpha(self, name: str, renorm: bool = True) -> float:
        """
        The largest alpha for `name` that stays inside his own range.

        Walks outward and stops at the first step whose cosine to the base
        falls below the measured floor. Walking rather than solving because
        renormalisation makes the relationship non-linear, and a closed form
        would be a closed form for the OTHER case.
        """
        if name not in self.directions:
            raise KeyError(f"no direction called {name!r}")
        floor = self.floor_cos or 0.0
        if floor <= 0.0:
            return FALLBACK_MAX_ALPHA
        a, best = 0.0, 0.0
        while a <= 4.0:
            v = self.vector({name: 1.0}, a, renorm=renorm)
            if cosine(v, self.base) < floor:
                break
            best, a = a, a + 0.05
        return round(best, 2)


# ----------------------------------------------------------------- pure math
def dot(a, b) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def norm(a) -> float:
    return math.sqrt(dot(a, a))


def cosine(a, b) -> float:
    na, nb = norm(a), norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return dot(a, b) / (na * nb)


def blend(base, taus: dict, weights: dict, alpha: float = 1.0,
          renorm: bool = True) -> list:
    """
    base + alpha * sum(w_i * tau_i), optionally rescaled to |base|.

    Pure arithmetic on lists, with no engine anywhere near it, because this is
    the part that has to be right and the part that can be tested without a
    model, a DLL or a GPU.
    """
    out = list(base)
    for name, w in weights.items():
        t = taus.get(name)
        if t is None:
            raise KeyError(f"no direction called {name!r}")
        if len(t) != len(out):
            raise ValueError(
                f"direction {name!r} has {len(t)} values but the base "
                f"embedding has {len(out)}: these came from different encoders")
        k = alpha * float(w)
        for i, tv in enumerate(t):
            out[i] += k * tv
    if renorm:
        nb, no = norm(base), norm(out)
        if no > 0 and nb > 0:
            s = nb / no
            out = [v * s for v in out]
    return out


# -------------------------------------------------------------- extraction
def embedding_of(adapter, wav: Path) -> list:
    """
    One register's speaker embedding, as plain floats.

    Goes through the adapter rather than reimplementing the encoder, because a
    tau is only meaningful to the encoder that produced it -- the axes of this
    space are whatever that model learned, and another build's axes are not the
    same axes. The engine fingerprint is stored with the set for exactly this
    reason.
    """
    get = getattr(adapter, "_speaker_embedding", None)
    if get is None:
        raise TypeError(
            f"{type(adapter).__name__} cannot expose a speaker embedding, so "
            f"there is nothing to take a difference of. Emotion directions "
            f"need the in-process qwen3 adapter.")
    emb = get(Path(wav))
    if emb is None:
        raise ValueError(
            f"the speaker encoder returned nothing for {Path(wav).name}. "
            f"A tau cannot be built from a reference the engine would not "
            f"encode -- it would silently become a vector of zeros.")
    buf, n = emb
    return [float(buf[i]) for i in range(int(n))]


def extract(adapter, takes: dict, *, neutral: str = "explaining",
            session: str = "", log=print) -> TauSet:
    """
    Turn a set of register recordings into directions.

    `takes` maps register name -> wav, exactly as `enrollment.registers`
    writes them. The neutral take is the origin; every other take becomes a
    direction away from it.
    """
    takes = dict(takes)
    if neutral not in takes:
        raise ValueError(
            f"the {neutral!r} take is the origin of this space -- every "
            f"direction is measured from it -- so a set without it has no "
            f"directions in it, only points.")
    if len(takes) < 2:
        raise ValueError(
            "one register gives no direction. A direction is a DIFFERENCE, "
            "so at least two takes from the same sitting are needed.")

    xs, dim = {}, 0
    for name, wav in takes.items():
        xs[name] = embedding_of(adapter, Path(wav))
        if dim and len(xs[name]) != dim:
            raise ValueError(
                f"{name} encoded to {len(xs[name])} floats but the others to "
                f"{dim}: these did not come from one encoder.")
        dim = len(xs[name])
        log(f"  {name:<12} {dim} floats, |x| = {norm(xs[name]):.3f}")

    base = xs[neutral]
    ts = TauSet(engine=str(getattr(adapter, "voice_fingerprint", lambda: "")()),
                base_name=neutral, base=base, dim=dim,
                session=session or time.strftime("%Y%m%d-%H%M%S"))
    for name, x in xs.items():
        if name == neutral:
            continue
        tau = [a - b for a, b in zip(x, base)]
        c = cosine(x, base)
        d = Direction(name=name, values=tau, cosine=round(c, 5),
                      norm_ratio=round(norm(tau) / (norm(base) or 1.0), 4))
        if c >= TOO_CLOSE_COS:
            # Loudly, because the failure is otherwise invisible: a tiny tau
            # scaled up by alpha sounds like the engine being unstable, and
            # the instability gets blamed on the method rather than on the
            # recording it came from.
            d.note = (f"only {(1 - c) * 1000:.1f} thousandths from "
                      f"'{neutral}' -- this direction is mostly encoder noise. "
                      f"Re-record it further away.")
            log(f"  ! {name}: {d.note}")
        ts.directions[name] = d

    # The floor: the closest any two of his REAL takes came to each other.
    # Everything beyond that cosine is outside the range his own deliveries
    # covered, which is the only defensible place to stop.
    names = list(xs)
    worst, pair = 1.0, ""
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = cosine(xs[a], xs[b])
            if c < worst:
                worst, pair = c, f"{a}/{b}"
    ts.floor_cos, ts.floor_between = round(worst, 5), pair
    log(f"  floor: {worst:.4f} (the {pair} pair) -- the least alike two of "
        f"your own takes were")
    return ts


# ------------------------------------------------------------------- storage
def save(ts: TauSet, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ts.as_dict(), indent=2), encoding="utf-8")
    return path


def load(path: Path) -> TauSet:
    from realme.core.textio import read_text
    d = json.loads(read_text(path))
    ts = TauSet(engine=d.get("engine", ""), base_name=d.get("base_name", ""),
                base=list(d.get("base") or []), floor_cos=d.get("floor_cos", 0.0),
                floor_between=d.get("floor_between", ""),
                session=d.get("session", ""), dim=int(d.get("dim") or 0))
    for k, v in (d.get("directions") or {}).items():
        ts.directions[k] = Direction(
            name=v.get("name", k), values=list(v.get("values") or []),
            cosine=v.get("cosine", 0.0), norm_ratio=v.get("norm_ratio", 0.0),
            note=v.get("note", ""))
    return ts


def check_engine(ts: TauSet, adapter) -> None:
    """
    Refuse a tau built by a different encoder.

    The numbers would go through -- same length, same arithmetic, an answer
    every time -- and mean nothing, because the two models do not share axes.
    A silent wrong answer is the failure mode this project keeps meeting, so
    this one is checked rather than hoped for.
    """
    now = str(getattr(adapter, "voice_fingerprint", lambda: "")())
    # Only the engine part: the reference clip legitimately differs between
    # extraction (a register) and use (the ordinary voice).
    a = (ts.engine or "").split(":")[0]
    b = now.split(":")[0]
    if a and b and a != b:
        raise ValueError(
            f"these directions were extracted by {a!r} and you are rendering "
            f"with {b!r}. Embedding axes are not shared between engines: the "
            f"arithmetic would succeed and the result would be arbitrary. "
            f"Re-run `realme voice tau` with this engine.")


def report(ts: TauSet) -> str:
    """What was extracted, and how far each direction may be pushed."""
    if not ts.directions:
        return "no directions extracted"
    lines = [f"origin: '{ts.base_name}'   {ts.dim} floats   engine "
             f"{ts.engine.split(':')[0] or 'unknown'}",
             f"floor:  {ts.floor_cos:.4f} cosine, from the {ts.floor_between} "
             f"pair of your own takes",
             "",
             f"{'direction':<12} {'cos to base':>12} {'|tau|/|x|':>10} "
             f"{'alpha max':>10}   note",
             "-" * 78]
    for name, d in sorted(ts.directions.items()):
        try:
            amax = f"{ts.usable_alpha(name):.2f}"
        except Exception:
            amax = "?"
        lines.append(f"{name:<12} {d.cosine:12.4f} {d.norm_ratio:10.3f} "
                     f"{amax:>10}   {d.note}")
    lines += ["",
              "alpha max is where a synthetic embedding stops being as close "
              "to your neutral",
              "take as two of your real takes are to each other. Past it the "
              "voice is being",
              "pushed somewhere you never went, which may still sound fine -- "
              "but it is no",
              "longer a claim this measurement supports."]
    return "\n".join(lines)


# ---------------------------------------------------------------- auditioning
#: What to say while sweeping. One sentence, because the thing being judged is
#: a timbre and a manner rather than a performance, and a listener comparing
#: nine renders will not sit through a paragraph nine times. It carries a
#: claim worth pressing and a hedge worth conceding, so the same words can
#: plausibly be delivered either way.
SWEEP_TEXT = ("The adjustment did not fix it, and I think we have to say so "
              "plainly — though I will admit the data are thinner than I would "
              "like.")

#: Where to sample each direction. Zero first, as the control: at alpha = 0
#: the blend returns the base embedding exactly, so that render goes through
#: the SAME new code path as every other one and any difference from the
#: ordinary voice is plumbing, not expression. Negative goes past neutral, the
#: other way -- the half of the range no recording covers.
SWEEP_ALPHAS = (-0.5, 0.0, 0.35, 0.7, 1.0, 1.4)


@dataclass
class Heard:
    """ONE render. A cell of the sweep holds `repeat` of these.

    One row per render rather than per cell, on purpose: the raw draws are what
    make a noise estimate possible, and a table that stored only the mean would
    have thrown away the only evidence about its own reliability.
    """
    direction: str
    alpha: float
    rep: int = 0
    wav: str = ""
    cos_to_base: float = 0.0     # of the embedding handed to the engine
    cos_render: float = 0.0      # of the embedding RE-EXTRACTED from the audio
    seconds: float = 0.0
    rate_wps: float = 0.0
    f0_median_hz: float = 0.0
    f0_range_st: float = 0.0
    energy_db: float = 0.0
    beyond: bool = False         # outside the range his own takes covered
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def sweep(adapter, ts: TauSet, outdir: Path, *, directions=None,
          alphas=SWEEP_ALPHAS, text: str = "", renorm: bool = True,
          repeat: int = 3, log=print) -> list:
    """
    Render one sentence at several strengths along each direction.

    Two measurements per render, and they answer different questions.

    `cos_to_base` is arithmetic: how far the vector we HANDED the engine sits
    from the enrolled one. It is exact and tells us nothing about the audio.

    `cos_render` is the round trip -- the embedding re-extracted from the audio
    that came back. This is the one that matters, because it is the only
    evidence that the engine did anything with the vector at all. If it stays
    flat while `cos_to_base` falls away, the shift is being ignored somewhere
    between here and the decoder, and every listening impression after that
    would be imagination. A sweep whose round trip does not move is a failed
    sweep, however good the renders sound.

    REPEATS ARE NOT OPTIONAL, and the first version of this function shipped
    without them, which was a mistake of exactly the kind this project has
    already paid for twice. The engine samples at temperature 0.9 with no
    seed, so one render of one condition is one draw. A single-draw sweep
    produced pitch swinging 65 Hz across alpha and read like a strong effect
    until the three alpha = 0 cells -- which are the SAME vector rendered three
    times -- turned out to disagree with each other by 14 Hz.

    That accident is now the design. alpha = 0 is rendered for every direction,
    so the sweep carries its own replicate set: same condition, several draws,
    no extra cost. `noise()` reads the spread off those and `sweep_report`
    refuses to call anything an effect until it clears them.
    """
    from realme.expressive.prosody import measure
    check_engine(ts, adapter)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    say = text or SWEEP_TEXT
    words = len([w for w in say.split() if any(c.isalnum() for c in w)])
    setter = getattr(adapter, "set_expression", None)
    if setter is None:
        raise TypeError(
            f"{type(adapter).__name__} cannot be handed a speaker embedding, "
            f"so there is nothing to sweep. Emotion directions need the "
            f"in-process qwen3 adapter (`realme setup adapters --tts qwen3`).")

    names = list(directions or sorted(ts.directions))
    repeat = max(1, int(repeat))
    if repeat < 2:
        log("  NOTE: --repeat 1 leaves no way to tell an effect from a draw.")
    out = []
    for name in names:
        if name not in ts.directions:
            raise KeyError(f"no direction called {name!r}")
        cap = ts.usable_alpha(name, renorm=renorm)
        for a in alphas:
            # Built ONCE per cell, not once per repeat: the vector is the
            # condition, and rebuilding it would leave open the question of
            # whether the repeats were even the same condition.
            vec = ts.vector({name: 1.0}, a, renorm=renorm)
            handed = round(cosine(vec, ts.base), 5)
            for rep in range(repeat):
                h = Heard(direction=name, alpha=float(a), rep=rep,
                          cos_to_base=handed, beyond=abs(a) > cap)
                tag = f"{'m' if a < 0 else 'p'}{abs(a):.2f}"
                wav = outdir / (f"{name}_{tag}.wav" if repeat == 1
                                else f"{name}_{tag}_r{rep + 1}.wav")
                try:
                    setter(vec, alpha=float(a))
                    adapter.synthesize(say, wav)
                except Exception as e:
                    h.note = f"would not render: {e}"
                    log(f"  {name} a={a:+.2f} r{rep + 1}: {h.note}")
                    out.append(h)
                    continue
                finally:
                    setter(None)
                h.wav = str(wav)
                p = measure(wav, words=words)
                h.seconds, h.rate_wps = p.seconds, p.rate_wps
                h.f0_median_hz, h.f0_range_st = p.f0_median_hz, p.f0_range_st
                h.energy_db = p.energy_db
                try:
                    h.cos_render = round(
                        cosine(embedding_of(adapter, wav), ts.base), 5)
                except Exception as e:
                    h.note = f"could not re-encode the render: {e}"
                log(f"  {name:<11} a={a:+.2f} r{rep + 1}  cos {h.cos_to_base:.4f}"
                    f" -> render {h.cos_render:.4f}  {p.f0_median_hz:5.0f} Hz  "
                    f"{p.rate_wps:4.1f} w/s  {p.energy_db:6.1f} dB"
                    + ("   BEYOND" if h.beyond else ""))
                out.append(h)
    (outdir / "sweep.json").write_text(
        json.dumps([h.as_dict() for h in out], indent=2), encoding="utf-8")
    return out


def cells(heard: list) -> dict:
    """The sweep collapsed to one entry per (direction, alpha), with its draws."""
    import statistics as _st
    out = {}
    for h in heard:
        if not h.wav:
            continue
        out.setdefault((h.direction, h.alpha), []).append(h)
    agg = {}
    for key, rows in out.items():
        def col(attr):
            v = [getattr(r, attr) for r in rows if getattr(r, attr)]
            if not v:
                return (0.0, 0.0, 0)
            return (_st.median(v), max(v) - min(v), len(v))
        agg[key] = {
            "n": len(rows),
            "cos_to_base": rows[0].cos_to_base,
            "beyond": rows[0].beyond,
            "cos_render": col("cos_render"),
            "f0_median_hz": col("f0_median_hz"),
            "f0_range_st": col("f0_range_st"),
            "rate_wps": col("rate_wps"),
            "energy_db": col("energy_db"),
        }
    return agg


def noise(heard: list) -> dict:
    """
    How much two renders of the SAME vector differ.

    Taken from the alpha = 0 cells, pooled across directions. Every one of
    those is the base embedding rendered again -- identical vector, identical
    words -- so the spread among them is the engine's sampling and nothing
    else. It is a free control: the sweep had to render them anyway as the
    baseline, and they double as the yardstick every other cell is judged by.

    Returns {metric: (sd, n)}. With a handful of draws the sd is itself rough,
    which is why what it feeds is a screening rule and not a test.
    """
    import statistics as _st
    zero = [h for h in heard if h.alpha == 0.0 and h.wav]
    out = {}
    for k in ("cos_render", "f0_median_hz", "f0_range_st", "rate_wps",
              "energy_db"):
        v = [getattr(h, k) for h in zero if getattr(h, k)]
        out[k] = ((_st.stdev(v) if len(v) > 1 else 0.0), len(v))
    return out


def moved(heard: list) -> dict:
    """Did the round trip respond to alpha at all, per direction?

    The question the whole sweep exists to answer, reduced to one number per
    direction: the spread of the CELL MEDIANS of `cos_render` across the alphas
    tried. Medians rather than raw draws, so one unlucky render cannot create
    a response that is not there. A spread near zero means the engine gave back
    the same speaker whatever it was handed, which is a null result about the
    PLUMBING and must not be read as a null result about emotion vectors.
    """
    agg = cells(heard)
    out = {}
    for name in sorted({d for d, _ in agg}):
        vals = [v["cos_render"][0] for (d, _), v in agg.items()
                if d == name and v["cos_render"][0]]
        out[name] = round(max(vals) - min(vals), 5) if len(vals) > 1 else 0.0
    return out


#: Below this spread in the round-trip cosine, the renders are not responding
#: to the vector. Set at a tenth of TOO_CLOSE_COS's margin: if alpha swinging
#: from -0.5 to +1.4 moves the re-extracted embedding less than half a
#: thousandth, nothing is getting through.
RESPONSE_FLOOR = 0.0005

#: How many noise standard deviations a swing must clear before the sweep will
#: call it an effect. Two, which is a screening rule and not a test -- with
#: three to nine draws the sd is itself uncertain, and the honest use of this
#: number is to stop a non-effect being described as one, not to certify the
#: ones that pass.
CREDIBLE_SD = 2.0


def sweep_report(heard: list, ts: TauSet | None = None) -> str:
    if not heard:
        return "nothing rendered"
    agg, nz = cells(heard), noise(heard)
    lines = [f"{'direction':<11} {'alpha':>6} {'n':>2} {'cos handed':>10} "
             f"{'cos render':>16} {'pitch Hz':>15} {'rate':>13} {'loud dB':>14}",
             "-" * 94]
    for (name, a) in sorted(agg, key=lambda k: (k[0], k[1])):
        v = agg[(name, a)]
        def cell(k, fmt):
            m, sp, _ = v[k]
            return f"{m:{fmt}} ±{sp:{fmt}}"
        lines.append(
            f"{name:<11} {a:+6.2f} {v['n']:2d} {v['cos_to_base']:10.4f} "
            f"{cell('cos_render', '.4f'):>16} {cell('f0_median_hz', '5.1f'):>15} "
            f"{cell('rate_wps', '.2f'):>13} {cell('energy_db', '5.1f'):>14}"
            + ("  BEYOND" if v["beyond"] else ""))
    lines += ["",
              "median of n draws, ± the full range across them.",
              "",
              f"NOISE, from the alpha = 0 cells — the same vector rendered "
              f"{nz['f0_median_hz'][1]} times:"]
    for k, lab, fmt in (("f0_median_hz", "pitch", "5.2f"),
                        ("rate_wps", "rate", "5.3f"),
                        ("energy_db", "loudness", "5.2f"),
                        ("cos_render", "cos render", "7.5f")):
        lines.append(f"  {lab:<11} sd {nz[k][0]:{fmt}}")
    if nz["f0_median_hz"][1] < 3:
        lines.append("  (too few draws to estimate this; run with --repeat 3 "
                     "or more)")

    spread = moved(heard)
    lines += ["", "round-trip response (how far the re-extracted embedding "
                  "moved across alpha):"]
    for name, sp in spread.items():
        verdict = "NOTHING GOT THROUGH" if sp < RESPONSE_FLOOR else "responds"
        lines.append(f"  {name:<11} {sp:.5f}   {verdict}")

    lines += ["", "does alpha change the delivery, or is it the engine "
                  "sampling?"]
    for name in sorted({d for d, _ in agg}):
        for k, lab in (("f0_median_hz", "pitch"), ("rate_wps", "rate"),
                       ("energy_db", "loudness")):
            meds = [v[k][0] for (d, _), v in agg.items() if d == name and v[k][0]]
            if len(meds) < 2:
                continue
            swing, sd = max(meds) - min(meds), nz[k][0]
            if sd <= 0:
                lines.append(f"  {name:<11} {lab:<9} swing {swing:7.2f}   "
                             f"no noise estimate")
                continue
            ratio = swing / sd
            lines.append(
                f"  {name:<11} {lab:<9} swing {swing:7.2f}  = {ratio:5.1f} x "
                f"noise   "
                + ("clears the noise" if ratio >= CREDIBLE_SD
                   else "WITHIN THE NOISE"))
    if any(sp < RESPONSE_FLOOR for sp in spread.values()):
        lines += ["",
                  "! A direction whose round trip does not move is a broken "
                  "connection, not a",
                  "  finding about emotion. Check that the adapter is the "
                  "in-process qwen3 one",
                  "  and that `set_expression` is reaching the synthesis call "
                  "before judging",
                  "  anything by ear."]
    lines += ["",
              "A swing that clears the noise says alpha CHANGED something. It "
              "does not say the",
              "change was the one the register was named for, and it does not "
              "say the mapping",
              "is orderly — read down the column and see whether the numbers "
              "move monotonically",
              "with alpha. A swing that is large but jumps about is the engine "
              "reacting to a",
              "perturbation, not a dial.",
              "",
              "BEYOND marks a render whose embedding sits further from your "
              "neutral take than",
              "any two of your own recorded takes sit from each other. Not "
              "necessarily bad --",
              "but it is extrapolation, and the ear is the only judge left out "
              "there."]
    return "\n".join(lines)


def contact_sheet(heard: list, out: Path, gap_s: float = 0.5):
    """Every render in one file, in the order printed, no spoken labels."""
    import numpy as np
    from realme.verify.acoustic import read_wav
    from realme.expressive.wavio import write_wav, resample
    pieces, rate = [], None
    for h in heard:
        if not h.wav or not Path(h.wav).is_file():
            continue
        y, sr = read_wav(Path(h.wav))
        if rate is None:
            rate = sr
        elif sr != rate:
            y = resample(y, sr, rate)
        pieces.append(np.asarray(y, dtype=np.float64))
        pieces.append(np.zeros(int((rate or 24000) * gap_s)))
    if not pieces:
        return None
    return write_wav(Path(out), np.concatenate(pieces), rate or 24000)
