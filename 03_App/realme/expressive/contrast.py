"""
Does this exchange have dynamics, or is it two flat voices taking turns?

The measurement that matters here is not how expressive any single turn is. A
conversation sounds personal because each turn is audibly a RESPONSE to the one
before it -- leaning in after a concession, softening after a sharp question --
so the quantity of interest is contrast BETWEEN turns.

Two things have to be kept apart, and conflating them would make the metric
useless in opposite directions.

WITHIN a turn, pitch movement is mostly instability. The steadiness score this
project already uses was built to catch exactly that: a cloned voice whose
pitch wanders inside one utterance sounds broken, not expressive.

BETWEEN turns, the same movement is intent. A press that sits a tone above a
concession is the thing we are trying to produce.

So a single "variability" number scores an improvement and a defect
identically. They are measured separately and reported separately, and the
ratio between them is a screening number rather than a verdict.

THE CONFOUND. In a two-speaker dialogue, the largest between-turn pitch
difference is that the two speakers are different people. Measured across all
turns, an exchange between a low male voice and a higher one scores as
enormously dynamic while both read from a phone book. Every between-turn
quantity here is therefore computed WITHIN a speaker and then averaged across
speakers -- never across the whole transcript.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict

#: Below this many turns from one speaker, a between-turn spread is noise.
#: Three gives two intervals, which is the fewest that can show a pattern
#: rather than a single step.
MIN_TURNS_PER_SPEAKER = 3

#: Floor on the within-turn figure when forming the ratio, in semitones.
#: Without it a perfectly steady synthetic voice divides by ~0 and the
#: separation ratio reports a triumph that is really a measurement artefact.
WITHIN_FLOOR_ST = 0.25


def _st(hz_a: float, hz_b: float) -> float:
    if hz_a <= 0 or hz_b <= 0:
        return 0.0
    return 12.0 * math.log2(hz_a / hz_b)


def _spread(values: list[float]) -> float:
    """Population standard deviation. Zero for fewer than two values."""
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


@dataclass
class SpeakerDynamics:
    speaker: str
    turns: int = 0
    within_pitch_st: float = 0.0     # median spread INSIDE a turn: instability
    between_pitch_st: float = 0.0    # spread of turn medians: intent
    adjacent_pitch_st: float = 0.0   # mean step from one turn to the next
    between_rate_pct: float = 0.0    # spread of speaking rate, % of own mean
    adjacent_rate_pct: float = 0.0
    between_energy_db: float = 0.0
    lead_pause_spread_s: float = 0.0
    separation: float = 0.0          # between / within, screening only
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Dynamics:
    speakers: list[SpeakerDynamics] = field(default_factory=list)
    measured_turns: int = 0
    skipped_turns: int = 0

    def summary(self) -> str:
        if not self.speakers:
            return "no turns could be measured"
        bits = []
        for s in self.speakers:
            bits.append(f"{s.speaker}: within {s.within_pitch_st:.2f} st, "
                        f"between {s.between_pitch_st:.2f} st "
                        f"(x{s.separation:.1f}), rate +-{s.between_rate_pct:.0f}%")
        tail = (f"; {self.skipped_turns} turn(s) unmeasurable"
                if self.skipped_turns else "")
        return " | ".join(bits) + tail

    def as_dict(self) -> dict:
        return {"speakers": [s.as_dict() for s in self.speakers],
                "measured_turns": self.measured_turns,
                "skipped_turns": self.skipped_turns}


def dynamics(measured: list[tuple[str, object]]) -> Dynamics:
    """
    `measured` is [(speaker, TurnProsody), ...] in the order spoken.

    Turns that could not be measured are counted and dropped rather than
    contributing a zero: a failed pitch track entered as 0 Hz would look like
    the most dramatic interval in the conversation.
    """
    out = Dynamics()
    by_speaker: dict[str, list] = {}
    for speaker, p in measured:
        if not getattr(p, "ok", False) or p.f0_median_hz <= 0:
            out.skipped_turns += 1
            continue
        out.measured_turns += 1
        by_speaker.setdefault(speaker or "?", []).append(p)

    for speaker, turns in sorted(by_speaker.items()):
        d = SpeakerDynamics(speaker=speaker, turns=len(turns))
        withins = sorted(t.f0_spread_st for t in turns)
        mid = len(withins) // 2
        d.within_pitch_st = (withins[mid] if len(withins) % 2
                             else (withins[mid - 1] + withins[mid]) / 2)
        if len(turns) < MIN_TURNS_PER_SPEAKER:
            d.note = (f"only {len(turns)} turn(s); between-turn figures need at "
                      f"least {MIN_TURNS_PER_SPEAKER}")
            out.speakers.append(d)
            continue

        # Everything below is WITHIN this speaker, never across speakers.
        ref = sorted(t.f0_median_hz for t in turns)[len(turns) // 2]
        rel_st = [_st(t.f0_median_hz, ref) for t in turns]
        d.between_pitch_st = _spread(rel_st)
        d.adjacent_pitch_st = (sum(abs(b - a) for a, b in zip(rel_st, rel_st[1:]))
                               / (len(rel_st) - 1))

        rates = [t.rate_wps for t in turns if t.rate_wps > 0]
        if len(rates) >= 2:
            mean_rate = sum(rates) / len(rates)
            d.between_rate_pct = 100.0 * _spread(rates) / mean_rate
            d.adjacent_rate_pct = (100.0 * sum(abs(b - a) for a, b in
                                               zip(rates, rates[1:]))
                                   / (len(rates) - 1) / mean_rate)
        d.between_energy_db = _spread([t.energy_db for t in turns])
        d.lead_pause_spread_s = _spread([t.lead_silence_s for t in turns])
        d.separation = d.between_pitch_st / max(d.within_pitch_st, WITHIN_FLOOR_ST)
        out.speakers.append(d)
    return out
