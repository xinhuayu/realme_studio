#!/usr/bin/env python3
"""
How much text can we hand Gemini TTS in ONE call -- measured, not read off a
specification.

Two numbers get confused with each other, and only one of them matters:

  * The API limit. Documented: 8,192 input tokens, 16,384 output tokens --
    which at the ~32 audio tokens per second these models bill is about eight
    and a half minutes of speech per request. A 200-word slide note is roughly
    260 input tokens and 90 seconds of audio. It is not close to the limit.
  * The QUALITY limit. Autoregressive speech models drop sentences, repeat
    clauses, drift in rate and -- for a cloned voice, which is the case we
    care about -- drift in identity, long before they hit the API limit.
    `realme/pipeline/speak.py` chunks at ~260 characters for exactly this
    reason, and cites the long-form benchmark that found models advertising
    single-pass generation degrading worse than chunked alternatives.

This script measures the second one. The method is the only one available
without a transcriber: synthesize each sentence ALONE (short calls, the
regime engines are reliable in), add up the durations, then synthesize the
same sentences TOGETHER in one call and compare. Audio that comes back
materially shorter than the sum of its parts has lost text. Audio materially
longer has stalled or repeated. The size at which that starts is the answer.

It is a screen, not a proof: every WAV is kept, numbered, so the two or three
sizes that matter can be listened to. Listening is the proof.

    python 03_App\\probe_gemini_tts.py --prebuilt Kore
    python 03_App\\probe_gemini_tts.py              # your enrolled voice

Costs about twenty cents at current prices; it prints the estimate and the
actual before and after.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The rate and the token-per-second figure come from the adapter, which is
#: also what the renderer bills against. Two copies of a price is how a probe
#: comes to quote one number and a render log another.
from realme.adapters.tts_gemini import (  # noqa: E402
    PRICE_PER_MTOK, TOKENS_PER_SECOND)

#: Twelve sentences of the kind of prose this tool actually narrates: defined
#: terms, numbers with decimal points, an abbreviation, a parenthesis, a
#: question. Deliberately not a tongue-twister -- the failure being looked for
#: is dropped text over length, and ordinary text finds it honestly.
FALLBACK = """\
Confounding is the central problem of observational epidemiology, and it is \
worth being precise about what it is. A confounder is a common cause of both \
the exposure and the outcome, which is not the same thing as a variable that \
happens to be associated with both. The distinction matters because adjusting \
for the wrong variable can introduce bias rather than remove it. Consider a \
cohort study of coffee drinking and lung cancer, where smoking is more common \
among coffee drinkers. If we do not adjust for smoking, the crude relative \
risk of 2.4 is almost entirely an artefact of that imbalance. After \
stratification by smoking status, the adjusted estimate falls to 1.05, with a \
confidence interval that comfortably includes the null. This is the classic \
demonstration, and e.g. Rothman's textbook presents it in essentially this \
form. Now compare that with a variable on the causal pathway, such as a \
biomarker that the exposure itself raises. Adjusting for a mediator removes \
part of the very effect we set out to measure, so the estimate shrinks for a \
reason that has nothing to do with bias. How, then, should we decide which \
variables to adjust for? The honest answer is that the data cannot tell us; \
the causal diagram has to be drawn first, from substantive knowledge, and \
only then does the arithmetic have a meaning worth reporting.\
"""


def sentences(text: str) -> list[str]:
    """Split with the project's own splitter, so the probe measures the same
    units the renderer would actually send."""
    from realme.pipeline import prosody
    return [u for u in prosody.split_delivery_units(text, mode="sentence") if u]


def narration_from_projects() -> tuple[str, str]:
    """The newest project's narration, so the measurement is on HIS prose."""
    try:
        from realme.core.env import data_home
        from realme.core.textio import read_text
    except Exception:
        return "", ""
    root = data_home() / "projects"
    if not root.is_dir():
        return "", f"!{root}"
    best, when = None, 0.0
    # Both depths: a project is normally <projects>/<name>/manifest.json, but
    # a lecture split into parts nests one level further.
    for man in list(root.glob("*/manifest.json")) + list(
            root.glob("*/*/manifest.json")):
        try:
            m = man.stat().st_mtime
        except OSError:
            continue
        if m > when:
            best, when = man, m
    if best is None:
        return "", f"!{root}"
    try:
        data = json.loads(read_text(best))
    except Exception:
        return "", f"!{root}"
    parts = []
    for seg in (data.get("segments") or []):
        t = (seg.get("narration") or seg.get("text") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts), best.parent.name


def money(model: str, seconds: float, metered: int | None) -> tuple[float, str]:
    rate = PRICE_PER_MTOK.get(model, 9.0)
    if metered:
        return metered / 1e6 * rate, "metered"
    return seconds * TOKENS_PER_SECOND / 1e6 * rate, "estimated"


def verdict_for(speech_ratio: float, syllable_ratio: float | None = None) -> str:
    """What a rendering did, from two measurements rather than one.

    The first version of this compared WALL durations and called anything 8%
    short "text was probably dropped". Both halves of that were wrong, and the
    person listening to the files caught it before the arithmetic did.

      * Wall duration counts the lead-in and tail silence of every call, so a
        long call is compared against N copies of that padding -- about 0.7 s
        each, which is most of the apparent shortfall on its own. Speaking
        time is the comparable quantity.
      * A shortfall in speaking time with the SYLLABLES still there is not a
        drop, it is a hurry. Measured on the run that prompted this: 1345
        characters came back 9% short of speech time while carrying 92% of the
        syllables and sounding, to the listener, complete but rushed.

    So the content question and the pace question are answered separately.
    Syllable counting undercounts slightly when speech is fast -- nuclei merge
    -- so the drop threshold sits at 0.88, well below one missing sentence in
    seven (0.86) and well clear of the 0.92 that brisk delivery produces.
    """
    if syllable_ratio is not None and syllable_ratio < 0.88:
        return (f"DROPPED -- only {syllable_ratio * 100:.0f}% of the content "
                f"came back")
    if speech_ratio > 1.15:
        return f"LONG by {(speech_ratio - 1) * 100:.0f}% -- stalled or repeated"
    if speech_ratio < 0.93:
        return (f"RUSHED by {(1 - speech_ratio) * 100:.0f}% -- all there, said "
                f"faster than one sentence at a time")
    return "intact"


def is_intact(r: dict) -> bool:
    """Complete AND not hurried. Both, because either one spoils a lecture."""
    if not r.get("ok") or str(r.get("tag", "")).startswith("ceiling"):
        return False
    sp, sy = r.get("speech_ratio"), r.get("syllable_ratio")
    if not sp:
        return False
    if sy is not None and sy < 0.88:
        return False
    return 0.93 <= sp <= 1.15


def metered_tokens(usage: dict) -> int | None:
    if not usage:
        return None
    for k in ("candidates_token_count", "candidatesTokenCount",
              "output_token_count", "outputTokenCount",
              "total_token_count", "totalTokenCount"):
        v = usage.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="",
                    help="voice_... / voicekey_... (default: the enrolled one)")
    ap.add_argument("--prebuilt", default="",
                    help="A prebuilt voice name such as Kore, to measure the "
                         "model before enrolling your own")
    ap.add_argument("--model", default="")
    ap.add_argument("--text", type=Path,
                    help="A .txt/.md of your own narration to measure on")
    ap.add_argument("--sizes", default="0,300,600,900,1200,2000,4000",
                    help="Target characters per call. 0 means one sentence.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where the WAVs go (default: ./gemini_probe)")
    ap.add_argument("--no-refine", dest="refine", action="store_false",
                    help="Skip the bisection between the last size that held "
                         "and the first that did not")
    ap.add_argument("--ceiling", action="store_true",
                    help="Also find the HARD limit by growing the request "
                         "until the API refuses, and print its exact words")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and the estimated cost, call nothing")
    a = ap.parse_args()

    from realme.adapters.tts_gemini import GeminiTTS, DEFAULT_MODEL, MODELS
    from realme.adapters.base import AdapterUnavailable
    from realme.core.media import duration_of
    from realme.enrollment.pace import speech_profile

    out = (a.out or Path.cwd() / "gemini_probe").resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- the text -------------------------------------------------------
    if a.text:
        from realme.core.textio import read_text
        body, source = read_text(a.text), str(a.text)
    else:
        body, source = narration_from_projects()
        if not body:
            # Say where it looked. "built-in sample" on its own reads as a
            # choice; it is usually a search that found nothing, and the
            # path is what tells you why.
            looked = source[1:] if source.startswith("!") else ""
            body = FALLBACK
            source = ("built-in sample"
                      + (f" (no projects found in {looked})" if looked else ""))
        else:
            source = f"your project '{source}'"
    sents = sentences(body)
    if len(sents) < 4:
        print("That text has too few sentences to measure anything.")
        return 1

    model = a.model or os.environ.get("REALME_GEMINI_TTS_MODEL", DEFAULT_MODEL)
    if model not in MODELS:
        print(f"{model} is not a Gemini TTS model: {', '.join(MODELS)}")
        return 1

    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    # Each ladder rung is a PREFIX of the same sentences, so every rung is
    # compared against sentences that were also measured individually. A rung
    # built from different text would compare two things at once.
    rungs = []
    for want in sizes:
        take, n = "", 0
        for s in sents:
            if take and want and len(take) + 1 + len(s) > want:
                break
            take = f"{take} {s}".strip()
            n += 1
            if not want:            # 0 = a single sentence
                break
        if n and (not rungs or rungs[-1][1] != n):
            rungs.append((take, n))

    needed = max(n for _, n in rungs)
    calib = sents[:needed]
    cal_chars = sum(len(s) for s in calib)
    lad_chars = sum(len(t) for t, _ in rungs)
    # ~15 characters a second of speech is the usual rate for this prose; used
    # ONLY for the up-front estimate, and replaced by measurement immediately.
    est_s = (cal_chars + lad_chars) / 15.0
    est_cost, _ = money(model, est_s, None)

    kw = {"model": model}
    if a.prebuilt:
        kw["voice"] = a.prebuilt
    elif a.voice:
        kw["voice"] = a.voice
    tts = GeminiTTS(**kw)

    # Where the key came from, named. A standalone script does not load .env
    # the way the Studio and the CLI do, and the first run of this said "No
    # GEMINI_API_KEY" to someone who had one -- in the right place.
    try:
        from realme.core.env import load as _load_env
        used = [str(p_) for p_ in _load_env()]
    except Exception:
        used = []
    have = bool(os.environ.get("GEMINI_API_KEY"))

    print(f"model      : {model}")
    print(f"voice      : {tts.voice or '(none enrolled)'}")
    print(f"api key    : {'found' if have else 'MISSING'}"
          + (f" (from {used[0]})" if used and have else ""))
    if tts.rate_match and abs(tts.rate_match - 1.0) > 1e-3:
        print(f"pace       : a lecture is retimed by {tts.rate_match:.3f} "
              f"after the call; the takes measured below are not")
    print(f"text       : {source} -- {len(sents)} sentences, "
          f"{len(body.split())} words")
    print(f"calibration: {len(calib)} sentences, one call each")
    print(f"ladder     : {len(rungs)} calls at "
          f"{', '.join(str(len(t)) for t, _ in rungs)} characters")
    print(f"estimate   : ~{est_s/60:.1f} min of audio, about ${est_cost:.2f}")
    print(f"wavs       : {out}")
    if a.dry_run:
        print("\n(dry run -- nothing was called)")
        return 0
    print()

    if a.prebuilt:
        # A prebuilt voice is not a clone of anyone, so the tier question --
        # which is about YOUR voice and YOUR lecture text leaving the machine
        # -- is still live for the text but not for the biometrics. The check
        # is not skipped; it is answered honestly for this one run.
        tts.voice = a.prebuilt
    try:
        tts.check()
    except AdapterUnavailable as e:
        print(f"Cannot run: {e}")
        return 1

    rows: list[dict] = []
    spend = 0.0

    def call(text: str, tag: str) -> dict | None:
        nonlocal spend
        wav = out / f"{tag}.wav"
        t0 = time.perf_counter()
        try:
            tts.synthesize(text, wav)
        except Exception as e:                    # the refusal IS the finding
            print(f"  {tag:<14} REFUSED  {str(e)[:160]}")
            return {"tag": tag, "chars": len(text), "ok": False,
                    "error": str(e)[:400]}
        wall = time.perf_counter() - t0
        dur = duration_of(wav)
        prof = speech_profile(wav)
        tok = metered_tokens(tts.last_usage)
        cost, how = money(model, dur, tok)
        spend += cost
        print(f"  {tag:<14} {len(text):>5} ch  {dur:>6.2f}s  "
              f"{prof['speech_s']:>6.2f}s talking  "
              f"{prof['articulation']:>4.2f} syl/s"
              f"{'  ' + str(tok) + ' tok' if tok else ''}")
        return {"tag": tag, "chars": len(text), "ok": True, "duration_s": dur,
                "speech_s": round(prof["speech_s"], 2),
                "syllables": prof["syllables"],
                "articulation": round(prof["articulation"], 2),
                "edge_silence_s": round(prof["lead_s"] + prof["tail_s"], 2),
                "wall_s": round(wall, 2), "tokens": tok, "token_source": how,
                "cost": cost, "wav": str(wav)}

    print("calibration -- one sentence per call, the regime engines are "
          "reliable in")
    per: list[dict] = []
    for i, s in enumerate(calib):
        r = call(s, f"cal_{i:02d}")
        if r:
            per.append(r)
    if not all(r.get("ok") for r in per):
        print("\nA single sentence failed. Nothing further is worth "
              "measuring until that is sorted out.")
        return 1
    cal_total = sum(r["duration_s"] for r in per)
    edge = sum(r["edge_silence_s"] for r in per) / max(len(per), 1)
    print(f"  -> {len(per)} sentences, {cal_total:.2f}s of audio, "
          f"{sum(r['speech_s'] for r in per):.2f}s of it talking")
    print(f"  -> {edge:.2f}s of lead-in and tail silence on each, which is why "
          f"speaking time\n     and not wall time is what gets compared below")

    def score(r, n):
        """Fill in the two ratios that decide the verdict."""
        exp_speech = sum(x["speech_s"] for x in per[:n])
        exp_syl = sum(x["syllables"] for x in per[:n])
        r["sentences"] = n
        r["expected_speech_s"] = round(exp_speech, 2)
        r["expected_syllables"] = exp_syl
        if r.get("ok") and exp_speech and exp_syl:
            r["speech_ratio"] = round(r["speech_s"] / exp_speech, 3)
            r["syllable_ratio"] = round(r["syllables"] / exp_syl, 3)
            r["verdict"] = verdict_for(r["speech_ratio"], r["syllable_ratio"])
            print(f"  {'':<14} {n:>3} sentences: speaking time "
                  f"{r['speech_ratio']:.3f}x, content "
                  f"{r['syllable_ratio']:.3f}x  {r['verdict']}")
        return r

    print("\nladder -- the same sentences, fewer and larger calls")
    print("  speaking time says whether it hurried; syllables say whether it "
          "left anything out")
    for text, n in rungs:
        r = call(text, f"size_{len(text):05d}")
        if r:
            rows.append(score(r, n))

    if a.ceiling:
        print("\nceiling -- growing the request until the API refuses")
        unit = " ".join(sents)
        for mult in (2, 4, 8, 16, 32):
            text = (unit + " ") * mult
            r = call(text[:60000], f"ceiling_x{mult}")
            rows.append(r or {})
            if not r or not r.get("ok"):
                break

    intact = is_intact

    # The ladder brackets the answer; it does not find it. A rung at 839 that
    # holds and one at 1158 that does not says only "somewhere in between",
    # and the gap is wider than the thing being measured. Bisect by SENTENCE
    # COUNT, so every trial is still a prefix with a known expected duration.
    if a.refine:
        held = max((r["sentences"] for r in rows if intact(r)), default=0)
        broke = min((r["sentences"] for r in rows
                     if r.get("sentences") and not intact(r)), default=0)
        if held and broke and broke - held > 1:
            print(f"\nrefining -- it held at {held} sentences and did not at "
                  f"{broke}")
            for _ in range(3):
                mid = (held + broke) // 2
                if mid <= held or mid >= broke:
                    break
                text = " ".join(sents[:mid])
                r = call(text, f"refine_{len(text):05d}")
                if not r:
                    break
                rows.append(score(r, mid))
                if intact(r):
                    held = mid
                else:
                    broke = mid

    good = [r for r in rows if intact(r)]
    largest = max((r["chars"] for r in good), default=0)

    # One call is an anecdote. These models are sampled, so the size that
    # matters is the one that holds TWICE -- and a second trial costs a cent.
    confirmed = None
    if largest:
        winner = max(good, key=lambda r: r["chars"])
        again = call(" ".join(sents[:winner["sentences"]]), "confirm")
        if again and again.get("ok"):
            score(again, winner["sentences"])
            confirmed = intact(again)
            rows.append(again)
            print(f"  {'':<14} second take at {winner['chars']} characters: "
                  f"{'holds' if confirmed else 'DID NOT hold'}")

    print(f"\nspent about ${spend:.2f}")
    print("\n--------------------------------------------------------------")
    # The pace trend is a finding in its own right. A size that is "rushed"
    # is not unusable -- the hurry is uniform and the renderer already applies
    # a measured retime -- so report the correction it would need rather than
    # only ruling the size out.
    print("\nhow the pace moved with the size of the call:")
    for r in sorted((x for x in rows if x.get("speech_ratio")),
                    key=lambda x: x["chars"]):
        pace = 1.0 / r["speech_ratio"]
        print(f"  {r['chars']:>5} ch  {r['articulation']:>4.2f} syl/s  "
              f"{'faster' if pace > 1 else 'slower'} than one-at-a-time by "
              f"{abs(pace - 1) * 100:>4.1f}%   {r['verdict']}")
    if rows:
        slowest = min((x for x in rows if x.get("speech_ratio")),
                      key=lambda x: x["articulation"], default=None)
        if slowest:
            print(f"  the most deliberate delivery was at {slowest['chars']} "
                  f"characters")

    if largest and confirmed is False:
        print(f"{largest} characters held once and not twice. These models are "
              f"sampled,\nso that is a size that works sometimes, which is the "
              f"worst kind.\nDrop to the next rung down, or keep the pipeline "
              f"default.")
    elif largest:
        best = max(good, key=lambda r: r["chars"])
        print(f"Largest call that came back intact, twice: {largest} "
              f"characters (~{largest//6} words).")
        corr = 1.0 / best["speech_ratio"] if best.get("speech_ratio") else 1.0
        if abs(corr - 1) >= 0.02:
            print(f"At that size it still speaks {abs(corr-1)*100:.0f}% "
                  f"{'faster' if corr > 1 else 'slower'} than it does one "
                  f"sentence at a time.\nThat is uniform, so it is a retime, "
                  f"not a reason to go smaller:\n"
                  f"  realme voice gemini --pace\n"
                  f"measures it against your own recording and stores the "
                  f"correction.")
        print("Listen to that one and the one above it before trusting the "
              "number:")
        print(f"  {out}")
        print("\nIf it sounds right, make it the engine's chunk size:")
        print(f"  realme voice gemini --max-chars {largest}")
        print("Nothing changes until you do -- the renderer keeps using the "
              "pipeline default (260) until this is set.")
    else:
        print("Nothing above one sentence came back intact. Keep the pipeline "
              "default.")
    print("--------------------------------------------------------------")

    report = out / "probe.json"
    report.write_text(json.dumps(
        {"model": model, "voice": tts.voice, "source": source,
         "calibration": per, "ladder": rows, "spend_usd": round(spend, 4),
         "largest_intact_chars": largest,
         "tokens_per_second_assumed": TOKENS_PER_SECOND,
         "when": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2), encoding="utf-8")
    print(f"\nFull numbers: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
