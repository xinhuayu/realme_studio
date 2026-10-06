"""
FFmpeg compositing. Every segment is rendered to a uniform encode so the concat
demuxer can stream-copy them; the master is therefore assembled with zero
re-encode and zero generational loss.
"""
from __future__ import annotations
from pathlib import Path
from realme.core.media import require, run, duration_of, probe, MediaError

V_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
          "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0"]
A_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def pad_audio(src: Path, dst: Path, tail_s: float) -> Path:
    """Append the inter-slide breath pause to the audio itself, so audio and
    video always describe the same duration. No drift is possible."""
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if tail_s <= 0:
        run([require("ffmpeg"), "-y", "-i", str(src), "-c", "copy", str(dst)], "copy audio")
    else:
        run([require("ffmpeg"), "-y", "-i", str(src),
             "-af", f"apad=pad_dur={tail_s}", "-c:a", "pcm_s16le", str(dst)],
            f"pad audio +{tail_s}s")
    return dst


def slide_rect(slide_png: Path, width: int, height: int) -> tuple[int, int, int, int]:
    """
    Where the slide actually lands inside the frame after letterboxing.

    Signalling needs this: a highlight box is expressed in page coordinates, and
    if you map it onto the full frame instead of the slide's letterboxed rect,
    every cue is offset by the size of the black bars.
    """
    from PIL import Image
    with Image.open(slide_png) as im:
        iw, ih = im.size
    scale = min(width / iw, height / ih)
    sw, sh = int(iw * scale), int(ih * scale)
    return (width - sw) // 2, (height - sh) // 2, sw, sh


def render_segment(slide_png: Path, audio: Path, out_mp4: Path,
                   width=1920, height=1080, fps=30,
                   avatar_mp4: Path | None = None, layout="slide_only",
                   cues: list | None = None) -> Path:
    """Render ONE slide segment. Duration is taken from the audio, always."""
    out_mp4 = Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    dur = duration_of(audio)

    fit = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")

    signal_chain = ""
    if cues:
        from realme.pipeline.signalling import filtergraph
        ox, oy, sw, sh = slide_rect(slide_png, width, height)
        signal_chain = filtergraph(cues, width, height, slide_w=sw, slide_h=sh,
                                   x_off=ox, y_off=oy)

    if avatar_mp4 and Path(avatar_mp4).exists() and layout != "slide_only":
        if layout == "side_by_side":
            sw = int(width * 0.68) // 2 * 2
            aw = width - sw
            fc = (f"[0:v]scale={sw}:{height}:force_original_aspect_ratio=decrease,"
                  f"pad={sw}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[l];"
                  f"[1:v]scale={aw}:{height}:force_original_aspect_ratio=increase,"
                  f"crop={aw}:{height},setsar=1[r];[l][r]hstack=inputs=2[v]")
        else:  # pip
            pw = int(width * 0.25) // 2 * 2
            fc = (f"[0:v]{fit}[bg];[1:v]scale={pw}:-2,setsar=1[pip];"
                  f"[bg][pip]overlay=W-w-40:H-h-40[v]")
        if signal_chain:
            fc = fc.replace("[v]", "[vsig]") + f";[vsig]{signal_chain}[v]"
        cmd = [require("ffmpeg"), "-y",
               "-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}", "-i", str(slide_png),
               "-stream_loop", "-1", "-t", f"{dur:.3f}", "-i", str(avatar_mp4),
               "-i", str(audio),
               "-filter_complex", fc, "-map", "[v]", "-map", "2:a",
               "-r", str(fps), "-t", f"{dur:.3f}", *V_ARGS, *A_ARGS, str(out_mp4)]
    else:
        vf = fit + ("," + signal_chain if signal_chain else "")
        cmd = [require("ffmpeg"), "-y",
               "-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}", "-i", str(slide_png),
               "-i", str(audio),
               "-vf", vf, "-r", str(fps), "-t", f"{dur:.3f}",
               *V_ARGS, *A_ARGS, str(out_mp4)]

    run(cmd, f"render segment {out_mp4.name}")
    got = duration_of(out_mp4)
    if abs(got - dur) > 0.25:
        raise MediaError(f"{out_mp4.name}: rendered {got:.2f}s but audio is {dur:.2f}s "
                         f"({got - dur:+.2f}s drift).")
    return out_mp4


def concat(segments: list[Path], out_mp4: Path) -> Path:
    out_mp4 = Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if not segments:
        raise MediaError("Nothing to concatenate.")
    listing = out_mp4.parent / f"_concat_{out_mp4.stem}.txt"
    listing.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in segments), encoding="utf-8")
    run([require("ffmpeg"), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", "-movflags", "+faststart", str(out_mp4)], "concat master")
    return out_mp4


def mux_subtitles(master: Path, srt: Path, out: Path) -> Path:
    """Soft subtitle track (mov_text). Selectable, not burned in."""
    out = Path(out)
    run([require("ffmpeg"), "-y", "-i", str(master), "-i", str(srt),
         "-map", "0", "-map", "1", "-c", "copy", "-c:s", "mov_text",
         "-metadata:s:s:0", "language=eng", str(out)], "mux subtitles")
    return out


def audio_master(parts: list[Path], out: Path, lufs: float = -16.0) -> Path:
    """Concatenate dialogue turns and loudness-normalize to podcast standard.

    The concat FILTER, not the demuxer. The demuxer takes the stream
    parameters from the first file, so a 22.05 kHz Piper guest following a
    24 kHz cloned instructor -- the default cast -- was decoded at 24 kHz:
    8% fast and a semitone sharp, and the transcript timestamps drifted with
    it. The filter resamples every input to one rate first.
    """
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise MediaError("audio_master: no turns to concatenate")
    raw = out.parent / f"_raw_{out.stem}.wav"
    inputs = []
    for p in parts:
        inputs += ["-i", str(Path(p).resolve())]
    chain = "".join(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono[a{i}];"
                    for i in range(len(parts)))
    fc = chain + "".join(f"[a{i}]" for i in range(len(parts))) \
        + f"concat=n={len(parts)}:v=0:a=1[out]"
    run([require("ffmpeg"), "-y", *inputs, "-filter_complex", fc, "-map", "[out]",
         "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", str(raw)], "concat turns")
    run([require("ffmpeg"), "-y", "-i", str(raw),
         "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11", "-c:a", "libmp3lame",
         "-b:a", "128k", str(out)], "loudness master")
    return out
