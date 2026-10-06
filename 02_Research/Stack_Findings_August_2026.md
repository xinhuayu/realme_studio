# Component landscape — verified August 2026

Every claim below was checked against primary sources (official docs, actual
LICENSE files, repo READMEs) rather than comparison articles. Items marked
*unverified* could not be confirmed from a primary source.

## Talking-head / avatar (deferred, but for when you return to it)

The field splits in a way most comparisons blur:

- **Lip-sync / inpainting** (Wav2Lip, MuseTalk, LatentSync) needs an existing
  *driving video* and repaints the mouth. Given a static portrait you get a
  frozen body with a moving mouth. Wrong tool for a lecture.
- **Audio-driven portrait animation** (EchoMimicV3, InfiniteTalk, Hallo2) makes
  motion from a single image. This is what a digital twin needs.

LivePortrait is *neither* — it is video/pose-driven retargeting.

| Model | Licence | VRAM | Notes |
|---|---|---|---|
| **EchoMimicV3** | Apache 2.0 | 12 GB | Most active; AAAI 2026; Flash Pro Jan 2026. **Recommended.** |
| InfiniteTalk | Apache 2.0 | low-VRAM + fp8 | Unlimited length, image-to-video |
| LongCat-Video-Avatar-1.5 | MIT | ≥2 GPUs | Newest (May 2026) |
| Hallo2 | MIT | A100 tested | Purpose-built for 1hr @ 4K — but dormant since Jan 2025 |

**Licence landmines:** Wav2Lip is personal/research only (LRS2 training data).
Sonic is CC BY-NC-SA and needs 32 GB. LivePortrait's MIT code depends on
InsightFace models that are research-only — this contaminates many pipelines,
so audit the whole dependency tree. SadTalker is obsolete (README news ends
June 2023).

## Voice

| Model | Licence | Clones? | Verdict |
|---|---|---|---|
| **Chatterbox** (Resemble) | **MIT** | ~5–10s ref | Best licence/quality combination. Turbo variant 6× realtime; Nano runs on CPU. |
| IndexTTS-2.5 | bilibili licence | yes | Commercial use permitted below 100M MAU / ¥1B revenue — a university qualifies. Strong emotion control. |
| F5-TTS | code MIT / **weights CC-BY-NC** | yes | **Blocked** — the weights are what you ship |
| Fish Speech | research licence | yes | **Blocked** for commercial |
| XTTS-v2 (Coqui) | CPML — licence URL now 404s | yes | Avoid; company status unclear |
| Kokoro | Apache 2.0 | **no** | Fixed voicepacks; cannot be your voice |

**Hosted**, at ~54,000 characters per hour of narration:

| Service | Cost/hr | Notes |
|---|---|---|
| Google Chirp 3 HD | ~$1.62 | no cloning |
| **Google Chirp 3 Instant Custom Voice** | ~$3.24 | ~10s reference + verbatim spoken consent; **allowlist-gated via sales** |
| Gemini native TTS | ~$0.90–1.80 | 2 speakers max, **no cloning**, 32k session context with documented quality drift on long content |
| ElevenLabs | ~$9 (Pro) | Creator $22/mo unlocks professional cloning; API-grade audio gated at $99 |
| Cartesia | Pro $4/mo, ~133 min | Cheapest credible commercial-licensed cloning |

## Google first-party avatar: nothing fits

Veo 3.1 caps at 8s clips (extension to ~148s, forced to 720p), and its audio is
**generated and always on** — there is no way to supply your cloned voice as the
driving track. Gemini Omni Flash has the same limitation and would run ~$180 for
a 30-minute lecture. No first-party lip-sync entry found in Vertex Model Garden
(*partially unverified* — the catalog needs console auth to enumerate fully).

## Commercial avatar APIs, if you ever want to skip the GPU

**HeyGen** has the clearest pricing and the most rigorous consent process
(sub-30s consent video, exact provided text, submitted alongside the avatar
footage). Avatar III at $0.0167/sec ≈ **$30 per 30-minute lecture**; Avatar IV/V
≈ $120. Programmatic twin *creation* is Enterprise-gated; rendering is not.

**Synthesia** Creator is $89/mo with API — but its 360 minutes is **per year**,
about 12–18 lectures. Custom avatars are a $1,000/year add-on. D-ID, Argil and
Captions pricing could not be verified from primary sources.

## Slides → images + notes

**Do not use the Slides API `getThumbnail` endpoint.** It caps at 1600px wide
(below 1080p video needs), is an "expensive read" limited to 60 calls/min/user,
and its URLs expire in 30 minutes. A 60-slide deck exhausts your per-user
minute budget.

**Correct path:** Drive `files.export` → PDF (one request per deck, normal
quota, no expiry) → `pdftoppm -r <dpi>` at any resolution. Speaker notes come
separately from `presentations.get` → `slideProperties.notesPage` →
`notesProperties.speakerNotesObjectId`; note the docs warn the shape may not
exist on every slide. For PPTX, `soffice --headless --convert-to pdf` first.

## Apps Script, 2026

- **Workspace Events API for Drive went GA May 2026** — push notifications via
  Pub/Sub. This is the right primitive for a Drive-watching pipeline, replacing
  polling triggers entirely.
- Rhino runtime sunset 31 Jan 2026; unmigrated scripts stop executing.
- Quotas unchanged: **6 min per execution** (fatal to in-script rendering);
  trigger runtime 90 min/day on consumer Gmail vs 6 hours on Workspace.
