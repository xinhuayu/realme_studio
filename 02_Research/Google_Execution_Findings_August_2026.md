# Google execution options — verified August 2026

## Two findings that reshape the design

**1. Colab's ToS prohibits "creating deepfakes" — on paid tiers too.** The
disallowed-actions list in the [Colab FAQ](https://research.google.com/colaboratory/faq.html)
names it alongside "connecting to remote proxies" and "media serving… not
related to interactive compute". A Gradio `share=True` tunnel is a proxy by
construction. There is no published exemption for a consented self-likeness and
no appeal for a terminated runtime. Google has terminated free-tier sessions
running Stable Diffusion WebUIs before. **Do not host the avatar step here.**

**2. YouTube auto-publish is blocked by default, and worse than "unlisted".**
Videos uploaded via `videos.insert` from an unverified API project created after
28 July 2020 are restricted to **private**, and YouTube Help states you cannot
appeal. The fix is a compliance audit; the pragmatic answer for a dozen videos a
month is to upload private and flip visibility by hand.

## Colab

| Tier | Price | Compute units | Session |
|---|---|---|---|
| Free | $0 | none | ≤12 h, ~90 min idle |
| Pro | $9.99/mo | 100 CU | ≤24 h |
| Pro+ | $49.99/mo | 500 CU *(one source says 600 — unresolved)* | ≤24 h, **background execution** |

Google deliberately does not publish GPU allocation or burn rates. Third-party
observation (*unverified*): Free ≈ T4 only; Pro/Pro+ ≈ T4/L4/A100. Reported
burn: T4 ~1.19 CU/h, L4 ~1.71, A100-40 ~5.40. Treat with suspicion — the implied
A100 price is far below Compute Engine on-demand.

**Free Colab Pro for US higher education** was announced July 2025 as a one-year
offer. *Whether it is still open in Aug 2026 is unverified* — check the signup
page; it is the highest-value item here if live.

**What breaks a "1-click notebook":** GPU lottery (code sized for A100 OOMs on a
T4); VM filesystem wiped so multi-GB weights re-download each session; Drive
OAuth click cannot be scripted away; Drive FUSE times out on large directories;
base-image drift breaks pinned installs; closing the tab kills the run except on
Pro+; CU exhaustion silently demotes you to free-tier policy, where the web-UI
prohibition is actively enforced.

## Cloud Run GPU

L4 24 GB (no quota request needed since GA; 3 GPUs auto-allocated) and RTX PRO
6000 96 GB. Scale-to-zero supported. Cold start ~20s even with multi-GB weights
(block-level image streaming) — irrelevant next to a 10–40 min render.

**The 60-minute cap is the real constraint.** Services: 60 min per request.
Jobs: 168 h *"or 1 hour if using GPUs"*. Both paths cap GPU work at an hour.
Design for chunking: `--tasks N --parallelism K`, one task per ~5-minute
segment, then concatenate in a final CPU-only task which gets the full budget.

Cost, us-central1, L4 + 4 vCPU + 16 GiB, zonal redundancy off: **~$1.05/h**.

## Cheaper alternatives

- **Compute Engine `g2-standard-4` Spot: $0.424/h** — 60% below Cloud Run, no
  timeout cap. Costs: preemption (checkpoint per chunk) and owning the VM
  lifecycle (a self-deleting startup script, or it bills all month). Persistent
  disk at ~$0.10/GB-month can exceed the compute for light use — bake weights
  into a custom image instead.
- **Colab Enterprise / Vertex Workbench: skip.** ~17% management premium for
  IAM features a solo instructor doesn't need, and no Drive mounting.

## Credits

- Free trial **$300** ≈ 285 h of Cloud Run L4 or ~700 h of g2 Spot.
- **[Google Cloud Research Credits](https://edu.google.com/programs/credits/research/):
  up to $5,000 for faculty**, $1,000 for PhD students. Requires an accredited
  institution — *not available on a personal Gmail account*. Credits are
  non-commercial, expire 365 days from redemption, must be activated within 60
  days. $5,000 ≈ 11,800 hours of g2 Spot L4. **Apply first if you have a `.edu`
  address; it likely makes the cost question moot for a year.**

## Storage and delivery

Split them: **model weights → Cloud Storage** (same region, concurrent download
— Google explicitly recommends this over GCS FUSE above 10 GB, which cannot
parallelize). **Rendered video → Drive**, because GCS egress at $0.12/GB is a
real recurring cost (~$2.40/mo at 13 lectures) that Drive avoids. Caveat: Drive
API quota overages begin incurring charges later in 2026.

YouTube quota is a non-issue — `videos.insert` dropped to ~100 units in Dec 2025
and moved to a granular bucket of 100 uploads/day in June 2026. Only the
private-lock policy matters.

## Recommended topology, one instructor, small budget

For the voice-over-slides and podcast scope: **no GPU needed at all.** A Colab
CPU runtime or any laptop runs the whole pipeline; the only per-unit cost is the
TTS provider.

If a GPU becomes necessary (self-hosted cloning or, later, the avatar), the
ranking is: Cloud Run Jobs chunked under the 60-min cap (~$9/mo at 13
lectures), or g2 Spot with a self-deleting startup script (~$4–9/mo, more
operational burden, no timeout ceiling). Colab Pro is price-competitive and
easiest to start but buys a GPU lottery, no persistence, and a ToS clause that
names the use case.
