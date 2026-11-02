"""Figures for the report, from the measurements (see the data below and stages.json)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, QUIET, GRID, AXIS = '#0b0b0b', '#52514e', '#e4e3df', '#8a8984'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MUTED = '#9b9a95'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': AXIS,
                     'axes.labelcolor': QUIET, 'xtick.color': QUIET, 'ytick.color': QUIET,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})

def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches='tight')
    fig.savefig(os.path.join(HERE, name.replace('.pdf', '.png')), bbox_inches='tight', dpi=160)
    plt.close(fig)

def box(ax, x, y, w, h, title, lines, main=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=(BLUE if main else 'white'), alpha=1, ec=(BLUE if main else AXIS),
                                lw=(1.6 if main else 1.0)))
    if main:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                    fc='white', alpha=0.88, ec='none'))
    ax.text(x + 0.15, y + h - 0.22, title, fontsize=9, fontweight='bold', color=INK, va='top')
    for i, s in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.55 - 0.27 * i, s, fontsize=8, color=(INK if main else QUIET), va='top')

def arrow(ax, pts, dashed=False):
    for (a, b) in zip(pts[:-1], pts[1:]):
        last = (a, b) == (pts[-2], pts[-1])
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=('-|>' if last else '-'), mutation_scale=9,
                                     color=AXIS, lw=1.0, ls=('--' if dashed else '-'), shrinkA=0, shrinkB=0))

def fig1():
    fig, ax = plt.subplots(figsize=(6.4, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis('off')
    box(ax, 0.2, 6.0, 4.4, 1.1, 'Text', ['Text tokens from the Qwen tokenizer', 'read as the speech advances'])
    box(ax, 5.2, 6.0, 4.4, 1.1, 'Reference voice', ['Speaker encoder: one embedding,', 'computed once'])
    box(ax, 2.4, 4.4, 5.0, 1.1, 'Talker (Qwen3 backbone)', ['28 layers; predicts the semantic token', 'of each frame'], main=True)
    box(ax, 2.4, 2.8, 5.0, 1.1, 'Acoustic predictor', ['5 layers, run 15 times per frame;', 'each run waits for the token before it'], main=True)
    box(ax, 2.4, 1.55, 5.0, 0.85, 'One frame: 16 audio tokens, 80 ms', ['12.5 frames for each second of speech'])
    box(ax, 2.4, 0.0, 5.0, 1.1, 'Vocoder', ['Transformer, upsampling, four convolution', "stages: a sentence's tokens to 24 kHz"])
    arrow(ax, [(2.4, 6.0), (2.4, 5.75), (4.0, 5.75), (4.0, 5.5)])
    arrow(ax, [(7.4, 6.0), (7.4, 5.75), (5.8, 5.75), (5.8, 5.5)])
    arrow(ax, [(4.9, 4.4), (4.9, 3.9)]); arrow(ax, [(4.9, 2.8), (4.9, 2.4)]); arrow(ax, [(4.9, 1.55), (4.9, 1.1)])
    arrow(ax, [(2.4, 1.97), (1.6, 1.97), (1.6, 4.95), (2.4, 4.95)])
    ax.text(1.5, 3.45, 'next frame', rotation=90, fontsize=8, color=QUIET, ha='right', va='center')
    ax.text(5.0, 1.3, 'all frames of a sentence', fontsize=8, color=QUIET, va='center')
    save(fig, 'fig1_model.pdf')

def fig2():
    fig, ax = plt.subplots(figsize=(6.6, 5.6)); ax.set_xlim(0, 10.7); ax.set_ylim(0, 8.5); ax.axis('off')
    items = [(7.4, 'Slides and notes', ['Supplied by the instructor'], False),
             (6.1, 'Text preparation', ['Abbreviations spelled out; respellings', 'checked by a phonemizer'], False),
             (4.8, 'Ledger', ['Each take keyed by text, delivery and voice;', 'only new or changed sentences go on'], False),
             (3.2, 'Token generation (memory-bound)', ['Talker and acoustic predictor, 12.5 frames', 'a second; sentences share each pass'], True),
             (1.6, 'Vocoder (arithmetic-bound)', ['Tokens to 24 kHz samples, per sentence;', 'about 61 GFLOP per second of speech'], True),
             (0.0, 'Assembly', ['Slide videos encoded on free threads;', 'joined with captions and a check'], False)]
    hs = {7.4: 0.95, 6.1: 1.15, 4.8: 1.15, 3.2: 1.25, 1.6: 1.25, 0.0: 1.15}
    for y, t, l, m in items:
        box(ax, 0.2, y, 6.1, hs[y], t, l, main=m)
    ys = [y for y, *_ in items]
    for a, b in zip(ys[:-1], ys[1:]):
        arrow(ax, [(3.0, a), (3.0, b + hs[b])])
    box(ax, 7.0, 4.8, 3.6, 1.15, 'Reference recording', ["The instructor's voice"])
    box(ax, 7.0, 3.2, 3.6, 1.25, 'Speaker encoder', ['One embedding, computed', 'once, in every prompt'])
    arrow(ax, [(8.8, 4.8), (8.8, 4.45)]); arrow(ax, [(7.0, 3.6), (6.3, 3.6)])
    ax.text(6.62, 3.72, 'voice', fontsize=8, color=QUIET, ha='center', va='bottom')
    save(fig, 'fig2_pipeline.pdf')

def fig3():
    steps = ['Original\nengine', 'Threads,\nno copies', 'One-pass\nsnake', 'Sentences\nin passes', '8-bit\nweights',
             '32-bit\nvocoder', 'Vocoder\nin place', 'Vocoder\nkernel', 'Fewer\noperations']
    series = {'1 at a time': ({0: 3.79, 1: 2.82, 2: 2.59, 3: 2.54, 4: 1.77, 5: 1.49, 6: 1.31, 7: 1.25}, MUTED),
              '4 per pass': ({3: 1.66, 4: 1.35, 5: 1.07, 6: 0.79, 7: 0.68}, '#9cc1ee'),
              '8 per pass': ({3: 1.51, 4: 1.30, 5: 1.02, 6: 0.74, 7: 0.58, 8: 0.52}, BLUE),
              '16 per pass': ({6: 0.72, 7: 0.57, 8: 0.48}, ORANGE)}
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(1.0, color=AXIS, ls='--', lw=1); ax.text(8.45, 1.05, 'real time', fontsize=8, color=QUIET, ha='right')
    for name, (d, c) in series.items():
        xs = sorted(d); ys = [d[x] for x in xs]
        ax.plot(xs, ys, color=c, lw=2, marker='o', ms=4, label=name)
        dy = {'1 at a time': 0.0, '4 per pass': 0.12, '8 per pass': 0.0, '16 per pass': -0.12}[name]
        ax.text(xs[-1] + 0.12, ys[-1] + dy, f'{ys[-1]:.2f}', fontsize=8, color=INK, va='center')
    ax.text(0.12, 3.79, '3.79', fontsize=8, color=INK, va='center')
    ax.set_xticks(range(len(steps))); ax.set_xticklabels(steps, fontsize=7.5)
    ax.set_ylim(0, 4.1); ax.set_xlim(-0.3, 8.7)
    ax.set_ylabel('Seconds of computation per second of speech')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    save(fig, 'fig3_speed.pdf')

def fig4():
    s = json.load(open(os.path.join(HERE, 'stages.json')))
    parts = [('predictor', 'Acoustic predictor', BLUE), ('talker', 'Talker', AQUA),
             ('vocoder', 'Vocoder', ORANGE), ('other', 'Other', MUTED)]
    rows = [('revised', 'Batched predictor attention,\nkept predictor graphs,\nordered talker attention'), ('earlier', 'Earlier settings')]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    for i, (key, label) in enumerate(rows):
        left = 0
        for k, name, c in parts:
            v = s[key][k]
            ax.barh(i, v, left=left, color=c, height=0.55, edgecolor='white', linewidth=2, label=(name if i == 0 else None))
            if v > 2:
                ax.text(left + v / 2, i, f'{v:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            left += v
        ax.text(left + 0.4, i, f'{s[key]["total"]:.1f} s', va='center', fontsize=8.5, color=INK)
    ax.set_yticks([0, 1]); ax.set_yticklabels([r[1] for r in rows], fontsize=8)
    ax.set_xlim(0, 46); ax.set_xlabel('Seconds for a stream of 16 sentences (73.6 s of speech)')
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.0))
    save(fig, 'fig4_stages.pdf')

def fig5():
    eng = [(1, 1.25), (2, 0.86), (4, 0.68), (8, 0.58), (12, 0.58), (16, 0.57)]
    gem = [(1, 0.56, 0), (2, 0.45, 0), (5, 0.34, 0), (6, 0.34, 0), (7, 0.35, 1), (10, 0.33, 1), (11, 0.33, 1)]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(1.0, color=AXIS, ls='--', lw=1); ax.text(10, 1.03, 'real time', fontsize=8, color=QUIET, ha='right')
    ax.plot(*zip(*eng), color=BLUE, lw=2, marker='o', ms=4, label='Local engine (computation per pass)')
    ax.plot([g[0] for g in gem], [g[1] for g in gem], color=ORANGE, lw=2, label='Gemini TTS (waiting per request)')
    for n, v, d in gem:
        ax.plot(n, v, 'o', ms=5, mfc=('white' if d else ORANGE), mec=ORANGE, mew=1.4)
    ax.text(1.3, 1.25, '1.25', fontsize=8, color=INK, va='center'); ax.text(1.3, 0.61, '0.56', fontsize=8, color=INK)
    ax.text(16.2, 0.57, '0.57', fontsize=8, color=INK, va='center'); ax.text(11.3, 0.33, '0.33', fontsize=8, color=INK, va='center')
    ax.set_xlabel('Sentences per pass (local) or per request (Gemini)')
    ax.set_ylabel('Seconds per second of speech')
    ax.set_ylim(0, 1.4); ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    ax.legend(frameon=False, fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.02))
    save(fig, 'fig5_grouping.pdf')

def fig0():
    """Visual abstract: (A) the parts of qwen3-tts.cpp and what was changed; (B) the trend."""
    fig = plt.figure(figsize=(7.0, 8.6))
    ax = fig.add_axes([0.02, 0.47, 0.96, 0.51]); ax.set_xlim(0, 10); ax.set_ylim(0, 7.5); ax.axis('off')
    ax.text(0.0, 7.45, 'A. What was changed in qwen3-tts.cpp (blue, marked +)', fontsize=10, fontweight='bold', color=INK, va='top')
    def b(x, y, w, h, title, grey=(), blue=(), main=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.1',
                                    fc=('#e7f0fb' if main else 'white'), ec=(BLUE if main else AXIS), lw=(1.5 if main else 1.0)))
        ax.text(x + 0.12, y + h - 0.16, title, fontsize=8.6, fontweight='bold', color=INK, va='top')
        yy = y + h - 0.52
        for s in grey:
            ax.text(x + 0.12, yy, s, fontsize=7.6, color=QUIET, va='top'); yy -= 0.27
        for s in blue:
            ax.text(x + 0.12, yy, s, fontsize=7.6, color=BLUE, va='top'); yy -= 0.27
    xs = [0.0, 2.55, 5.1, 7.65]; w = 2.35
    b(0.0, 5.85, 10.0, 1.15, 'C interface and pipeline', ['originally one sentence at a time'],
      ['+ 16 sentences a pass, refilled as they finish;  + 12 predictor steps, a take check, all 15 for a failed take'])
    b(xs[0], 3.95, w, 1.6, 'Text tokenizer', ['byte-pair encoding', 'unchanged'])
    b(xs[1], 3.95, w, 1.6, 'Speaker encoder', ['run once per voice', 'unchanged'])
    b(xs[2], 3.95, w, 1.6, 'Transformers', ['talker + predictor'], ['+ passes, 8-bit kernel', '+ one attention for all', '+ graphs kept'], main=True)
    b(xs[3], 3.95, w, 1.6, 'Vocoder', [], ['+ 32-bit, read in place', '+ convolution kernel', '+ 8-bit or GPU options'], main=True)
    b(0.0, 3.0, 10.0, 0.6, 'GGUF loader: 16- or 8-bit weights', [], [])
    ax.text(9.88, 3.3, '+ the 8-bit model, mapped as it is; one conversion at load', fontsize=7.6, color=BLUE, ha='right', va='center')
    b(0.0, 1.9, 10.0, 0.8, 'GGML: a graph built, computed and reset for every step', [],
      ['+ the thread count applied; one thread pool; tinyBLAS gated to the vocoder'])
    b(0.0, 0.6, 7.4, 1.05, 'GPU and other accelerators', ['tried first by the original engine'])
    b(7.65, 0.6, 2.35, 1.05, 'CPU', ['was the fallback;', 'now the target'], [], main=True)
    ax.text(0.0, 0.32, '+ built on each computer for its processor: AVX2, AVX-VNNI, hybrid cores, NEON', fontsize=7.6, color=BLUE, va='top')
    for x in xs:
        c = x + w / 2
        arrow(ax, [(c, 5.85), (c, 5.55)]); arrow(ax, [(c, 3.95), (c, 3.6)])
        arrow(ax, [(c, 1.9), (c, 1.65)])
    arrow(ax, [(5.0, 3.0), (5.0, 2.7)])
    bx = fig.add_axes([0.1, 0.07, 0.86, 0.32])
    fig.text(0.02, 0.425, 'B. What it bought on a 15-watt, 6-core laptop processor without a GPU', fontsize=10, fontweight='bold', color=INK)
    steps = ['Original', 'Threads,\nno copies', 'One-pass\nsnake', 'Passes', '8-bit\nweights', '32-bit\nvocoder',
             'Vocoder\nin place', 'Vocoder\nkernel', 'Fewer\noperations', '12 steps\n+ check']
    one = {0: 3.79, 1: 2.82, 2: 2.59, 3: 2.54, 4: 1.77, 5: 1.49, 6: 1.31, 7: 1.25}
    best = {3: 1.51, 4: 1.30, 5: 1.02, 6: 0.72, 7: 0.57, 8: 0.48}
    lect = {5: 1.09, 6: 0.86, 7: 0.68, 8: 0.63, 9: 0.59}
    bx.grid(axis='y', color=GRID, lw=0.8); bx.set_axisbelow(True)
    bx.axhline(1.0, color=AXIS, ls=':', lw=1); bx.text(0.0, 1.06, 'real time', fontsize=8, color=QUIET)
    for d, c, ls, lab in [(one, MUTED, '-', 'Engine, one sentence at a time'),
                          (best, BLUE, '-', 'Engine, best pass (8 or 16 sentences)'),
                          (lect, BLUE, '--', 'Full lecture, recording / video length')]:
        xs_ = sorted(d); ys_ = [d[x] for x in xs_]
        bx.plot(xs_, ys_, color=c, ls=ls, lw=2, marker='o', ms=4, label=lab)
    for x, v, dy in [(0, 3.79, 0.12), (7, 1.25, 0.12), (8, 0.48, -0.2), (9, 0.59, 0.12)]:
        bx.text(x + 0.12, v + dy, f'{v:.2f}', fontsize=8, fontweight='bold', color=INK)
    bx.set_xticks(range(len(steps))); bx.set_xticklabels(steps, fontsize=7.2)
    bx.set_ylim(0, 4.1); bx.set_xlim(-0.3, 9.6)
    bx.set_ylabel('Seconds per second of speech\n(lecture: minutes per minute of video)', fontsize=8)
    bx.legend(frameon=False, fontsize=8, loc='upper right')
    save(fig, 'fig0_visual.pdf')

for f in (fig0, fig1, fig2, fig3, fig4, fig5):
    f()
