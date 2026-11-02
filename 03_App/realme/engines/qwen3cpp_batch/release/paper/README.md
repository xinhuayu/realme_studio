# Report

`realme-qwen3-tts-report.pdf`: *Running AI models locally on a small budget:
how software design made a cloned-voice lecture narrator practical on an
ordinary CPU* (Xinhua Yu, November 2026).

What limits Qwen3-TTS on a laptop processor without a GPU, which changes to
qwen3-tts.cpp were kept and which did not pay, and how a full 578-sentence
lecture recorded with this engine compares with a hosted voice.

`source/` rebuilds it: `report.md` is the text; `python figures.py` draws the
figures from the measurements written in it and in `stages.json`;
`python build_tex.py` writes `report.tex` with pandoc and compiles it with
pdflatex.
