# Vendored: kNN-VC (bshall/knn-vc)

Source: https://github.com/bshall/knn-vc — MIT licence. The WavLM model code
under `wavlm/` originates from microsoft/unilm, also MIT.

**Why a copy lives here.** Same reason qwen3-tts.cpp does: `torch.hub.load`
fetches this at run time from a GitHub repository that a stranger controls, and
a lecture pipeline that stops working because someone renamed a branch is not a
pipeline. It is 80 KB of Python. The model weights are *not* vendored — those
are 1.3 GB and `realme engine vc --install` downloads them.

Files are byte-identical to upstream so a future diff is meaningful. They use
top-level imports (`from wavlm.WavLM import ...`), so this directory is put on
`sys.path` at load time rather than being rewritten into package-relative form.

Paper: Baas, van Niekerk, Kamper, *Voice Conversion With Just Nearest Neighbors*
(Interspeech 2023).
