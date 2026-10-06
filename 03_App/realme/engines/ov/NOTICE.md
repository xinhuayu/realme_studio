# Vendored: OpenVoice v2 tone colour converter (myshell-ai/OpenVoice)

Source: https://github.com/myshell-ai/OpenVoice — MIT licence.

**Why a copy, and why only part of one.** `pip install git+https://…` needs git
on PATH, which a Windows machine does not have by default — reported as
`Cannot find command 'git'`. And the package pins `librosa==0.9.1` and pulls in
cn2an, jieba, eng_to_ipa and whisper for parts of itself RealMe never touches,
which is a large surface to fail on for a converter that needs none of it.

What is here is the converter and what it imports: `api.py`, `models.py`,
`modules.py`, `attentions.py`, `commons.py`, `transforms.py`,
`mel_processing.py`, `utils.py`. These are byte-identical to upstream. The one
addition is `text/__init__.py`, a stub that resolves the module-level import
`api.py` makes for its BaseSpeakerTTS class and raises if anything calls it.

Weights are not vendored — `realme engine vc --install --vc-backend openvoice`
fetches them from the `myshell-ai/OpenVoiceV2` repository on Hugging Face. The
S3 URL that upstream's README used returns 404.

## Known upstream defect, worked around rather than patched

`ToneColorConverter.__init__(*args, **kwargs)` forwards every keyword to
`OpenVoiceBaseClass.__init__`, which takes only `config_path` and `device`. So
its own documented `enable_watermark=False` raises `TypeError`, and omitting it
makes the constructor import `wavmark` and download a second model. RealMe
subclasses it in `adapters/vc.py` and calls the base initialiser directly, so
these files stay identical to upstream and a future release can be dropped in.
