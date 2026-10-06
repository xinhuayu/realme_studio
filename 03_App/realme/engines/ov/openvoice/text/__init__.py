"""
A stub for the part of OpenVoice this project does not use.

`openvoice/api.py` imports `text_to_sequence` at module level for its
BaseSpeakerTTS class. RealMe never uses that class -- piper says the words --
but the import still has to resolve, and upstream's real one drags in cn2an,
jieba, eng_to_ipa and a pinned librosa, which is most of why installing the
package fails on a current Python.

So: resolve the import, and raise loudly if anything ever actually calls it.
Silence here would mean text going through an untested path.
"""


def text_to_sequence(*args, **kwargs):
    raise NotImplementedError(
        "OpenVoice's own TTS is not vendored -- RealMe uses it as a voice "
        "converter only, with piper generating the speech.")


def cleaned_text_to_sequence(*args, **kwargs):
    return text_to_sequence()
