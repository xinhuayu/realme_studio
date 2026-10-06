def mel(*args, **kwargs):
    """
    Not implemented, on purpose.

    `mel_processing.py` imports this at module level but the tone colour
    converter never calls it -- it works on linear spectrograms. Returning a
    plausible-looking filterbank would be worse than failing: the output would
    be subtly wrong and nothing would say so.
    """
    raise NotImplementedError(
        "This is RealMe's minimal librosa stand-in, used only because the real "
        "librosa cannot be imported on this machine. Mel filters are not part "
        "of the voice-conversion path; if you are seeing this, something is "
        "calling code that needs the real library.")
