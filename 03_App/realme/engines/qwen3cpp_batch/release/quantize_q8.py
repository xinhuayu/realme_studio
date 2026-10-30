"""Write an 8-bit (Q8_0) copy of the voice model with the engine's own library.

    python tools/quantize_q8.py <library> <models dir> <output dir>

<library> is build/libqwen3tts.so (Linux), build/libqwen3tts.dylib (macOS) or
build\\qwen3tts.dll (Windows). <models dir> holds qwen3-tts-0.6b-f16.gguf and
qwen3-tts-tokenizer-f16.gguf. The output folder gets the 8-bit transformer and
a copy of the vocoder file, so it can be passed to the CLI's -m on its own.
The matrices each step multiplies by are quantized; embeddings, norms, the
text projection, the speaker encoder and the vocoder are left as they are.
"""
import ctypes
import os
import shutil
import sys
from pathlib import Path

F16, Q8, TOK = ("qwen3-tts-0.6b-f16.gguf", "qwen3-tts-0.6b-q8_0.gguf",
                "qwen3-tts-tokenizer-f16.gguf")


def main(argv) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    lib_path, src, out = Path(argv[1]).resolve(), Path(argv[2]), Path(argv[3])
    if not (src / F16).is_file() or not (src / TOK).is_file():
        print(f"{src} needs {F16} and {TOK}")
        return 1
    if os.name == "nt":
        os.add_dll_directory(str(lib_path.parent))
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.qwen3_tts_realme_quantize_q8
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
    fn.restype = ctypes.c_int32
    out.mkdir(parents=True, exist_ok=True)
    part = out / (Q8 + ".part")
    err = ctypes.create_string_buffer(512)
    print("writing the 8-bit model (a minute or two) ...")
    n = fn(str(src / F16).encode("utf-8"), str(part).encode("utf-8"), err, 512)
    if n <= 0:
        part.unlink(missing_ok=True)
        print("failed:", err.value.decode("utf-8", "replace"))
        return 1
    os.replace(part, out / Q8)
    shutil.copy2(src / TOK, out / TOK)
    print(f"{n} matrices in 8 bits: {out / Q8} "
          f"({(out / Q8).stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
