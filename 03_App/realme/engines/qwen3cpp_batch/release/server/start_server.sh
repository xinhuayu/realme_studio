#!/usr/bin/env sh
# Start the voice server (see server/qwen3_tts_server.py --help).
cd "$(dirname "$0")/.."
exec python3 server/qwen3_tts_server.py "$@"
