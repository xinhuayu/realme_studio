# Voices

Put a recording of each voice here as a WAV file: `realme.wav` is the voice
"realme", the server's default (change it as `voice` in `voice_server.json`);
`anna.wav` would be the voice "anna", and so on. 10 to 60 seconds of one person's ordinary speech, in a quiet room,
works best; any sample rate, mono or stereo. The server turns each into a
speaker embedding once and keeps it in `.embeddings/`, so later starts are
quick. A ready-made embedding (1024 raw float32 values) can be given as
`<name>.f32` instead.

Clone only your own voice, or a voice you have permission to use, and say
so where the audio is published.

In a source checkout, recordings and embeddings are not committed (see
`.gitignore`); in a voice server folder, keep them private.
