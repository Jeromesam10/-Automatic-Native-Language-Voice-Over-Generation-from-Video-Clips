# Reference voices

Put the speaker reference recording here as `reference.wav`
(6–30 seconds of clean speech, mono, 16–24 kHz WAV).

The Voice Clone API resolves `speaker_reference` relative to the `Voice_Clone/`
folder, so the default `voices/reference.wav` points at this file. Without it,
`POST /voice` returns HTTP 400 with a "Reference voice file(s) not found" message.

Only use recordings of people who have consented to having their voice cloned.
