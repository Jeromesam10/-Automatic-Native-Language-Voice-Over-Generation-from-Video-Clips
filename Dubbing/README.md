# Dubbing

Takes a video with speech in one language and produces the same video with
the speech translated into another language, spoken in the **original
speaker's cloned voice**, placed at the **original timing**, over the
**original music and background sounds**.

```
video ─► extract audio (ffmpeg)
      ─► split voice / background (Demucs)
      ─► speech-to-text with timestamps (faster-whisper)
      ─► translate each line (local M2M100 model, or Google Translate)
      ─► speaker reference: 3–20 s of the speaker's own voice, cut from the video
      ─► speak each line in the cloned voice (Voice_Clone service)
      ─► fit each line into its time slot (speed up to 1.35×, then trim)
      ─► mix with background
      ─► lip sync the on-screen speaker (MuseTalk 1.5, see Lipsync/README.md)
      ─► mux into the video + subtitles
```

Lip sync is skipped automatically when there is no clear on-screen speaker
(narration over B-roll); the result's `lip_sync` field says what happened.

## Outputs

Each run writes a folder `Dubbing/outputs/job_<id>/` with:

| File | What |
|---|---|
| `<name>.<lang>.mp4` | Final dubbed video (original picture, new audio, soft subtitle track) |
| `subtitles.<lang>.srt` / `subtitles.<source>.srt` | Translated and original subtitles |
| `dubbed_audio.wav` | Final mixed soundtrack |
| `vocals.wav`, `background.wav` | Separated original voice and background |
| `speaker_reference.wav` | Voice sample used for cloning |
| `result.json` | Transcript, translation and per-line timing report |

## Setup

```bash
pip install -r Dubbing/requirements.txt
```
```bash
pip install --no-deps demucs openunmix dora-search
```
```bash
pip install julius einops lameenc omegaconf retrying submitit treetable
```

ffmpeg is bundled through `imageio-ffmpeg`; a system ffmpeg on PATH is used if present.
Voice cloning needs the Voice_Clone service with Coqui TTS, in its own
environment (Coqui pins library versions that conflict with VideoToText):

```bash
pip install -r Voice_Clone/requirements.txt   # includes coqui-tts
```

XTTS asks you to accept the Coqui Public Model License (non-commercial) on
first download; set `COQUI_TOS_AGREED=1` once you have agreed to it.
`.claude/launch.json` has ready-made `voice-clone` and `dubbing` launch entries.

## Run

Start the voice service, then dub from the command line or through the API:

```bash
cd Voice_Clone && uvicorn api:app --port 8001
```
```bash
cd Dubbing && python main.py ../VideoToText/videos/Video-525.mp4 hi
```
```bash
cd Dubbing && uvicorn api:app --port 8003
```

`POST /dub` (also available through the orchestrator at `POST /dub`):

```json
{"video_path": "C:/videos/talk.mp4", "target_language": "hi"}
```

Options: `source_language` (auto-detected), `tts_backend` (`clone` default,
or `edge` for a stock voice with no cloning), `speaker_reference` (a WAV to
use instead of the voice from the video), `separate_vocals` (default true).

## Language support for the cloned voice

| Target | How it is voiced | Similarity to speaker |
|---|---|---|
| en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi | XTTS v2 zero-shot cloning | Good |
| ta, te, kn, ml, bn, gu, mr, pa | Edge TTS voice + FreeVC voice conversion | Rough (FreeVC is trained on English speech) |
| others | `edge` backend only (stock voice) | None |

## Settings (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `small` | `medium` / `large-v3` are far more accurate, especially for Indian languages, but slower |
| `WHISPER_DEVICE` | `auto` | `cuda` with a GPU |
| `TTS_BACKEND` | `clone` | `edge` = no cloning |
| `TRANSLATOR` | `local` | `local` = facebook/m2m100_418M (MIT, offline; no Telugu/Punjabi). `google` = free endpoint, rate-limited and may return a bot check |
| `TRANSLATION_MODEL` | `facebook/m2m100_418M` | `facebook/m2m100_1.2B` is better and slower |
| `VOICE_API_URL` | `http://127.0.0.1:8001/voice` | Voice_Clone service |
| `MAX_SPEEDUP` | `1.35` | Above this, speech starts to sound rushed |
| `SEPARATE_VOCALS` | `1` | `0` skips Demucs; the background is then lost |
| `LIP_SYNC` | `1` | `0` skips lip sync |

## Known limits

- **One speaker.** All speech is cloned from one combined reference. Videos with
  several people need speaker diarization (e.g. pyannote) to give each their own voice.
- **Lip sync is slow on CPU** (about 1–2 s per frame of speech); set
  `LIP_SYNC=0` or pass `--no-lipsync` for quick drafts.
- **Translation length.** Some languages run 20–40% longer than English. Lines
  that do not fit after 1.35× speed-up are trimmed with a fade; `result.json`
  reports `trimmed_seconds` per line.
- **CPU speed.** On CPU, expect several minutes per minute of video (Demucs,
  Whisper and XTTS are all heavy). A GPU makes it many times faster.
- **Short clips.** Cloning needs at least 3 s of the speaker's voice (6 s+ is
  recommended; the result's `warnings` says when there was less). Whisper
  `small` can mishear short or noisy speech; use `WHISPER_MODEL=medium` or larger.
- **Licensing.** XTTS v2 is non-commercial (Coqui Public Model License); Edge
  TTS and the free Google Translate endpoint are unofficial. Replace them before
  commercial use.

## Measured on the sample clip (CPU only, no GPU)

`VideoToText/videos/Video-525.mp4` (11.9 s, one spoken line), English → Hindi,
cloned voice: **~3–4 minutes** end to end (Demucs ~20 s, Whisper ~13 s,
translation model load ~30 s, XTTS load + synthesis most of the rest). With the
`edge` backend: ~80 s.
- **Consent.** Only clone voices of people who agreed to it.
