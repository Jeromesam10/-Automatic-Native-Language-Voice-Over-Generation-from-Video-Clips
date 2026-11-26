# Automatic Native-Language Voice-Over Generation from Video Clips

Turns a video into a version in the viewer's own language. Two modes, both
started through the orchestrator:

## 1. Narration — `POST /run`
For videos **without** a speaker (B-roll, product shots): describe what is on
screen and narrate it.

```
video ─► VideoToText (8002)  sample frames, YOLO objects + Qwen2.5-VL caption
      ─► Translator  (8000)  translate caption (local M2M100; gTTS preview)
      ─► Voice_Clone (8001)  speak it in the voice of speaker_reference
      ─► Dubbing     (8003)  /narrate: voice-over on the video, original sound at 25%
      ─► narrated .mp4
```

```json
POST http://127.0.0.1:8080/run
{"input": "VideoToText/videos/Video-525.mp4", "language": "ta",
 "speaker_reference": "voices/reference.wav"}
```

## 2. Dubbing — `POST /dub`
For videos **with** someone speaking: translate what they say, in their own
cloned voice, at the original timing, with lip sync.

```
video ─► Dubbing (8003)  Demucs voice/background split ─► Whisper speech-to-text
                         ─► translate each line ─► speaker sample cut from the video
      ─► Voice_Clone (8001)  each line in the speaker's cloned voice
      ─► Dubbing (8003)  fit lines to original timing ─► mix with background
                         ─► Lipsync (MuseTalk 1.5, skipped if no on-screen face)
                         ─► dubbed .mp4 + subtitles
```

```json
POST http://127.0.0.1:8080/dub
{"video_path": "C:/videos/talk.mp4", "target_language": "hi"}
```

## Services

| Service | Port | Folder | Python env |
|---|---|---|---|
| Translator | 8000 | `LanguageTranslator(TextToAudio)/` | root `.venv` |
| Voice_Clone | 8001 | `Voice_Clone/` | `Voice_Clone/.venv` (Coqui TTS) |
| VideoToText | 8002 | `VideoToText/` | root `.venv` |
| Dubbing (+ Lipsync) | 8003 | `Dubbing/`, `Lipsync/` | root `.venv` |
| Orchestrator | 8080 | `Langraph_Orchestrater/` | root `.venv` |

All five are defined in `.claude/launch.json`. Narration needs 8000–8003;
dubbing needs 8001 and 8003. Start the orchestrator last. Services run on
one machine and pass each other absolute file paths.

Video paths may be absolute or relative to the project root (or to
`VideoToText/`). Missing videos or voice samples are rejected with 404 before
any model runs. A downstream "bad input" (e.g. no speech in the video) is
returned as a 4xx; a service that is down or crashes is returned as 502.

Details: [Dubbing/README.md](Dubbing/README.md), [Lipsync/README.md](Lipsync/README.md).

## Hardware

The models are large: Qwen2.5-VL-3B (~7 GB RAM), XTTS (~2 GB), MuseTalk
(~4.4 GB free RAM). An 8 GB machine can run one heavy stage at a time; for
production use 16 GB+ and an NVIDIA GPU.
