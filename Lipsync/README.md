# Lipsync

Re-renders the on-screen speaker's mouth so it matches the dubbed voice.
Used by the Dubbing pipeline as the final step (on by default, `--no-lipsync`
or `"lip_sync": false` to skip). Built on **MuseTalk 1.5** (Tencent Music
Lyra Lab, MIT licence, commercial use allowed).

## How it works

1. **Find the speaker.** OpenCV's YuNet detector finds the largest face in
   each frame that falls inside a dubbed line. Faces smaller than 12% of the
   frame (background people, posters, murals) are ignored. If fewer than half
   of the speech frames contain a usable face, lip sync is **skipped** and the
   original picture is kept (e.g. narration over B-roll).
2. **Crop.** A MuseTalk-style box: centred on the lower nose, from chin up the
   same distance, jaw-wide, +10 px below the chin; smoothed over ±2 frames.
3. **Audio.** Whisper-tiny encoder features of the *dubbed voice only*
   (no music), sliced per frame by timestamp.
4. **Render.** MuseTalk's UNet inpaints the lower half of the 256×256 face in
   VAE latent space in one step (not diffusion).
5. **Blend.** The new mouth/jaw is pasted back with a feathered elliptical
   mask. Frames outside dubbed lines are passed through untouched.

## Models (`Lipsync/models/`, ~4.3 GB)

All from the publishers' official repositories:

| File | Source | Licence |
|---|---|---|
| `musetalkV15/unet.pth`, `musetalk.json` | huggingface.co/TMElyralab/MuseTalk | MIT |
| `sd-vae/` | huggingface.co/stabilityai/sd-vae-ft-mse | MIT |
| `whisper/` | huggingface.co/openai/whisper-tiny | Apache-2.0 |
| `face_detection_yunet_2023mar.onnx` | github.com/opencv/opencv_zoo | MIT |

```bash
python -c "from huggingface_hub import hf_hub_download as d; [d('TMElyralab/MuseTalk', f, local_dir='Lipsync/models') for f in ('musetalkV15/unet.pth','musetalkV15/musetalk.json')]"
```
```bash
python -c "from huggingface_hub import snapshot_download as s; s('stabilityai/sd-vae-ft-mse', local_dir='Lipsync/models/sd-vae', allow_patterns=['config.json','diffusion_pytorch_model.safetensors']); s('openai/whisper-tiny', local_dir='Lipsync/models/whisper', allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"
```
```bash
curl -L -o Lipsync/models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

MuseTalk's own download script points Hugging Face at a third-party mirror
(`hf-mirror.com`) and fetches a face-parsing model from Google Drive; neither
is used here. `Lipsync/MuseTalk/` is the upstream repo kept for reference and
its licence; its code is not imported.

Python packages (root venv): `diffusers`, `librosa`, `transformers`, `torch`,
`opencv-python` (5.x, includes YuNet). Keep `huggingface_hub<2` — newer
versions break `transformers` 5.x.

## Run on its own

Re-run only the lip-sync step of a finished dubbing job:

```bash
python Lipsync/main.py input.mp4 Dubbing/outputs/job_xxx/dubbed_voice.wav out.mp4 --audio Dubbing/outputs/job_xxx/dubbed_audio.wav
```

## Memory

On CPU, lip sync needs about **4.4 GB of available RAM** (the UNet alone is
3.4 GB). It checks before loading and, when there is less, skips with a clear
message — the dub is still delivered with the original picture — instead of
pushing Windows into heavy swapping. Stop other model servers (Voice_Clone,
an idle Dubbing API) before a lip-sync run on an 8 GB machine.

## Verification status

- **Verified** (on the MuseTalk sample face + speech, with the UNet/VAE call
  stubbed): face found in 268/268 frames, crop lands on mouth/chin/jaw and
  tracks the head, per-frame audio windows are the expected (50, 384), blend
  and H.264 encode work, output length matches the input, peak RAM < 1 GB.
- **Not yet verified:** the MuseTalk UNet/VAE inference itself, i.e. the
  quality of the generated mouths. It needs a machine with ≥ 4.4 GB free RAM
  (or a GPU).

## Limits

- **Speed:** on CPU roughly 1–2 s per re-rendered frame (25–30 frames per
  second of speech). A CUDA GPU is used automatically and is ~50× faster.
- **One face.** The largest face is synced; two people talking at once are not
  told apart (needs speaker diarization + active-speaker detection).
- **Profile views, hands over the mouth, fast head turns** fall back to the
  original frame or can look smeared.
- **Resolution:** the mouth is generated at 256×256 face size, so very large
  close-ups get a slightly softer mouth than the rest of the face.
