"""Lip sync with MuseTalk 1.5 (MIT, TMElyralab).

Re-renders the mouth region of the on-screen speaker so it matches new audio.

Differences from the reference MuseTalk inference script, to run on a plain
CPU/Windows setup with weights only from official sources:
  * Face boxes come from OpenCV's built-in YuNet detector (opencv_zoo ONNX)
    instead of mmpose/DWPose + S3FD.
  * The new mouth is blended back with a feathered jaw-shaped mask instead of
    the BiSeNet face-parsing model (whose weights are only on Google Drive).
  * Audio windows are picked by frame *timestamp* (works for 29.97 fps and
    variable frame rate) instead of int(fps) frame counting.
  * Only frames inside the dubbed speech windows are re-rendered; every other
    frame is passed through untouched.
  * The MuseTalk repo's code is not imported; the small pieces needed are
    re-implemented here. Weights load with torch.load(weights_only=True).
"""
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

MODELS_DIR = Path(__file__).resolve().parent / "models"

_LATENT_SIZE = 256           # MuseTalk works on 256x256 face crops
_EXTRA_MARGIN = 10           # v1.5: extend crop 10 px below the chin
_AUDIO_FPS = 50              # whisper encoder frames per second
_PAD_LEFT = _PAD_RIGHT = 2   # audio context, in video frames, each side


class LipSyncError(Exception):
    pass


class InsufficientMemory(LipSyncError):
    """Not enough free RAM to load MuseTalk without heavy swapping."""


# MuseTalk on CPU keeps ~3.4 GB of UNet weights resident plus VAE, Whisper and
# activations. Below this much *available* RAM, loading pushes the OS into
# swapping hard enough to freeze the machine, so refuse instead.
MIN_AVAILABLE_MB = 4500


def _check_memory():
    import psutil
    if torch.cuda.is_available():
        return
    avail = psutil.virtual_memory().available // 2**20
    if avail < MIN_AVAILABLE_MB:
        raise InsufficientMemory(
            f"Lip sync needs about {MIN_AVAILABLE_MB / 1024:.1f} GB of free RAM on CPU, "
            f"but only {avail / 1024:.1f} GB is available. Close other apps or run on a "
            "machine with more memory or a GPU.")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class _PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)


class _Models:
    def __init__(self, models_dir=MODELS_DIR, device=None):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import AutoFeatureExtractor, WhisperModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models_dir = Path(models_dir)
        unet_dir = models_dir / "musetalkV15"
        for required in (unet_dir / "unet.pth", unet_dir / "musetalk.json",
                         models_dir / "sd-vae", models_dir / "whisper",
                         models_dir / "face_detection_yunet_2023mar.onnx"):
            if not required.exists():
                raise LipSyncError(f"Missing lip-sync model file: {required} (see Lipsync/README.md)")

        print("Loading MuseTalk 1.5 models...")
        config = json.loads((unet_dir / "musetalk.json").read_text())
        # Build the UNet without allocating weights, then attach the
        # memory-mapped checkpoint tensors directly (assign=True). Peak RAM is
        # one copy of the 3.4 GB weights instead of two, and the pages are
        # file-backed so the OS can drop them under memory pressure.
        with torch.device("meta"):
            self.unet = UNet2DConditionModel(**config)
        state = torch.load(unet_dir / "unet.pth", map_location="cpu",
                           weights_only=True, mmap=True)
        self.unet.load_state_dict(state, assign=True)
        del state
        self.unet.to(self.device).eval()

        self.vae = AutoencoderKL.from_pretrained(models_dir / "sd-vae").to(self.device).eval()
        self.scale = self.vae.config.scaling_factor

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(models_dir / "whisper")
        self.whisper = WhisperModel.from_pretrained(models_dir / "whisper").to(self.device).eval()
        self.pe = _PositionalEncoding().to(self.device)
        self.timesteps = torch.tensor([0], device=self.device)
        self.yunet_path = str(models_dir / "face_detection_yunet_2023mar.onnx")


_models = None


def _get_models():
    global _models
    if _models is None:
        _check_memory()
        _models = _Models()
    return _models


# ---------------------------------------------------------------------------
# Audio features
# ---------------------------------------------------------------------------

@torch.no_grad()
def _audio_features(m, voice_wav):
    """Whisper-tiny encoder hidden states for the whole voice track,
    padded so any frame's window can be sliced. Shape (T, 5, 384)."""
    import librosa

    audio, sr = librosa.load(str(voice_wav), sr=16000)
    seg = 30 * sr
    feats = []
    for start in range(0, len(audio), seg):
        chunk = audio[start:start + seg]
        inp = m.feature_extractor(chunk, return_tensors="pt", sampling_rate=sr).input_features
        hidden = m.whisper.encoder(inp.to(m.device), output_hidden_states=True).hidden_states
        feats.append(torch.stack(hidden, dim=2)[0])          # (1500, layers, 384)
        # each 30 s window yields 1500 frames; keep only the real audio part
        real = math.ceil(len(chunk) / sr * _AUDIO_FPS)
        feats[-1] = feats[-1][:real]
    feats = torch.cat(feats, dim=0)
    pad_l = torch.zeros_like(feats[: 2 * _PAD_LEFT])
    pad_r = torch.zeros_like(feats[: 2 * 3 * _PAD_RIGHT])
    return torch.cat([pad_l, feats, pad_r], dim=0), len(audio) / sr


def _audio_window(feats, t, fps):
    """Audio context for the video frame shown at time t (seconds)."""
    per_frame = _AUDIO_FPS / fps
    width = 2 * (_PAD_LEFT + _PAD_RIGHT + 1)
    idx = int(math.floor(t * _AUDIO_FPS))
    idx = max(0, min(idx, feats.shape[0] - width))
    clip = feats[idx: idx + width]                           # (10, layers, 384)
    return clip.reshape(-1, clip.shape[-1])                  # (10*layers, 384)


# ---------------------------------------------------------------------------
# Face detection / crop boxes
# ---------------------------------------------------------------------------

class _FaceTracker:
    """Finds the main (largest) face with YuNet and turns it into a
    MuseTalk-style crop box: centred on the lower nose bridge, from the chin
    up the same distance, jaw width wide."""

    def __init__(self, model_path, min_face_ratio=0.12, score=0.8):
        self.model_path = model_path
        self.min_face_ratio = min_face_ratio
        self.score = score
        self.detector = None

    def box(self, frame):
        h, w = frame.shape[:2]
        scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        sh, sw = small.shape[:2]
        if self.detector is None:
            self.detector = cv2.FaceDetectorYN.create(self.model_path, "", (sw, sh), self.score)
        self.detector.setInputSize((sw, sh))
        _, faces = self.detector.detect(small)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3]) / scale
        x, y, fw, fh = face[:4]
        # Ignore small faces: background people, posters, murals.
        if fw < self.min_face_ratio * min(w, h):
            return None
        eye_y = (face[5] + face[7]) / 2
        nose_y = face[9]
        centre = eye_y + 0.75 * (nose_y - eye_y)
        chin = y + fh
        half = chin - centre
        x1, x2 = int(max(0, x)), int(min(w, x + fw))
        y1, y2 = int(max(0, centre - half)), int(min(h, chin))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.float32)


def _smooth(boxes, radius=2):
    """Moving average over neighbouring frames to stop the crop jittering."""
    out = list(boxes)
    for i, b in enumerate(boxes):
        if b is None:
            continue
        near = [boxes[j] for j in range(max(0, i - radius), min(len(boxes), i + radius + 1))
                if boxes[j] is not None]
        out[i] = np.mean(near, axis=0)
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _to_tensor(crops, device, half_mask):
    x = torch.from_numpy(np.stack(crops)[..., ::-1].copy()).float() / 255.0   # BGR->RGB
    x = x.permute(0, 3, 1, 2)
    if half_mask:
        x[:, :, _LATENT_SIZE // 2:, :] = 0          # hide the mouth half
    return ((x - 0.5) / 0.5).to(device)


@torch.no_grad()
def _render_batch(m, crops, audio_windows):
    masked = m.vae.encode(_to_tensor(crops, m.device, True)).latent_dist.sample() * m.scale
    ref = m.vae.encode(_to_tensor(crops, m.device, False)).latent_dist.sample() * m.scale
    latents = torch.cat([masked, ref], dim=1)
    audio = m.pe(torch.stack(audio_windows).to(m.device))
    pred = m.unet(latents, m.timesteps, encoder_hidden_states=audio).sample
    img = m.vae.decode(pred / m.scale).sample
    img = ((img / 2 + 0.5).clamp(0, 1) * 255).round().byte()
    return img.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]   # RGB->BGR uint8


_mask_cache = {}


def _jaw_mask(w, h):
    """Feathered mask over mouth, cheeks and jaw (lower part of the crop)."""
    key = (w, h)
    if key not in _mask_cache:
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (w // 2, int(h * 0.74)), (int(w * 0.40), int(h * 0.27)),
                    0, 0, 360, 1.0, -1)
        k = max(3, (min(w, h) // 8) | 1)
        _mask_cache[key] = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    return _mask_cache[key]


def _paste(frame, face, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    face = cv2.resize(face, (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    region = frame[y1:y2, x1:x2].astype(np.float32)
    mask = _jaw_mask(w, h)
    frame[y1:y2, x1:x2] = (face * mask + region * (1 - mask)).clip(0, 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lip_sync(video_path, voice_wav, out_path, windows, ffmpeg="ffmpeg",
             batch_size=8, min_face_coverage=0.5, log=print):
    """Re-render the speaker's mouth to match `voice_wav` inside `windows`
    ([(start_s, end_s), ...]). Writes a video-only H.264 file to `out_path`.

    Returns a report dict. If no usable speaking face is found, nothing is
    written and report["applied"] is False (the caller keeps the original).
    """
    m = _get_models()
    tracker = _FaceTracker(m.yunet_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise LipSyncError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    def active(t):
        return any(s <= t <= e for s, e in windows)

    # Pass 1: timestamps and face boxes for frames inside the speech windows.
    times, boxes = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = len(times) / fps
        times.append(t)
        boxes.append(tracker.box(frame) if active(t) else None)
    cap.release()
    if not times:
        raise LipSyncError("No frames could be read from the video")

    active_idx = [i for i, t in enumerate(times) if active(t)]
    with_face = [i for i in active_idx if boxes[i] is not None]
    coverage = len(with_face) / len(active_idx) if active_idx else 0.0
    report = {"applied": False, "frames_total": len(times), "frames_in_speech": len(active_idx),
              "frames_with_face": len(with_face), "face_coverage": round(coverage, 2), "fps": round(fps, 3)}
    if not active_idx or coverage < min_face_coverage:
        report["reason"] = (f"No clear on-screen speaker: a face was found in {coverage:.0%} of the "
                            f"speech frames (need {min_face_coverage:.0%}). Lip sync skipped.")
        return report

    boxes = _smooth(boxes)
    log(f"Lip sync: {len(with_face)} frames with a face out of {len(active_idx)} speech frames")

    feats, _ = _audio_features(m, voice_wav)

    # Pass 2: stream frames, re-render the mouth on frames with a face.
    cap = cv2.VideoCapture(str(video_path))
    encoder = None        # started on the first frame, from its real (rotated) size
    pending = []          # (frame, box or None, index)
    done = 0

    def start_encoder(width, height):
        return subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", str(out_path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def flush():
        nonlocal done
        todo = [(k, p) for k, p in enumerate(pending) if p[1] is not None]
        if todo:
            crops, wins = [], []
            for _, (frame, box, idx) in todo:
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                crops.append(cv2.resize(crop, (_LATENT_SIZE, _LATENT_SIZE),
                                        interpolation=cv2.INTER_LANCZOS4))
                wins.append(_audio_window(feats, times[idx], fps))
            faces = _render_batch(m, crops, wins)
            for (k, (frame, box, idx)), face in zip(todo, faces):
                pending[k] = (_paste(frame, face, box), box, idx)
            done += len(todo)
            log(f"Lip sync: rendered {done}/{len(with_face)} frames")
        for frame, _, _ in pending:
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        pending.clear()

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height = frame.shape[0]
            if encoder is None:
                encoder = start_encoder(frame.shape[1], frame.shape[0])
            box = None
            if idx < len(boxes) and boxes[idx] is not None:
                x1, y1, x2, y2 = boxes[idx].astype(int)
                y2 = min(y2 + _EXTRA_MARGIN, height)
                box = (x1, y1, x2, y2)
            pending.append((frame, box, idx))
            if sum(1 for p in pending if p[1] is not None) >= batch_size:
                flush()
            idx += 1
        flush()
    finally:
        cap.release()
        code, err = 1, "no frames decoded"
        if encoder is not None:
            encoder.stdin.close()
            err = encoder.stderr.read().decode("utf-8", "replace")
            code = encoder.wait()
    if code != 0:
        raise LipSyncError(f"ffmpeg failed while encoding lip-synced video: {err[-500:]}")

    report.update(applied=True, frames_rendered=done)
    return report
