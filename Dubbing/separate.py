"""Split the soundtrack into the speaker's voice and everything else
(music, effects, ambience) with Demucs, so the background can be kept under
the dubbed voice."""
import numpy as np
import soundfile as sf

from config import DEMUCS_MODEL

_model = None


def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model
        _model = get_model(DEMUCS_MODEL)
        _model.eval()
    return _model


def separate_vocals(wav_path, vocals_path, background_path):
    """wav_path: stereo WAV at the model's sample rate (44.1 kHz).
    Writes vocals and background (no_vocals) WAVs; returns their paths."""
    import torch
    from demucs.apply import apply_model

    model = _get_model()
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sr != model.samplerate:
        raise ValueError(f"Expected {model.samplerate} Hz audio, got {sr}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    wav = torch.from_numpy(audio.T.copy())             # (channels, samples)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * std + mean

    names = model.sources
    vocals = sources[names.index("vocals")]
    background = sources.sum(0) - vocals

    sf.write(str(vocals_path), vocals.cpu().numpy().T, sr)
    sf.write(str(background_path), background.cpu().numpy().T, sr)
    return vocals_path, background_path
