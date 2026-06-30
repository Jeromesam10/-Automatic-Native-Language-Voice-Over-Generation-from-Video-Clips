from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from django.conf import settings

# Let unsupported MPS ops fall back to CPU instead of crashing (e.g. MMS-TTS).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_VC_MODEL = "voice_conversion_models/multilingual/vctk/freevc24"
_WAVLM_URI = "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/WavLM-Large.pt"
_WAVLM_MIN_BYTES = 1_200_000_000
_REFERENCE_MIN_SECONDS = 6.0
_REFERENCE_GOOD_SECONDS = 15.0

_MMS_LANGUAGES: dict[str, str] = {
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
    "bn": "ben",
    "gu": "guj",
    "mr": "mar",
    "pa": "pan",
    "hi": "hin",
    "as": "asm",
    "or": "ory",
    "ur": "urd",
    "ne": "npi",
    "si": "sin",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "pl": "pol",
    "nl": "nld",
    "cs": "ces",
    "hu": "hun",
    "tr": "tur",
    "uk": "ukr",
    "ro": "ron",
    "el": "ell",
    "sv": "swe",
    "da": "dan",
    "no": "nor",
    "fi": "fin",
    "sk": "slk",
    "bg": "bul",
    "hr": "hrv",
    "sr": "srp",
    "sl": "slv",
    "lt": "lit",
    "lv": "lav",
    "et": "est",
    "ca": "cat",
    "eu": "eus",
    "gl": "glg",
    "cy": "cym",
    "ga": "gle",
    "is": "isl",
    "sq": "als",
    "mk": "mkd",
    "bs": "bos",
    "be": "bel",
    "ja": "jpn",
    "ko": "kor",
    "zh": "cmn",
    "zh-cn": "cmn",
    "zh-tw": "cmn",
    "vi": "vie",
    "th": "tha",
    "id": "ind",
    "ms": "zsm",
    "fil": "fil",
    "my": "mya",
    "km": "khm",
    "lo": "lao",
    "jv": "jav",
    "ar": "ara",
    "fa": "fas",
    "he": "heb",
    "ku": "kmr",
    "ps": "pbt",
    "sw": "swh",
    "am": "amh",
    "yo": "yor",
    "ha": "hau",
    "zu": "zul",
    "xh": "xho",
    "af": "afr",
    "so": "som",
    "rw": "kin",
    "lg": "lug",
    "ny": "nya",
    "sn": "sna",
    "st": "sot",
    "tn": "tsn",
    "qu": "quy",
    "gn": "grn",
    "ht": "hat",
}

_XTTS_LANGUAGES = frozenset(
    {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"}
)

_mms_models: dict[str, tuple[object, object, int]] = {}
_xtts = None
_vc = None


class TTSError(Exception):
    """Raised when local TTS synthesis fails."""


def _import_torch():
    import torch

    return torch


def _import_numpy():
    import numpy as np

    return np


def _import_soundfile():
    import soundfile as sf

    return sf


def _import_torchaudio():
    import torchaudio

    return torchaudio


def voices_dir() -> Path:
    return Path(settings.VOICEOVER_VOICES_DIR)


def _get_device() -> str:
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_vc_device() -> str:
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_xtts_device() -> str:
    """XTTS has ops unsupported on Apple MPS, so prefer CUDA or CPU."""
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_mms_device() -> str:
    """MMS-TTS (VITS) has conv ops unsupported on Apple MPS, so avoid it."""
    torch = _import_torch()
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_language(text: str) -> str:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    lang = detect(text)
    if lang.startswith("zh"):
        return "zh-cn"
    return lang


# Unicode script ranges -> default TTS language code for that script. Kana is
# listed before the CJK ideograph range so Japanese is matched before Chinese.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("ta", 0x0B80, 0x0BFF),
    ("te", 0x0C00, 0x0C7F),
    ("kn", 0x0C80, 0x0CFF),
    ("ml", 0x0D00, 0x0D7F),
    ("si", 0x0D80, 0x0DFF),
    ("hi", 0x0900, 0x097F),
    ("bn", 0x0980, 0x09FF),
    ("gu", 0x0A80, 0x0AFF),
    ("pa", 0x0A00, 0x0A7F),
    ("or", 0x0B00, 0x0B7F),
    ("ar", 0x0600, 0x06FF),
    ("he", 0x0590, 0x05FF),
    ("ja", 0x3040, 0x30FF),
    ("ko", 0xAC00, 0xD7AF),
    ("th", 0x0E00, 0x0E7F),
    ("el", 0x0370, 0x03FF),
    ("ru", 0x0400, 0x04FF),
    ("zh-cn", 0x4E00, 0x9FFF),
)

# Languages that share a script: never override between members of one group.
_SCRIPT_GROUPS: dict[str, frozenset[str]] = {
    "ta": frozenset({"ta"}),
    "te": frozenset({"te"}),
    "kn": frozenset({"kn"}),
    "ml": frozenset({"ml"}),
    "si": frozenset({"si"}),
    "hi": frozenset({"hi", "mr", "ne"}),
    "bn": frozenset({"bn", "as"}),
    "gu": frozenset({"gu"}),
    "pa": frozenset({"pa"}),
    "or": frozenset({"or"}),
    "ar": frozenset({"ar", "fa", "ur"}),
    "he": frozenset({"he"}),
    "ja": frozenset({"ja"}),
    "ko": frozenset({"ko"}),
    "th": frozenset({"th"}),
    "el": frozenset({"el"}),
    "ru": frozenset({"ru", "uk", "bg", "sr", "mk", "be"}),
    "zh-cn": frozenset({"zh", "zh-cn", "zh-tw"}),
}


def _dominant_script_language(text: str, threshold: float = 0.3) -> str | None:
    """Return the TTS language code for the text's dominant non-Latin script."""
    counts: dict[str, int] = {}
    total = 0
    for char in text:
        if not char.isalpha():
            continue
        total += 1
        code_point = ord(char)
        for code, low, high in _SCRIPT_RANGES:
            if low <= code_point <= high:
                counts[code] = counts.get(code, 0) + 1
                break

    if total == 0 or not counts:
        return None

    code, hits = max(counts.items(), key=lambda item: item[1])
    if hits / total < threshold:
        return None
    return code


def _resolve_language(text: str, language: str | None) -> str:
    script_language = _dominant_script_language(text)

    if language:
        lang = language.lower()
        if script_language:
            group = _SCRIPT_GROUPS.get(script_language, frozenset({script_language}))
            if lang not in group:
                print(
                    f"Requested language '{lang}' does not match the text script; "
                    f"using detected language '{script_language}' instead."
                )
                return script_language
        return lang

    if script_language:
        return script_language
    return detect_language(text)


def _resolve_mms_code(language: str) -> str:
    language = language.lower()
    if language in _MMS_LANGUAGES:
        return _MMS_LANGUAGES[language]
    if len(language) == 3 and language.isalpha():
        return language
    raise TTSError(
        f"No local MMS-TTS model mapped for '{language}'. "
        "Pass a 3-letter ISO 639-3 code (e.g. tam, fra, vie)."
    )


def _backend_for_language(language: str, prefer: str | None, clone: bool) -> str:
    if prefer and prefer != "auto":
        return prefer

    if clone:
        if language in _XTTS_LANGUAGES:
            return "xtts"
        if language in _MMS_LANGUAGES or (len(language) == 3 and language.isalpha()):
            return "mms_vc"
        raise TTSError(
            f"Language '{language}' is not supported for local voice cloning. "
            f"Try a 3-letter MMS code or one of: {', '.join(sorted(set(_MMS_LANGUAGES) | _XTTS_LANGUAGES))}"
        )

    if language in _MMS_LANGUAGES or (len(language) == 3 and language.isalpha()):
        return "mms"
    if language in _XTTS_LANGUAGES:
        return "xtts"
    raise TTSError(f"Language '{language}' is not supported.")


def list_supported_languages() -> dict[str, str]:
    mms_only = sorted(code for code in _MMS_LANGUAGES if code not in _XTTS_LANGUAGES)
    return {
        "xtts": ", ".join(sorted(_XTTS_LANGUAGES)),
        "mms": ", ".join(mms_only),
    }


def _normalize_speaker_wav(
    speaker_wav: Path | str | list[Path | str],
) -> list[Path]:
    if isinstance(speaker_wav, (str, Path)):
        paths = [Path(speaker_wav)]
    else:
        paths = [Path(path) for path in speaker_wav]

    resolved = [path.resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise TTSError(
            "Reference voice file(s) not found: "
            + ", ".join(missing)
            + ". Record 6-30 seconds of clean speech and save under audio/voices/."
        )
    _warn_about_reference_quality(resolved)
    return resolved


def _default_speaker_wavs() -> list[Path]:
    wavs = sorted(path for path in voices_dir().glob("*.wav") if path.is_file())
    if wavs:
        return wavs
    return [voices_dir() / "reference.wav"]


def _audio_duration(path: Path) -> float:
    sf = _import_soundfile()
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def _warn_about_reference_quality(speaker_files: list[Path]) -> None:
    durations = [_audio_duration(path) for path in speaker_files]
    total_duration = sum(durations)
    files = ", ".join(f"{path.name}={duration:.1f}s" for path, duration in zip(speaker_files, durations))

    print(f"Reference voice: {files} (total {total_duration:.1f}s)")
    if total_duration < _REFERENCE_MIN_SECONDS:
        print(
            "WARNING: The reference voice is too short for reliable cloning. "
            f"Use at least {_REFERENCE_MIN_SECONDS:.0f}s; 15-30s of clean speech is much better."
        )
    elif total_duration < _REFERENCE_GOOD_SECONDS:
        print(
            "NOTE: Voice matching may be weak with short references. "
            "For better similarity, add more clean WAV clips under audio/voices/."
        )


def _load_audio_mono(path: Path, sample_rate: int):
    np = _import_numpy()
    torchaudio = _import_torchaudio()
    waveform, source_rate = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform.squeeze(0).cpu().numpy().astype(np.float32)


def _prepare_voice_conversion_target(speaker_files: list[Path], output_path: Path, sample_rate: int = 24000) -> Path:
    np = _import_numpy()
    sf = _import_soundfile()
    if len(speaker_files) == 1:
        return speaker_files[0]

    pause = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    parts: list = []
    for path in speaker_files:
        audio = _load_audio_mono(path, sample_rate)
        peak = np.max(np.abs(audio)) if len(audio) else 0.0
        if peak > 0:
            audio = audio / peak * 0.8
        if parts:
            parts.append(pause)
        parts.append(audio)

    merged_path = output_path / "merged_reference.wav"
    sf.write(str(merged_path), np.concatenate(parts), sample_rate)
    return merged_path


def _split_text(text: str, max_chars: int = 220) -> list[str]:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start : start + max_chars])
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks or [text]


def _get_mms(language: str):
    from transformers import AutoTokenizer, VitsModel

    mms_code = _resolve_mms_code(language)
    if mms_code not in _mms_models:
        model_id = f"facebook/mms-tts-{mms_code}"
        print(f"Loading local MMS-TTS model: {model_id}")
        model = VitsModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        sample_rate = int(model.config.sampling_rate)
        device = _get_mms_device()
        model = model.to(device)
        model.eval()
        _mms_models[mms_code] = (model, tokenizer, sample_rate)

    return _mms_models[mms_code]


def _synthesize_mms_chunk(text: str, language: str):
    torch = _import_torch()
    np = _import_numpy()
    model, tokenizer, sample_rate = _get_mms(language)
    device = _get_mms_device()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        waveform = model(**inputs).waveform.squeeze().cpu().numpy()
    return waveform.astype(np.float32), sample_rate


def _synthesize_mms(text: str, language: str):
    np = _import_numpy()
    chunks = _split_text(text, max_chars=180)
    parts: list = []
    sample_rate = 16000

    for index, chunk in enumerate(chunks):
        print(f"  MMS chunk {index + 1}/{len(chunks)}")
        audio, sample_rate = _synthesize_mms_chunk(chunk, language)
        if parts:
            parts.append(np.zeros(int(sample_rate * 0.2), dtype=np.float32))
        parts.append(audio)

    return np.concatenate(parts), sample_rate


_torch_load_patched = False


def _enable_trusted_torch_load() -> None:
    """Force ``weights_only=False`` for Coqui's ``torch.load`` calls.

    PyTorch >=2.6 defaults ``weights_only=True``, which refuses to unpickle the
    XTTS/FreeVC config classes shipped inside the official Coqui checkpoints.
    These models come from a trusted source, so full unpickling is safe here.
    """
    global _torch_load_patched
    if _torch_load_patched:
        return

    torch = _import_torch()
    original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _patched_load
    _torch_load_patched = True


def _get_xtts():
    global _xtts
    if _xtts is None:
        _enable_trusted_torch_load()
        from TTS.api import TTS

        print(f"Loading local XTTS model: {_XTTS_MODEL}")
        _xtts = TTS(_XTTS_MODEL).to(_get_xtts_device())
    return _xtts


def _wavlm_checkpoint_path() -> Path:
    from trainer.io import get_user_data_dir

    return Path(get_user_data_dir("tts")) / "wavlm" / "WavLM-Large.pt"


def _wavlm_checkpoint_valid(path: Path) -> bool:
    torch = _import_torch()
    if not path.is_file() or path.stat().st_size < _WAVLM_MIN_BYTES:
        return False
    try:
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def _download_wavlm_checkpoint(path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(".pt.part")

    print("Downloading local WavLM checkpoint (~1.2 GB, one-time only)...")
    with urllib.request.urlopen(_WAVLM_URI) as response:
        expected_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with partial_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if expected_size:
                    pct = downloaded * 100 // expected_size
                    print(f"  {pct}% ({downloaded // (1024 * 1024)} MB)", end="\r", flush=True)
        print()

    if expected_size and downloaded < expected_size:
        partial_path.unlink(missing_ok=True)
        raise TTSError(
            f"WavLM download incomplete ({downloaded}/{expected_size} bytes). "
            "Check your internet connection and try again."
        )

    partial_path.replace(path)


def _ensure_wavlm_checkpoint() -> None:
    path = _wavlm_checkpoint_path()
    if _wavlm_checkpoint_valid(path):
        return

    if path.exists():
        print(f"Removing corrupted WavLM checkpoint at {path}")
        path.unlink()

    _download_wavlm_checkpoint(path)
    if not _wavlm_checkpoint_valid(path):
        path.unlink(missing_ok=True)
        raise TTSError("Downloaded WavLM checkpoint is still invalid.")


def _get_vc():
    global _vc
    if _vc is None:
        _enable_trusted_torch_load()
        from TTS.api import TTS

        _ensure_wavlm_checkpoint()
        print(f"Loading local voice conversion model: {_VC_MODEL}")
        _vc = TTS(_VC_MODEL).to(_get_vc_device())
    return _vc


def _voice_convert(source_wav: Path, target_wav: Path, output_path: Path) -> None:
    print("Applying local voice conversion to match reference voice...")
    try:
        _get_vc().voice_conversion_to_file(
            source_wav=str(source_wav),
            target_wav=str(target_wav),
            file_path=str(output_path),
        )
    except RuntimeError as exc:
        global _vc
        if "zip archive" in str(exc).lower() or "pytorchstreamreader" in str(exc).lower():
            print("Voice conversion model was corrupted. Re-downloading and retrying...")
            _vc = None
            if _wavlm_checkpoint_path().exists():
                _wavlm_checkpoint_path().unlink()
            _get_vc().voice_conversion_to_file(
                source_wav=str(source_wav),
                target_wav=str(target_wav),
                file_path=str(output_path),
            )
        else:
            raise


def _synthesize_xtts(
    text: str,
    speaker_files: list[Path],
    output_path: Path,
    language: str,
    speaker_id: str | None,
    gpt_cond_len: int,
) -> None:
    print("Synthesizing with local XTTS voice cloning...")
    kwargs = {
        "text": text,
        "speaker_wav": [str(path) for path in speaker_files]
        if len(speaker_files) > 1
        else str(speaker_files[0]),
        "language": language,
        "file_path": str(output_path),
        "gpt_cond_len": gpt_cond_len,
    }
    if speaker_id:
        kwargs["speaker"] = speaker_id
    _get_xtts().tts_to_file(**kwargs)


def _synthesize_mms_only(text: str, output_path: Path, language: str) -> None:
    sf = _import_soundfile()
    print("Synthesizing with local MMS-TTS (no voice cloning)...")
    audio, sample_rate = _synthesize_mms(text, language)
    sf.write(str(output_path), audio, sample_rate)


def _synthesize_mms_vc(
    text: str,
    speaker_files: list[Path],
    output_path: Path,
    language: str,
) -> None:
    sf = _import_soundfile()
    print("Synthesizing with local MMS-TTS + voice conversion...")
    audio, sample_rate = _synthesize_mms(text, language)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        source_wav = tmp / "source.wav"
        sf.write(str(source_wav), audio, sample_rate)
        target_wav = _prepare_voice_conversion_target(speaker_files, tmp)
        _voice_convert(source_wav, target_wav, output_path)


def synthesize_speech(
    text: str,
    output_path: Path | str,
    speaker_wav: Path | str | list[Path | str] | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    backend: str = "auto",
    clone: bool = True,
    gpt_cond_len: int = 30,
) -> tuple[Path, str, str]:
    """Synthesize speech from text using fully local models.

    Returns (output_path, resolved_language, selected_backend).
    """
    output_path = Path(output_path)
    text = text.strip()
    if not text:
        raise TTSError("Text must not be empty.")

    resolved_language = _resolve_language(text, language)
    selected_backend = _backend_for_language(resolved_language, backend, clone)
    print(
        f"Language: {resolved_language} | Backend: {selected_backend} | "
        f"Clone: {clone} | Device: {_get_device()}"
    )

    if selected_backend == "mms":
        _synthesize_mms_only(text, output_path, resolved_language)
        return output_path, resolved_language, selected_backend

    if speaker_wav is None:
        speaker_wav = _default_speaker_wavs()
    speaker_files = _normalize_speaker_wav(speaker_wav)

    if selected_backend == "mms_vc":
        _synthesize_mms_vc(text, speaker_files, output_path, resolved_language)
    elif selected_backend == "xtts":
        _synthesize_xtts(
            text,
            speaker_files,
            output_path,
            resolved_language,
            speaker_id,
            gpt_cond_len,
        )
    else:
        raise TTSError(f"Unknown backend: {selected_backend}")

    return output_path, resolved_language, selected_backend
