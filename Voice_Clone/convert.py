from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import edge_tts
import numpy as np
import soundfile as sf
import torch
import torchaudio
from TTS.api import TTS

_AUDIO_DIR = Path(__file__).resolve().parent
VOICES_DIR = _AUDIO_DIR / "voices"
SUBTITLES_FILE = _AUDIO_DIR / "subtitles.txt"
OUTPUT_AUDIO_FILE = _AUDIO_DIR / "voiceover.wav"

_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
_VC_MODEL = "voice_conversion_models/multilingual/vctk/freevc24"
_WAVLM_URI = "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/WavLM-Large.pt"
_WAVLM_MIN_BYTES = 1_200_000_000
_SAMPLE_RATE = 24000


_EDGE_TTS_VOICES = {
    "ta": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "kn": "kn-IN-GaganNeural",
    "ml": "ml-IN-MidhunNeural",
    "bn": "bn-IN-BashkarNeural",
    "gu": "gu-IN-NiranjanNeural",
    "mr": "mr-IN-ManoharNeural",
    "pa": "pa-IN-OjasNeural",
}

_XTTS_LANGUAGES = frozenset(
    {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"}
)

_xtts: TTS | None = None
_vc: TTS | None = None


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_xtts() -> TTS:
    global _xtts
    if _xtts is None:
        _xtts = TTS(_XTTS_MODEL).to(_get_device())
    return _xtts


def _wavlm_checkpoint_path() -> Path:
    from trainer.io import get_user_data_dir

    return Path(get_user_data_dir("tts")) / "wavlm" / "WavLM-Large.pt"


def _wavlm_checkpoint_valid(path: Path) -> bool:
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

    print("Downloading WavLM checkpoint (~1.2 GB). This may take a few minutes...")
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

    if expected_size and downloaded < expected_size:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
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
        raise RuntimeError("Downloaded WavLM checkpoint is still invalid.")


def _get_vc() -> TTS:
    global _vc
    if _vc is None:
        _ensure_wavlm_checkpoint()
        _vc = TTS(_VC_MODEL).to(_get_device())
    return _vc


def detect_language(text: str) -> str:
    """Detect the primary language of subtitle text."""
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    lang = detect(text)
    if lang.startswith("zh"):
        return "zh-cn"
    return lang


def _resolve_language(text: str, language: str | None) -> str:
    if language:
        return language
    return detect_language(text)


def _backend_for_language(language: str) -> str:
    if language in _XTTS_LANGUAGES:
        return "xtts"
    if language in _EDGE_TTS_VOICES:
        return "edge_vc"
    raise ValueError(
        f"Language '{language}' is not supported. "
        f"Indian languages: {', '.join(sorted(_EDGE_TTS_VOICES))}. "
        f"International languages: {', '.join(sorted(_XTTS_LANGUAGES))}."
    )


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
        raise FileNotFoundError(
            "Reference voice file(s) not found: "
            + ", ".join(missing)
            + ". Record 6–30 seconds of clean speech and save it under audio/voices/."
        )
    return resolved


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


async def _edge_tts_save(text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def _mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    waveform, sample_rate = torchaudio.load(str(mp3_path))
    if sample_rate != _SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, _SAMPLE_RATE)
    sf.write(str(wav_path), waveform.squeeze().numpy(), _SAMPLE_RATE)


def _synthesize_xtts(
    text: str,
    speaker_files: list[Path],
    output_path: Path,
    language: str,
    speaker_id: str | None,
    gpt_cond_len: int,
) -> None:
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


def _synthesize_edge_vc(
    text: str,
    speaker_wav: Path,
    output_path: Path,
    language: str,
    voice: str | None = None,
) -> None:
    edge_voice = voice or _EDGE_TTS_VOICES.get(language)
    if not edge_voice:
        raise ValueError(f"No edge-tts voice configured for language '{language}'")

    chunks = _split_text(text, max_chars=400)
    pause = np.zeros(int(_SAMPLE_RATE * 0.25), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        merged_parts: list[np.ndarray] = []

        for index, chunk in enumerate(chunks):
            mp3_path = tmp / f"chunk_{index}.mp3"
            wav_path = tmp / f"chunk_{index}.wav"
            asyncio.run(_edge_tts_save(chunk, edge_voice, mp3_path))
            _mp3_to_wav(mp3_path, wav_path)
            audio, _ = sf.read(str(wav_path), dtype="float32")
            if merged_parts:
                merged_parts.append(pause)
            merged_parts.append(audio)

        source_wav = tmp / "source.wav"
        sf.write(str(source_wav), np.concatenate(merged_parts), _SAMPLE_RATE)

        try:
            _get_vc().voice_conversion_to_file(
                source_wav=str(source_wav),
                target_wav=str(speaker_wav),
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
                    target_wav=str(speaker_wav),
                    file_path=str(output_path),
                )
            else:
                raise


def subtitles_to_speech(
    subtitles_path: Path | str = SUBTITLES_FILE,
    output_path: Path | str = OUTPUT_AUDIO_FILE,
    speaker_wav: Path | str | list[Path | str] | None = None,
    speaker_id: str | None = None,
    language: str | None = None,
    voice: str | None = None,
    gpt_cond_len: int = 30,
) -> Path:
    """Read subtitle text and synthesize speech in a cloned voice.

    Language is auto-detected from the subtitle text when `language` is omitted.
    Tamil and other Indian languages use edge-tts + voice conversion; English,
    Hindi, and other international languages use XTTS voice cloning.
    """
    subtitles_path = Path(subtitles_path)
    output_path = Path(output_path)

    if speaker_wav is None:
        speaker_wav = VOICES_DIR / "reference.wav"

    text = subtitles_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text found in {subtitles_path}")

    speaker_files = _normalize_speaker_wav(speaker_wav)
    resolved_language = _resolve_language(text, language)
    backend = _backend_for_language(resolved_language)

    if backend == "edge_vc":
        _synthesize_edge_vc(
            text,
            speaker_files[0],
            output_path,
            resolved_language,
            voice=voice,
        )
    else:
        _synthesize_xtts(
            text,
            speaker_files,
            output_path,
            resolved_language,
            speaker_id,
            gpt_cond_len,
        )

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert subtitles to cloned speech.")
    parser.add_argument(
        "speaker_wav",
        nargs="?",
        default=str(VOICES_DIR / "reference.wav"),
        help="Reference voice recording",
    )
    parser.add_argument(
        "--language",
        "-l",
        help="Subtitle language code (e.g. ta, en, hi). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--voice",
        help="Override edge-tts voice (e.g. ta-IN-PallaviNeural for Tamil).",
    )
    args = parser.parse_args()

    result = subtitles_to_speech(
        speaker_wav=args.speaker_wav,
        language=args.language,
        voice=args.voice,
    )
    print(f"Saved cloned speech audio to {result}")
