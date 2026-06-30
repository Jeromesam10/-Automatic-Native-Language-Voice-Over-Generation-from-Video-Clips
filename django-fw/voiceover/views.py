import base64
import json
import tempfile
from pathlib import Path

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from voiceover.services import DeepSeekClient
from voiceover.services.deepseek import ChatMessage, DeepSeekError
from voiceover.services.prompts import PromptValidationError, build_prompt
from voiceover.services.translate import (
    build_translation_retry_prompt,
    build_translation_prompt,
    clean_translation_response,
    list_supported_translation_languages,
    needs_translation_retry,
    resolve_target_language,
    translation_system_prompt,
)


def _json_error(message: str, status: int = 400) -> JsonResponse:
    return JsonResponse({"error": message}, status=status)


def _stream_wav_and_cleanup(path: Path):
    try:
        with path.open("rb") as handle:
            yield from handle
    finally:
        path.unlink(missing_ok=True)


@require_GET
def health(request):
    client = DeepSeekClient()
    if not client.is_available():
        return _json_error(
            "Ollama is not running. Start it with: brew services start ollama",
            status=503,
        )

    return JsonResponse(
        {
            "status": "ok",
            "model": client.model,
            "models": client.list_models(),
        }
    )


@csrf_exempt
@require_POST
def summarize(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("Request body must be valid JSON.")

    try:
        prompt = build_prompt(payload.get("prompt"))
    except PromptValidationError as exc:
        return _json_error(str(exc))

    system = payload.get("system")
    if isinstance(system, str):
        system = system.strip() or None
    else:
        system = None
    client = DeepSeekClient()

    try:
        response = client.generate(prompt=prompt, system=system)
    except DeepSeekError as exc:
        return _json_error(str(exc), status=503)

    return JsonResponse({"model": client.model, "response": response})


@csrf_exempt
@require_POST
def chat(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("Request body must be valid JSON.")

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return _json_error("Field 'messages' must be a non-empty list.")

    messages: list[ChatMessage] = []
    for item in raw_messages:
        role = item.get("role", "").strip()
        content = item.get("content", "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            return _json_error(
                "Each message needs a valid role and non-empty content."
            )
        messages.append(ChatMessage(role=role, content=content))

    client = DeepSeekClient()

    try:
        response = client.chat(messages)
    except DeepSeekError as exc:
        return _json_error(str(exc), status=503)

    return JsonResponse({"model": client.model, "response": response})


@csrf_exempt
@require_POST
def translate(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return _json_error("Request body must be valid JSON.")

    text = (payload.get("text") or "").strip()
    if not text:
        return _json_error("Field 'text' is required and must not be empty.")

    target_language = (payload.get("target_language") or "").strip()
    if not target_language:
        return _json_error("Field 'target_language' is required.")

    try:
        target_profile = resolve_target_language(target_language)
    except ValueError as exc:
        return _json_error(str(exc))

    source_language = payload.get("source_language")
    if isinstance(source_language, str):
        source_language = source_language.strip() or None
    else:
        source_language = None

    prompt = build_translation_prompt(text, target_profile, source_language)
    client = DeepSeekClient()

    try:
        response = client.chat(
            [
                ChatMessage(role="system", content=translation_system_prompt()),
                ChatMessage(role="user", content=prompt),
            ]
        )
    except DeepSeekError as exc:
        return _json_error(str(exc), status=503)

    response = clean_translation_response(response)
    if needs_translation_retry(response, target_profile):
        retry_prompt = build_translation_retry_prompt(
            text,
            target_profile,
            source_language,
        )
        try:
            response = client.chat(
                [
                    ChatMessage(role="system", content=translation_system_prompt()),
                    ChatMessage(role="user", content=retry_prompt),
                ]
            )
        except DeepSeekError as exc:
            return _json_error(str(exc), status=503)
        response = clean_translation_response(response)

    result = {
        "model": client.model,
        "target_language": target_profile.name,
        "target_language_code": target_profile.code,
        "target_language_native": target_profile.native_name,
        "response": response,
    }
    if source_language:
        result["source_language"] = source_language

    return JsonResponse(result)


_VALID_BACKENDS = {"auto", "mms_vc", "mms", "xtts"}


@csrf_exempt
@require_POST
def generate_voiceover(request):
    """Convert narration text to speech using local TTS models."""
    from voiceover.services.tts import TTSError, synthesize_speech

    speaker_wav: Path | str | list[Path] | None = None

    content_type = request.content_type or ""
    if content_type.startswith("multipart/form-data"):
        text = (request.POST.get("text") or "").strip()
        language = (request.POST.get("language") or "").strip() or None
        backend = (request.POST.get("backend") or "auto").strip().lower()
        clone_raw = (request.POST.get("clone") or "true").strip().lower()
        response_format = (request.POST.get("response_format") or "audio").strip().lower()
        gpt_cond_len_raw = request.POST.get("gpt_cond_len", "30")

        uploaded = request.FILES.get("speaker_wav")
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp.close()
            speaker_wav = Path(tmp.name)
    else:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return _json_error("Request body must be valid JSON.")

        text = (payload.get("text") or "").strip()
        language = payload.get("language")
        if isinstance(language, str):
            language = language.strip() or None
        else:
            language = None

        backend = (payload.get("backend") or "auto").strip().lower()
        clone_raw = payload.get("clone", True)
        response_format = (payload.get("response_format") or "audio").strip().lower()
        gpt_cond_len_raw = payload.get("gpt_cond_len", 30)

    if not text:
        return _json_error("Field 'text' is required and must not be empty.")

    if backend not in _VALID_BACKENDS:
        return _json_error(
            f"Field 'backend' must be one of: {', '.join(sorted(_VALID_BACKENDS))}."
        )

    if isinstance(clone_raw, bool):
        clone = clone_raw
    else:
        clone = str(clone_raw).lower() not in {"0", "false", "no"}

    try:
        gpt_cond_len = int(gpt_cond_len_raw)
    except (TypeError, ValueError):
        return _json_error("Field 'gpt_cond_len' must be an integer.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        _, resolved_language, selected_backend = synthesize_speech(
            text=text,
            output_path=output_path,
            speaker_wav=speaker_wav,
            language=language,
            backend=backend,
            clone=clone,
            gpt_cond_len=gpt_cond_len,
        )
    except TTSError as exc:
        output_path.unlink(missing_ok=True)
        return _json_error(str(exc))
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return _json_error(f"TTS synthesis failed: {exc}", status=503)
    finally:
        if speaker_wav and isinstance(speaker_wav, Path):
            speaker_wav.unlink(missing_ok=True)

    if response_format == "json":
        audio_bytes = output_path.read_bytes()
        output_path.unlink(missing_ok=True)
        return JsonResponse(
            {
                "language": resolved_language,
                "backend": selected_backend,
                "clone": clone,
                "content_type": "audio/wav",
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
            }
        )

    response = StreamingHttpResponse(
        _stream_wav_and_cleanup(output_path),
        content_type="audio/wav",
    )
    response["Content-Disposition"] = 'inline; filename="voiceover.wav"'
    response["X-Voiceover-Language"] = resolved_language
    response["X-Voiceover-Backend"] = selected_backend
    return response


@require_GET
def list_tts_languages(request):
    from voiceover.services.tts import list_supported_languages

    languages = list_supported_languages()
    return JsonResponse(languages)


@require_GET
def list_translation_languages(request):
    return JsonResponse({"languages": list_supported_translation_languages()})
