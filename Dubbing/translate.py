"""Per-segment translation.

Backends (TRANSLATOR env var):
  local  - facebook/m2m100_418M run locally with transformers (default).
           MIT licensed, offline, no rate limits, ~100 languages incl.
           hi, ta, bn, gu, mr, ml, kn (not te, pa).
  google - free Google Translate endpoint via deep-translator. Unofficial:
           rate-limited and may answer with a bot check (HTTP 429).
"""
import os

TRANSLATOR = os.getenv("TRANSLATOR", "local")
LOCAL_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/m2m100_418M")

# Whisper language codes -> Google Translate codes where they differ.
_GOOGLE_CODES = {"zh": "zh-CN", "zh-cn": "zh-CN", "he": "iw"}

_local = None


class TranslationError(Exception):
    pass


def _base(lang):
    return (lang or "").lower().split("-")[0]


def translate_texts(texts, target_language, source_language=None, backend=None):
    if not texts:
        return []
    if source_language and _base(source_language) == _base(target_language):
        return list(texts)
    backend = backend or TRANSLATOR
    if backend == "local":
        return _translate_local(texts, target_language, source_language)
    if backend == "google":
        return _translate_google(texts, target_language, source_language)
    raise ValueError(f"Unknown translator '{backend}' (use 'local' or 'google')")


def _get_local():
    global _local
    if _local is None:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        print(f"Loading translation model '{LOCAL_MODEL}'...")
        tokenizer = M2M100Tokenizer.from_pretrained(LOCAL_MODEL)
        model = M2M100ForConditionalGeneration.from_pretrained(LOCAL_MODEL)
        model.eval()
        _local = (tokenizer, model)
    return _local


def _translate_local(texts, target_language, source_language):
    import torch

    if not source_language:
        raise TranslationError("The local translator needs the source language")
    tokenizer, model = _get_local()
    src, tgt = _base(source_language), _base(target_language)
    for lang in (src, tgt):
        if lang not in tokenizer.lang_code_to_id:
            raise TranslationError(
                f"Language '{lang}' is not supported by {LOCAL_MODEL}; "
                "try TRANSLATOR=google")

    tokenizer.src_lang = src
    out = []
    for text in texts:
        batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            generated = model.generate(
                **batch,
                forced_bos_token_id=tokenizer.get_lang_id(tgt),
                num_beams=4,
            )
        out.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip() or text)
    return out


def _translate_google(texts, target_language, source_language):
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import TooManyRequests

    source = _GOOGLE_CODES.get((source_language or "").lower(), _base(source_language)) or "auto"
    target = _GOOGLE_CODES.get(target_language.lower(), _base(target_language))
    translator = GoogleTranslator(source=source, target=target)
    out = []
    for text in texts:
        try:
            translated = translator.translate(text)
        except TooManyRequests as exc:
            raise TranslationError(
                "Google Translate refused the request (rate limit / bot check). "
                "Use TRANSLATOR=local or a licensed translation API.") from exc
        out.append((translated or text).strip())
    return out
