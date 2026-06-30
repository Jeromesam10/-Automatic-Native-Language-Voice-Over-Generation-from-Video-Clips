from __future__ import annotations

import re
from dataclasses import dataclass

_TRANSLATE_SYSTEM = (
    "You are a translation engine. Return only the final translated text. "
    "Do not explain, summarize, add labels, add markdown, or include reasoning. "
    "Preserve every detail, sentence, meaning, and tone from the source text."
)

_THINK_BLOCK_RE = re.compile(
    r"\x3cthink\x3e.*?\x3c/think\x3e|<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)
_RESPONSE_PREFIX_RE = re.compile(
    r"^\s*(translation|translated text|output)\s*:\s*",
    re.IGNORECASE,
)

_SCRIPT_THRESHOLDS = {
    "default": 0.4,
    "latin": 0.0,
}


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    native_name: str
    script_pattern: re.Pattern[str] | None = None
    script_threshold: float = _SCRIPT_THRESHOLDS["default"]


def _profile(
    code: str,
    name: str,
    native_name: str,
    script: str | None = None,
    *,
    script_threshold: float | None = None,
) -> LanguageProfile:
    pattern = re.compile(script) if script else None
    threshold = (
        _SCRIPT_THRESHOLDS["latin"]
        if pattern is None
        else (script_threshold or _SCRIPT_THRESHOLDS["default"])
    )
    return LanguageProfile(
        code=code,
        name=name,
        native_name=native_name,
        script_pattern=pattern,
        script_threshold=threshold,
    )


_LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "ta": _profile("ta", "Tamil", "தமிழ்", r"[\u0B80-\u0BFF]"),
    "te": _profile("te", "Telugu", "తెలుగు", r"[\u0C00-\u0C7F]"),
    "kn": _profile("kn", "Kannada", "ಕನ್ನಡ", r"[\u0C80-\u0CFF]"),
    "ml": _profile("ml", "Malayalam", "മലയാളം", r"[\u0D00-\u0D7F]"),
    "hi": _profile("hi", "Hindi", "हिन्दी", r"[\u0900-\u097F]"),
    "mr": _profile("mr", "Marathi", "मराठी", r"[\u0900-\u097F]"),
    "bn": _profile("bn", "Bengali", "বাংলা", r"[\u0980-\u09FF]"),
    "gu": _profile("gu", "Gujarati", "ગુજરાતી", r"[\u0A80-\u0AFF]"),
    "pa": _profile("pa", "Punjabi", "ਪੰਜਾਬੀ", r"[\u0A00-\u0A7F]"),
    "or": _profile("or", "Odia", "ଓଡ଼ିଆ", r"[\u0B00-\u0B7F]"),
    "as": _profile("as", "Assamese", "অসমীয়া", r"[\u0980-\u09FF]"),
    "si": _profile("si", "Sinhala", "සිංහල", r"[\u0D80-\u0DFF]"),
    "ur": _profile("ur", "Urdu", "اردو", r"[\u0600-\u06FF]"),
    "ne": _profile("ne", "Nepali", "नेपाली", r"[\u0900-\u097F]"),
    "ar": _profile("ar", "Arabic", "العربية", r"[\u0600-\u06FF]"),
    "fa": _profile("fa", "Persian", "فارسی", r"[\u0600-\u06FF]"),
    "he": _profile("he", "Hebrew", "עברית", r"[\u0590-\u05FF]"),
    "ja": _profile("ja", "Japanese", "日本語", r"[\u3040-\u30FF\u4E00-\u9FFF]"),
    "ko": _profile("ko", "Korean", "한국어", r"[\uAC00-\uD7AF\u1100-\u11FF]"),
    "zh": _profile("zh", "Chinese", "中文", r"[\u4E00-\u9FFF]"),
    "zh-cn": _profile("zh-cn", "Chinese (Simplified)", "简体中文", r"[\u4E00-\u9FFF]"),
    "zh-tw": _profile("zh-tw", "Chinese (Traditional)", "繁體中文", r"[\u4E00-\u9FFF]"),
    "th": _profile("th", "Thai", "ไทย", r"[\u0E00-\u0E7F]"),
    "km": _profile("km", "Khmer", "ខ្មែរ", r"[\u1780-\u17FF]"),
    "lo": _profile("lo", "Lao", "ລາວ", r"[\u0E80-\u0EDF]"),
    "my": _profile("my", "Burmese", "မြန်မာ", r"[\u1000-\u109F]"),
    "ru": _profile("ru", "Russian", "Русский", r"[\u0400-\u04FF]"),
    "uk": _profile("uk", "Ukrainian", "Українська", r"[\u0400-\u04FF]"),
    "bg": _profile("bg", "Bulgarian", "Български", r"[\u0400-\u04FF]"),
    "sr": _profile("sr", "Serbian", "Српски", r"[\u0400-\u04FF\u0500-\u052F]"),
    "mk": _profile("mk", "Macedonian", "Македонски", r"[\u0400-\u04FF]"),
    "be": _profile("be", "Belarusian", "Беларуская", r"[\u0400-\u04FF]"),
    "el": _profile("el", "Greek", "Ελληνικά", r"[\u0370-\u03FF]"),
    "en": _profile("en", "English", "English"),
    "fr": _profile("fr", "French", "Français"),
    "de": _profile("de", "German", "Deutsch"),
    "es": _profile("es", "Spanish", "Español"),
    "it": _profile("it", "Italian", "Italiano"),
    "pt": _profile("pt", "Portuguese", "Português"),
    "nl": _profile("nl", "Dutch", "Nederlands"),
    "pl": _profile("pl", "Polish", "Polski"),
    "tr": _profile("tr", "Turkish", "Türkçe"),
    "vi": _profile("vi", "Vietnamese", "Tiếng Việt", r"[\u00C0-\u1EF9]"),
    "id": _profile("id", "Indonesian", "Bahasa Indonesia"),
    "ms": _profile("ms", "Malay", "Bahasa Melayu"),
    "fil": _profile("fil", "Filipino", "Filipino"),
    "sw": _profile("sw", "Swahili", "Kiswahili"),
    "am": _profile("am", "Amharic", "አማርኛ", r"[\u1200-\u137F]"),
}

_LANGUAGE_ALIASES: dict[str, str] = {
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "hindi": "hi",
    "marathi": "mr",
    "bengali": "bn",
    "bangla": "bn",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "oriya": "or",
    "assamese": "as",
    "sinhala": "si",
    "urdu": "ur",
    "nepali": "ne",
    "arabic": "ar",
    "persian": "fa",
    "farsi": "fa",
    "hebrew": "he",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "mandarin": "zh",
    "simplified chinese": "zh-cn",
    "traditional chinese": "zh-tw",
    "thai": "th",
    "khmer": "km",
    "cambodian": "km",
    "lao": "lo",
    "burmese": "my",
    "myanmar": "my",
    "russian": "ru",
    "ukrainian": "uk",
    "bulgarian": "bg",
    "serbian": "sr",
    "macedonian": "mk",
    "belarusian": "be",
    "greek": "el",
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "dutch": "nl",
    "polish": "pl",
    "turkish": "tr",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "filipino": "fil",
    "tagalog": "fil",
    "swahili": "sw",
    "amharic": "am",
}


def resolve_target_language(target_language: str) -> LanguageProfile:
    normalized = target_language.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("Field 'target_language' is required.")

    code = _LANGUAGE_ALIASES.get(normalized, normalized)
    if code in _LANGUAGE_PROFILES:
        return _LANGUAGE_PROFILES[code]

    display_name = target_language.strip()
    return LanguageProfile(
        code=normalized,
        name=display_name,
        native_name=display_name,
    )


def list_supported_translation_languages() -> list[dict[str, str]]:
    return [
        {
            "code": profile.code,
            "name": profile.name,
            "native_name": profile.native_name,
        }
        for profile in sorted(
            _LANGUAGE_PROFILES.values(),
            key=lambda item: item.name,
        )
    ]


def _native_script_rule(profile: LanguageProfile) -> str:
    if profile.script_pattern is None:
        return (
            f"- Write in natural {profile.name} ({profile.native_name}).\n"
            "- Do not return the source text unchanged."
        )
    return (
        f"- Write only in native {profile.name} ({profile.native_name}) script.\n"
        "- Do not transliterate into English/Latin letters.\n"
        "- If the answer is mostly English words, it is wrong."
    )


def build_translation_prompt(
    text: str,
    profile: LanguageProfile,
    source_language: str | None = None,
) -> str:
    source_note = (
        f"Source language: {source_language}"
        if source_language
        else "Source language: auto-detect"
    )

    instructions = (
        f"Translate the text below into {profile.name} ({profile.native_name}).\n"
        f"{source_note}\n"
        "Rules:\n"
        "- Return only the translated text.\n"
        "- Keep the full meaning and all visual details.\n"
        "- Do not shorten the text.\n"
        "- Do not include the original text.\n"
        f"{_native_script_rule(profile)}\n\n"
        "Text to translate:\n"
    )

    if source_language:
        instructions = instructions.replace(
            "Translate the text below",
            f"Translate the text below from {source_language}",
        )

    return f"{instructions}{text}\n\nTranslated text:"


def build_translation_retry_prompt(
    text: str,
    profile: LanguageProfile,
    source_language: str | None = None,
) -> str:
    source_note = f" from {source_language}" if source_language else ""
    script_note = (
        f"Every sentence must be written in native {profile.name} "
        f"({profile.native_name}) script."
        if profile.script_pattern is not None
        else f"Write in natural {profile.name} ({profile.native_name})."
    )
    return (
        "The previous answer was invalid because it was not translated into "
        f"{profile.name} ({profile.native_name}).\n"
        f"Translate this text{source_note} into {profile.name} now.\n"
        "Return only the translation. Do not use English except unavoidable proper nouns.\n"
        f"{script_note}\n\n"
        f"{text}\n\n"
        f"{profile.name} translation:"
    )


def translation_system_prompt() -> str:
    return _TRANSLATE_SYSTEM


def clean_translation_response(response: str) -> str:
    response = _THINK_BLOCK_RE.sub("", response).strip()
    response = _RESPONSE_PREFIX_RE.sub("", response).strip()

    if (
        len(response) >= 2
        and response[0] == response[-1]
        and response[0] in {'"', "'"}
    ):
        response = response[1:-1].strip()

    return response


def needs_translation_retry(response: str, profile: LanguageProfile) -> bool:
    if profile.script_pattern is None or profile.script_threshold <= 0:
        return False

    letters = [char for char in response if char.isalpha()]
    if not letters:
        return True

    script_letters = profile.script_pattern.findall(response)
    return len(script_letters) / len(letters) < profile.script_threshold
