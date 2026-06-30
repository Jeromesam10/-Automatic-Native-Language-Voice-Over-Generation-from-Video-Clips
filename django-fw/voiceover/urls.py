from django.urls import path

from voiceover import views

urlpatterns = [
    path("health", views.health, name="voiceover-health-no-slash"),
    path("health/", views.health, name="voiceover-health"),
    path("summarize", views.summarize, name="voiceover-summarize-no-slash"),
    path("summarize/", views.summarize, name="voiceover-summarize"),
    path("chat", views.chat, name="voiceover-chat-no-slash"),
    path("chat/", views.chat, name="voiceover-chat"),
    path("translate", views.translate, name="voiceover-translate-no-slash"),
    path("translate/", views.translate, name="voiceover-translate"),
    path(
        "generate-voiceover",
        views.generate_voiceover,
        name="voiceover-generate-no-slash",
    ),
    path(
        "generate-voiceover/",
        views.generate_voiceover,
        name="voiceover-generate",
    ),
    path(
        "languages",
        views.list_tts_languages,
        name="voiceover-languages-no-slash",
    ),
    path(
        "languages/",
        views.list_tts_languages,
        name="voiceover-languages",
    ),
    path(
        "translation-languages",
        views.list_translation_languages,
        name="voiceover-translation-languages-no-slash",
    ),
    path(
        "translation-languages/",
        views.list_translation_languages,
        name="voiceover-translation-languages",
    ),
]
