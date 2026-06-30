from django.core.management.base import BaseCommand

from voiceover.services import DeepSeekClient
from voiceover.services.deepseek import DeepSeekError


class Command(BaseCommand):
    help = "Test the local DeepSeek/Ollama connection."

    def add_arguments(self, parser):
        parser.add_argument(
            "prompt",
            nargs="?",
            default="Write a one-sentence voice-over intro for a tech podcast.",
            help="Prompt to send to the model.",
        )

    def handle(self, *args, **options):
        client = DeepSeekClient()

        if not client.is_available():
            self.stderr.write(
                self.style.ERROR(
                    "Ollama is not reachable. Start it with: brew services start ollama"
                )
            )
            return

        self.stdout.write(f"Using model: {client.model}")

        try:
            response = client.generate(options["prompt"])
        except DeepSeekError as exc:
            self.stderr.write(self.style.ERROR(str(exc)))
            return

        self.stdout.write(self.style.SUCCESS(response))
