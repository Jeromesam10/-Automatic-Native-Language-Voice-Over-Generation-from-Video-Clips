# Voice Over Gen

Django API for generating voice-over text using a **local DeepSeek model** served by [Ollama](https://ollama.com).

## Features

- Local LLM inference (no cloud API keys required)
- REST API for health checks, generation, and chat
- Structured prompt support for multi-frame video descriptions
- One-command server setup via `setup.sh`

## Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| OS | Ubuntu 22.04+ / Debian 12+ / macOS | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB free | 20 GB+ |
| Python | 3.11 | 3.11 |

> Voice cloning uses Coqui TTS, which currently supports Python `<3.12`.
> Use a Python 3.11 virtual environment for the full voice-over audio pipeline.

### Model sizing guide

| RAM | Suggested model |
|-----|-----------------|
| 8 GB | `deepseek-r1:1.5b` |
| 12–16 GB | `deepseek-r1:7b` |
| 24 GB+ | `deepseek-r1:8b` |

---

## Quick start (one command)

On your server, clone the repo and run:

```bash
git clone <your-repo-url> voice-over-gen
cd voice-over-gen
chmod +x setup.sh
./setup.sh
```

Optional environment variables before running:

```bash
# Use a larger model
DEEPSEEK_MODEL=deepseek-r1:7b ./setup.sh

# Also install Nginx reverse proxy
INSTALL_NGINX=true SERVER_NAME=your-domain.com ./setup.sh
```

### Windows

On a Windows laptop, run the PowerShell script instead:

```powershell
git clone <your-repo-url> voice-over-gen
cd voice-over-gen
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Optional parameters:

```powershell
# Use a larger model
.\setup.ps1 -DeepSeekModel "deepseek-r1:7b"

# Change the app port
.\setup.ps1 -DjangoPort 8080
```

> Requires [Python 3.10+](https://www.python.org/downloads/) on PATH. Ollama is installed automatically via `winget` (or the official installer). If `ollama` is not found right after install, close and reopen PowerShell, then re-run the script.

The script will:

1. Install system packages (Linux)
2. Install and start Ollama
3. Pull the DeepSeek model
4. Create a Python virtual environment
5. Install Python dependencies
6. Create `.env` with a generated `SECRET_KEY`
7. Run Django migrations
8. Install a `systemd` service (Linux)
9. Optionally configure Nginx
10. Verify the API and DeepSeek connection

---

## Manual setup

Use this if you prefer step-by-step control.

### 1. Install Ollama

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama
sudo systemctl start ollama
```

**macOS:**

```bash
brew install ollama
brew services start ollama
```

### 2. Pull DeepSeek

```bash
ollama pull deepseek-r1:1.5b
```

Verify:

```bash
ollama list
curl http://127.0.0.1:11434/api/tags
```

### 3. Set up the Django app

```bash
cd voice-over-gen
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
SECRET_KEY=your-random-secret-key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,your-server-ip

DJANGO_PORT=8000
GUNICORN_WORKERS=2

OLLAMA_BASE_URL=http://127.0.0.1:11434
DEEPSEEK_MODEL=deepseek-r1:1.5b
OLLAMA_TIMEOUT=120
```

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

Run migrations:

```bash
python manage.py migrate
```

Test DeepSeek:

```bash
python manage.py test_deepseek "Write a one-sentence voice-over intro."
```

### 4. Run locally (development)

```bash
python manage.py runserver 0.0.0.0:8000
```

### 5. Deploy with Gunicorn + systemd (production)

Install the service file:

```bash
sudo cp deploy/voiceover-gen.service /etc/systemd/system/voiceover-gen.service
```

Update placeholders in the service file (`__APP_DIR__`, `__USER__`, etc.) or run `./setup.sh` which does this automatically.

```bash
sudo systemctl daemon-reload
sudo systemctl enable voiceover-gen
sudo systemctl start voiceover-gen
sudo systemctl status voiceover-gen
```

View logs:

```bash
sudo journalctl -u voiceover-gen -f
```

### 6. Optional: Nginx reverse proxy

```bash
sudo apt-get install -y nginx
sudo cp deploy/nginx-voiceover.conf /etc/nginx/sites-available/voiceover-gen
```

Edit `server_name` and port, then enable:

```bash
sudo ln -s /etc/nginx/sites-available/voiceover-gen /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

For HTTPS, add Certbot:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## API reference

Base URL: `http://<host>:8000/api/v1/voiceover`

### Health check

```bash
GET /api/v1/voiceover/health
```

Response:

```json
{
  "status": "ok",
  "model": "deepseek-r1:1.5b",
  "models": ["deepseek-r1:1.5b"]
}
```

### Generate text

```bash
POST /api/v1/voiceover/summarize
Content-Type: application/json
```

**Simple string prompt:**

```json
{
  "prompt": "Write a 15-second voice-over for a product demo."
}
```

**Structured prompt (multi-frame summaries):**

```json
{
  "prompt": {
    "task": "Summarize descriptions into a single paragraph",
    "data": {
      "objects": ["car"],
      "frames": [
        { "description": "A black sports car on a rooftop at dusk." },
        { "description": "Two luxury cars with a city skyline behind them." }
      ]
    }
  }
}
```

Response:

```json
{
  "model": "deepseek-r1:1.5b",
  "response": "..."
}
```

### Chat

```bash
POST /api/v1/voiceover/chat
Content-Type: application/json
```

```json
{
  "messages": [
    { "role": "user", "content": "Write a short podcast intro." }
  ]
}
```

---

## Project structure

```
voice-over-gen/
├── config/                 # Django project settings
├── voiceover/
│   ├── services/
│   │   ├── deepseek.py     # Ollama client
│   │   └── prompts.py      # Prompt builder
│   ├── views.py            # API endpoints
│   └── urls.py
├── deploy/
│   ├── voiceover-gen.service
│   └── nginx-voiceover.conf
├── setup.sh                # Automated Linux/macOS setup
├── setup.ps1               # Automated Windows setup
├── requirements.txt
└── .env.example
```

---

## Troubleshooting

### Ollama not reachable

```bash
sudo systemctl status ollama      # Linux
brew services list                # macOS
curl http://127.0.0.1:11434/api/tags
```

### Model too slow or out of memory

Switch to a smaller model in `.env`:

```env
DEEPSEEK_MODEL=deepseek-r1:1.5b
```

Then:

```bash
ollama pull deepseek-r1:1.5b
sudo systemctl restart voiceover-gen
```

### Django `DisallowedHost` error

Add your server IP or domain to `.env`:

```env
ALLOWED_HOSTS=localhost,127.0.0.1,your-server-ip,your-domain.com
```

Restart the service after changes.

### API returns 503

- Confirm Ollama is running
- Confirm the model is pulled: `ollama list`
- Test directly: `python manage.py test_deepseek "hello"`

---

## Useful commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run development server
python manage.py runserver 0.0.0.0:8000

# Test DeepSeek from CLI
python manage.py test_deepseek "Your prompt here"

# Restart production service (Linux)
sudo systemctl restart voiceover-gen

# Pull a different model
ollama pull deepseek-r1:7b
```

---

## License

Internal POC project.
