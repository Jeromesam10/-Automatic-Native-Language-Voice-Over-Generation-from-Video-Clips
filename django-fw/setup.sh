#!/usr/bin/env bash
set -euo pipefail

# Voice Over Gen - one-shot server setup
# Installs Ollama + DeepSeek, Python deps, Django app, and optional systemd/nginx.

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${APP_DIR}/.venv"
ENV_FILE="${APP_DIR}/.env"
SERVICE_NAME="voiceover-gen"
NGINX_SITE_NAME="voiceover-gen"
INSTALL_NGINX="${INSTALL_NGINX:-false}"
SERVER_NAME="${SERVER_NAME:-_}"
DJANGO_PORT="${DJANGO_PORT:-8000}"
GUNICORN_WORKERS="${GUNICORN_WORKERS:-2}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  printf '\n==> %s\n' "$1"
}

warn() {
  printf 'WARNING: %s\n' "$1" >&2
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_os() {
  case "$(uname -s)" in
    Linux) echo "linux" ;;
    Darwin) echo "macos" ;;
    *)
      echo "Unsupported OS: $(uname -s)" >&2
      exit 1
      ;;
  esac
}

total_ram_gb() {
  local ram_kb=""
  if [[ "$(uname -s)" == "Linux" ]] && command -v free >/dev/null 2>&1; then
    ram_kb="$(free -k | awk '/^Mem:/{print $2}')"
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    ram_kb="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    ram_kb=$((ram_kb / 1024))
  else
    ram_kb=0
  fi

  if [[ "$ram_kb" -lt 1 ]]; then
    echo 8
  else
    echo $((ram_kb / 1024 / 1024 + 1))
  fi
}

choose_model() {
  if [[ -n "$DEEPSEEK_MODEL" ]]; then
    return
  fi

  local ram_gb
  ram_gb="$(total_ram_gb)"
  if [[ "$ram_gb" -lt 12 ]]; then
    DEEPSEEK_MODEL="deepseek-r1:1.5b"
  elif [[ "$ram_gb" -lt 24 ]]; then
    DEEPSEEK_MODEL="deepseek-r1:7b"
  else
    DEEPSEEK_MODEL="deepseek-r1:8b"
  fi

  log "Detected ~${ram_gb}GB RAM, using model: ${DEEPSEEK_MODEL}"
}

install_system_packages_linux() {
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found. Install python3, venv, curl, and git manually if needed."
    return
  fi

  log "Installing Linux system packages"
  sudo apt-get update
  sudo apt-get install -y curl ca-certificates python3 python3-venv python3-pip git
}

install_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    log "Ollama already installed"
    return
  fi

  log "Installing Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
}

start_ollama() {
  log "Starting Ollama service"
  if [[ "$(detect_os)" == "linux" ]]; then
    sudo systemctl enable ollama
    sudo systemctl start ollama
  elif command -v brew >/dev/null 2>&1; then
    brew services start ollama || true
  else
    warn "Start Ollama manually: ollama serve"
  fi

  for _ in {1..30}; do
    if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done

  echo "Ollama did not become ready on http://127.0.0.1:11434" >&2
  exit 1
}

pull_deepseek_model() {
  choose_model
  log "Pulling DeepSeek model: ${DEEPSEEK_MODEL}"
  ollama pull "${DEEPSEEK_MODEL}"
}

setup_python_env() {
  require_command "$PYTHON_BIN"

  log "Creating Python virtual environment"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  log "Installing Python dependencies"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  "${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"
}

generate_secret_key() {
  "${VENV_DIR}/bin/python" - <<'PY'
import secrets
print(secrets.token_urlsafe(50))
PY
}

create_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    log ".env already exists, keeping current file"
    # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a
    DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-r1:1.5b}"
    DJANGO_PORT="${DJANGO_PORT:-8000}"
    GUNICORN_WORKERS="${GUNICORN_WORKERS:-2}"
    return
  fi

  log "Creating .env"
  local secret_key
  secret_key="$(generate_secret_key)"

  cat >"$ENV_FILE" <<EOF
SECRET_KEY=${secret_key}
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1

DJANGO_PORT=${DJANGO_PORT}
GUNICORN_WORKERS=${GUNICORN_WORKERS}

OLLAMA_BASE_URL=http://127.0.0.1:11434
DEEPSEEK_MODEL=${DEEPSEEK_MODEL}
OLLAMA_TIMEOUT=120
EOF
}

run_django_setup() {
  log "Running Django migrations"
  (
    cd "$APP_DIR"
  # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a
    "${VENV_DIR}/bin/python" manage.py migrate --noinput
  )
}

install_systemd_service() {
  if [[ "$(detect_os)" != "linux" ]]; then
    warn "Skipping systemd setup on non-Linux host"
    return
  fi

  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found, skipping service install"
    return
  fi

  log "Installing systemd service: ${SERVICE_NAME}"
  local service_path="/etc/systemd/system/${SERVICE_NAME}.service"
  local run_user
  run_user="$(id -un)"

  sed \
    -e "s|__USER__|${run_user}|g" \
    -e "s|__GROUP__|${run_user}|g" \
    -e "s|__APP_DIR__|${APP_DIR}|g" \
    -e "s|__PORT__|${DJANGO_PORT}|g" \
    -e "s|__WORKERS__|${GUNICORN_WORKERS}|g" \
    "${APP_DIR}/deploy/voiceover-gen.service" | sudo tee "$service_path" >/dev/null

  sudo systemctl daemon-reload
  sudo systemctl enable "${SERVICE_NAME}"
  sudo systemctl restart "${SERVICE_NAME}"
}

install_nginx() {
  if [[ "$INSTALL_NGINX" != "true" ]]; then
    return
  fi

  if [[ "$(detect_os)" != "linux" ]]; then
    warn "Nginx install skipped on non-Linux host"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found, skipping nginx install"
    return
  fi

  log "Installing and configuring Nginx"
  sudo apt-get install -y nginx

  local nginx_path="/etc/nginx/sites-available/${NGINX_SITE_NAME}"
  sed \
    -e "s|__SERVER_NAME__|${SERVER_NAME}|g" \
    -e "s|__PORT__|${DJANGO_PORT}|g" \
    "${APP_DIR}/deploy/nginx-voiceover.conf" | sudo tee "$nginx_path" >/dev/null

  sudo ln -sf "$nginx_path" "/etc/nginx/sites-enabled/${NGINX_SITE_NAME}"
  sudo nginx -t
  sudo systemctl enable nginx
  sudo systemctl restart nginx
}

verify_setup() {
  log "Verifying setup"
  (
    cd "$APP_DIR"
  # shellcheck disable=SC1090
    set -a
    source "$ENV_FILE"
    set +a
    "${VENV_DIR}/bin/python" manage.py check
    "${VENV_DIR}/bin/python" manage.py test_deepseek "Say hello in one short sentence."
  )

  if curl -fsS "http://127.0.0.1:${DJANGO_PORT}/api/v1/voiceover/health" >/dev/null 2>&1; then
    log "API health check passed"
  else
    warn "API is not reachable yet on port ${DJANGO_PORT}"
    if [[ "$(detect_os)" == "linux" ]]; then
      warn "Check service logs: sudo journalctl -u ${SERVICE_NAME} -f"
    else
      warn "Start manually: ${VENV_DIR}/bin/python manage.py runserver 0.0.0.0:${DJANGO_PORT}"
    fi
  fi
}

print_summary() {
  cat <<EOF

Setup complete.

Project directory: ${APP_DIR}
Environment file:  ${ENV_FILE}
DeepSeek model:    ${DEEPSEEK_MODEL}
API health:        http://127.0.0.1:${DJANGO_PORT}/api/v1/voiceover/health

Useful commands:
  source .env && .venv/bin/python manage.py test_deepseek "Your prompt"
  curl http://127.0.0.1:${DJANGO_PORT}/api/v1/voiceover/health

If you exposed this server publicly, update ALLOWED_HOSTS in .env and restart the service.

EOF
}

main() {
  log "Voice Over Gen setup starting"
  local os
  os="$(detect_os)"

  if [[ "$os" == "linux" ]]; then
    install_system_packages_linux
  fi

  install_ollama
  start_ollama
  pull_deepseek_model
  setup_python_env
  create_env_file
  run_django_setup

  if [[ "$os" == "linux" ]]; then
    install_systemd_service
    install_nginx
  fi

  verify_setup
  print_summary
}

main "$@"
