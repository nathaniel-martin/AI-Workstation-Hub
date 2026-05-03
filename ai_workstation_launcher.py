#!/usr/bin/env python3
"""
AI Workstation Hub - Portable Ollama Edition
===========================================

Windows-native, no-admin, no-installer launcher for a local LLM workstation UI.

Default behavior:
  python ai_workstation_launcher.py

What it does:
  - Creates .ai_workstation_venv if needed.
  - Installs only lightweight UI/client dependencies.
  - Creates ai_workstation_settings.json.
  - Downloads portable Ollama into runtime/ollama when missing.
  - Starts managed Ollama locally.
  - Checks for portable Ollama updates daily.
  - Keeps the original flat Gradio layout.

Notes:
  - This intentionally avoids vLLM, WSL2, Docker, AutoAWQ, and Python GGUF bindings.
  - Supports HF GGUF download + Ollama import to avoid blocked Ollama registry pulls.
  - Adds a workspace-bound autonomous coding agent using the local Ollama model.
  - AWQ Safetensors can be downloaded separately, but portable Ollama import is GGUF-focused.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import venv
import warnings
import zipfile
from collections.abc import Generator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_NAME = "AI Workstation Hub"
ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = ROOT / "ai_workstation_settings.json"
DEFAULT_VENV_DIR = ROOT / ".ai_workstation_venv"
BOOTSTRAP_FLAG = "AI_WORKSTATION_BOOTSTRAPPED"
DEPENDENCY_MARKER = "ai_workstation_deps.json"

OLLAMA_RELEASES_API = "https://api.github.com/repos/ollama/ollama/releases/latest"
OLLAMA_RELEASES_PAGE = "https://github.com/ollama/ollama/releases"
DEFAULT_OLLAMA_HOST = "127.0.0.1"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_MODEL = "qwen3-14b-hf"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "ai_workstation.log"

BASE_REQUIREMENTS = [
    "gradio>=3.50,<6",
    "requests>=2.31",
    "nvidia-ml-py>=12.535",
]


def dependency_marker_path(venv_dir: Path) -> Path:
    return venv_dir / DEPENDENCY_MARKER


def dependency_signature() -> dict[str, Any]:
    return {
        "requirements": BASE_REQUIREMENTS,
        "bootstrap_version": 4,
    }


def dependency_marker_valid(venv_dir: Path) -> bool:
    marker = dependency_marker_path(venv_dir)
    if not marker.exists():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return data.get("signature") == dependency_signature()
    except Exception:
        return False


def write_dependency_marker(venv_dir: Path) -> None:
    marker = dependency_marker_path(venv_dir)
    marker.write_text(
        json.dumps({"signature": dependency_signature(), "written_utc": utc_iso()}, indent=2),
        encoding="utf-8",
    )


def venv_has_base_requirements(py: Path) -> bool:
    if not py.exists():
        return False
    code = "import gradio, requests; import pynvml; print('ok')"
    try:
        result = subprocess.run(
            [str(py), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        return result.returncode == 0
    except Exception:
        return False


def log_event(message: str) -> None:
    """Append a timestamped message to the local launcher log and echo it to console."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
    except Exception:
        print(message, flush=True)


def tail_log(max_chars: int = 12000) -> str:
    try:
        if not LOG_FILE.exists():
            return f"No log file yet. Expected: {LOG_FILE}"
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
    except Exception as exc:
        return f"Could not read log file: {type(exc).__name__}: {exc}"


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_OLLAMA_HOST
    ollama_port: int = DEFAULT_OLLAMA_PORT
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860
    runtime_dir: str = "runtime/ollama"
    models_dir: str = "runtime/ollama-models"
    auto_download_ollama: bool = True
    auto_update_ollama: bool = True
    update_check_interval_hours: int = 24
    auto_pull_model: bool = False
    last_update_check_utc: str = ""
    installed_ollama_version: str = ""
    last_known_ollama_version: str = ""
    open_browser: bool = True
    request_timeout_seconds: int = 15
    # Hugging Face GGUF download/import path for corporate networks where Ollama pulls are blocked.
    hf_repo: str = "Qwen/Qwen3-14B-GGUF"
    hf_file: str = ""  # Empty means auto-select a reasonable .gguf file from repo metadata.
    hf_download_dir: str = "models/huggingface"
    ollama_import_model_name: str = "qwen3-14b-hf"
    hf_token: str = ""  # Optional; prefer HF_TOKEN env var for secrets.
    # Local autonomous coding agent settings. The agent is intentionally workspace-bound.
    agent_workspace: str = "workspace"
    agent_max_steps: int = 12
    agent_allow_write: bool = True
    agent_allow_shell: bool = False

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.ollama_port}"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return now_utc().isoformat()


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_settings(path: Path) -> Settings:
    if not path.exists():
        settings = Settings()
        save_settings(path, settings)
        return settings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        base = asdict(Settings())
        base.update({k: v for k, v in data.items() if k in base})
        return Settings(**base)
    except Exception:
        backup = path.with_suffix(path.suffix + f".bad-{int(time.time())}")
        shutil.copy2(path, backup)
        settings = Settings()
        save_settings(path, settings)
        print(f"Settings file was invalid. Backed up to {backup.name} and created defaults.", flush=True)
        return settings


def save_settings(path: Path, settings: Settings) -> None:
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(os.environ.get("VIRTUAL_ENV"))


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if platform.system().lower() == "windows" else "bin/python")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    log_event("+ " + " ".join(str(x) for x in cmd))
    subprocess.check_call(cmd, env=env)


def bootstrap_if_needed(args: argparse.Namespace) -> None:
    """Create/use the managed venv, then launch the real app inside it.

    Important: do not use os.execve on Windows here. In some PowerShell/corporate
    environments it can terminate after setup without visibly continuing into the
    app. A foreground subprocess is more predictable and preserves console logs.
    """
    if args.no_bootstrap or in_venv() or os.environ.get(BOOTSTRAP_FLAG) == "1":
        return

    venv_dir = Path(args.venv_dir)
    py = venv_python(venv_dir)
    if not py.exists():
        print(f"Creating virtual environment: {venv_dir}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(str(venv_dir))

    deps_ready = False
    if not args.force_install and py.exists():
        if dependency_marker_valid(venv_dir) and venv_has_base_requirements(py):
            deps_ready = True
        elif venv_has_base_requirements(py):
            # The venv predates the marker feature but already has what we need.
            write_dependency_marker(venv_dir)
            deps_ready = True

    if not deps_ready:
        # One-time dependency setup. Do not upgrade/reinstall every launch.
        run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

        if args.wheelhouse:
            install_cmd = [
                str(py), "-m", "pip", "install",
                "--no-index", "--find-links", str(args.wheelhouse),
                *BASE_REQUIREMENTS,
            ]
        else:
            install_cmd = [str(py), "-m", "pip", "install", *BASE_REQUIREMENTS]
        run_cmd(install_cmd)
        write_dependency_marker(venv_dir)
    else:
        print("Managed virtual environment already prepared; skipping dependency install.", flush=True)

    env = os.environ.copy()
    env[BOOTSTRAP_FLAG] = "1"
    child_cmd = [str(py), str(Path(__file__).resolve()), *sys.argv[1:]]
    print("Launching AI Workstation inside managed virtual environment...", flush=True)
    run_cmd(child_cmd, env=env)
    raise SystemExit(0)

def import_runtime_modules() -> dict[str, Any]:
    warnings.filterwarnings("ignore", message=".*tuples.*deprecated.*", category=UserWarning)
    mods: dict[str, Any] = {}
    missing = []
    for name in ("gradio", "requests"):
        try:
            mods[name] = __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}. Re-run without --no-bootstrap.")
    try:
        import pynvml  # type: ignore
        mods["pynvml"] = pynvml
    except Exception:
        mods["pynvml"] = None
    return mods


def get_gpu_stats(mods: dict[str, Any]) -> str:
    pynvml = mods.get("pynvml")
    if not pynvml:
        return "GPU Monitor: unavailable"
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return (
            f"**GPU:** {name}  \n"
            f"**VRAM:** {info.used // 1024**2}MB / {info.total // 1024**2}MB  \n"
            f"**Load:** {util.gpu}% | **Temp:** {temp}°C"
        )
    except Exception as exc:
        return f"GPU Monitor: unavailable ({type(exc).__name__})"


def request_json(url: str, timeout: int = 15) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Workstation-Hub"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def latest_ollama_release(timeout: int = 15) -> tuple[str, str]:
    data = request_json(OLLAMA_RELEASES_API, timeout=timeout)
    version = str(data.get("tag_name") or "").lstrip("v")
    assets = data.get("assets") or []
    # Prefer the normal Windows NVIDIA-capable zip. Avoid mlx assets.
    for asset in assets:
        name = str(asset.get("name") or "").lower()
        if name == "ollama-windows-amd64.zip":
            return version, str(asset.get("browser_download_url"))
    for asset in assets:
        name = str(asset.get("name") or "").lower()
        if "ollama-windows-amd64" in name and name.endswith(".zip") and "mlx" not in name:
            return version, str(asset.get("browser_download_url"))
    raise RuntimeError("Could not find ollama-windows-amd64.zip in latest release assets.")


def download_file(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "AI-Workstation-Hub"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as f:
        length = resp.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else None
        read = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                print(f"Downloading Ollama... {pct}%", end="\r", flush=True)
        log_event(f"Downloaded Ollama package to {dest}")
    print("", flush=True)


def ollama_exe(runtime_dir: Path) -> Path:
    return runtime_dir / ("ollama.exe" if platform.system().lower() == "windows" else "ollama")


def install_ollama_portable(settings: Settings, *, force: bool = False) -> str:
    if platform.system().lower() != "windows":
        raise RuntimeError("This portable Ollama workflow currently targets native Windows.")
    runtime_dir = ROOT / settings.runtime_dir
    exe = ollama_exe(runtime_dir)
    if exe.exists() and not force:
        return "Portable Ollama already present."
    version, url = latest_ollama_release(settings.request_timeout_seconds)
    with tempfile.TemporaryDirectory(prefix="aiw_ollama_") as td:
        zip_path = Path(td) / "ollama-windows-amd64.zip"
        print(f"Downloading portable Ollama {version}...", flush=True)
        download_file(url, zip_path, timeout=max(30, settings.request_timeout_seconds))
        tmp_extract = Path(td) / "extract"
        tmp_extract.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_extract)
        if runtime_dir.exists():
            shutil.rmtree(runtime_dir)
        runtime_dir.parent.mkdir(parents=True, exist_ok=True)
        # Some zips contain files at root; some may contain a top-level folder.
        candidate = tmp_extract / "ollama.exe"
        if candidate.exists():
            shutil.copytree(tmp_extract, runtime_dir)
        else:
            nested = next((p for p in tmp_extract.iterdir() if p.is_dir() and (p / "ollama.exe").exists()), None)
            if not nested:
                raise RuntimeError("Downloaded Ollama ZIP did not contain ollama.exe.")
            shutil.copytree(nested, runtime_dir)
    settings.installed_ollama_version = version
    settings.last_known_ollama_version = version
    settings.last_update_check_utc = utc_iso()
    save_settings(DEFAULT_SETTINGS_PATH, settings)
    return f"Installed portable Ollama {version}."


def update_due(settings: Settings) -> bool:
    if not settings.auto_update_ollama:
        return False
    last = parse_dt(settings.last_update_check_utc)
    if not last:
        return True
    return (now_utc() - last).total_seconds() >= settings.update_check_interval_hours * 3600


def check_update(settings: Settings) -> str:
    version, _url = latest_ollama_release(settings.request_timeout_seconds)
    settings.last_known_ollama_version = version
    settings.last_update_check_utc = utc_iso()
    save_settings(DEFAULT_SETTINGS_PATH, settings)
    if settings.installed_ollama_version and version != settings.installed_ollama_version:
        return version
    return ""


OLLAMA_PROCESS: subprocess.Popen | None = None


def ollama_api_url(settings: Settings, path: str) -> str:
    return f"{settings.api_base}{path}"


def is_ollama_ready(settings: Settings, requests_mod: Any) -> bool:
    try:
        r = requests_mod.get(ollama_api_url(settings, "/api/tags"), timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama(settings: Settings, requests_mod: Any) -> str:
    global OLLAMA_PROCESS
    if is_ollama_ready(settings, requests_mod):
        log_event(f"Connected to existing Ollama runtime at {settings.api_base}")
        return "Connected to existing Ollama runtime."
    exe = ollama_exe(ROOT / settings.runtime_dir)
    if not exe.exists():
        if settings.auto_download_ollama:
            install_ollama_portable(settings)
        else:
            raise RuntimeError(f"Ollama not found at {exe}. Enable auto download or place portable Ollama there.")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{settings.host}:{settings.ollama_port}"
    models_dir = ROOT / settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("OLLAMA_MODELS", str(models_dir))
    flags = 0
    if platform.system().lower() == "windows":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    log_event("Starting portable Ollama runtime...")
    log_event(f"Ollama API: {settings.api_base}")
    log_event(f"Ollama executable: {exe}")
    log_event(f"OLLAMA_MODELS: {env.get('OLLAMA_MODELS')}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ollama_log = LOG_DIR / "ollama-runtime.log"
    log_handle = ollama_log.open("ab")
    OLLAMA_PROCESS = subprocess.Popen(
        [str(exe), "serve"],
        cwd=str(exe.parent),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    log_event(f"Managed Ollama PID: {OLLAMA_PROCESS.pid}; runtime log: {ollama_log}")
    deadline = time.time() + 30
    while time.time() < deadline:
        if is_ollama_ready(settings, requests_mod):
            return "Started portable Ollama runtime."
        if OLLAMA_PROCESS.poll() is not None:
            raise RuntimeError("Ollama process exited during startup.")
        time.sleep(0.5)
    raise RuntimeError("Ollama did not become ready within 30 seconds.")


def stop_ollama() -> None:
    global OLLAMA_PROCESS
    if OLLAMA_PROCESS and OLLAMA_PROCESS.poll() is None:
        with contextlib.suppress(Exception):
            if platform.system().lower() == "windows":
                OLLAMA_PROCESS.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                OLLAMA_PROCESS.terminate()
        try:
            OLLAMA_PROCESS.wait(timeout=8)
        except Exception:
            with contextlib.suppress(Exception):
                OLLAMA_PROCESS.kill()
    log_event("Managed Ollama stopped.")
    OLLAMA_PROCESS = None


def list_models(settings: Settings, requests_mod: Any) -> list[str]:
    try:
        r = requests_mod.get(ollama_api_url(settings, "/api/tags"), timeout=settings.request_timeout_seconds)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def model_present(model: str, settings: Settings, requests_mod: Any) -> bool:
    models = list_models(settings, requests_mod)
    if model in models:
        return True
    # Ollama model tags can normalize latest; compare repo prefix too.
    base = model.split(":", 1)[0]
    return any(x == base or x.startswith(base + ":") for x in models)


def pull_model_stream(model: str, settings: Settings, requests_mod: Any) -> Generator[str, None, None]:
    payload = {"name": model, "stream": True}
    log_event(f"Starting model pull: {model}")
    try:
        with requests_mod.post(ollama_api_url(settings, "/api/pull"), json=payload, stream=True, timeout=None) as r:
            log_event(f"Ollama pull HTTP status: {r.status_code}")
            r.raise_for_status()
            last = "pulling"
            last_emit = 0.0
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    log_event(f"Unparseable pull line: {line[:250]}")
                    continue
                if obj.get("error"):
                    msg = f"Model pull failed: {obj.get('error')}"
                    log_event(msg)
                    yield msg
                    return
                status = obj.get("status") or "pulling"
                digest = obj.get("digest") or ""
                total = obj.get("total") or 0
                completed = obj.get("completed") or 0
                if total and completed:
                    pct = completed * 100 / total
                    mb_done = completed / 1024 / 1024
                    mb_total = total / 1024 / 1024
                    last = f"{status}: {pct:.1f}% ({mb_done:.0f} / {mb_total:.0f} MB)"
                else:
                    last = str(status)
                if digest:
                    last = f"{last} [{digest[:18]}…]"
                now = time.time()
                if now - last_emit > 0.5 or "success" in status.lower() or "complete" in status.lower():
                    log_event(f"Pull {model}: {last}")
                    last_emit = now
                yield last
            log_event(f"Model pull finished: {model}; last status: {last}")
            yield last or "Model pull complete."
    except Exception as exc:
        msg = f"Model pull failed: {type(exc).__name__}: {exc}"
        log_event(msg)
        yield msg


def normalize_history(history: Any) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not history:
        return messages
    # Newer Gradio messages style.
    if isinstance(history, list) and history and isinstance(history[0], dict):
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages
    # Older Gradio tuple style: [(user, assistant), ...]
    if isinstance(history, list):
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_msg, assistant_msg = item[0], item[1]
                if user_msg:
                    messages.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": str(assistant_msg)})
    return messages


def chat_stream(message: str, history: Any, model: str, max_tokens: int, temperature: float, top_p: float, settings: Settings, requests_mod: Any) -> Generator[str, None, None]:
    if not message:
        yield ""
        return
    if not is_ollama_ready(settings, requests_mod):
        yield "Ollama is not running. Click Start / Connect first."
        return
    messages = normalize_history(history)
    messages.append({"role": "user", "content": message})
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "num_predict": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        },
    }
    try:
        with requests_mod.post(ollama_api_url(settings, "/api/chat"), json=payload, stream=True, timeout=None) as r:
            if r.status_code == 404:
                yield f"Model `{model}` is not available locally. If you imported from Hugging Face, use the imported model name such as `{settings.ollama_import_model_name}`. Otherwise use Pull Model or run: ollama pull {model}"
                return
            r.raise_for_status()
            response = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("error"):
                    yield f"Ollama error: {obj['error']}"
                    return
                chunk = (obj.get("message") or {}).get("content") or ""
                if chunk:
                    response += chunk
                    yield response
                if obj.get("done"):
                    break
    except Exception as exc:
        yield f"Chat error: {type(exc).__name__}: {exc}"


def start_connect_ui(settings: Settings, requests_mod: Any, model: str, auto_pull: bool) -> Generator[tuple[str, str], None, None]:
    try:
        yield "Status: preparing Ollama...", ""
        start_msg = start_ollama(settings, requests_mod)
        status_lines = [start_msg]
        if update_due(settings):
            try:
                new_version = check_update(settings)
                if new_version and settings.auto_update_ollama:
                    status_lines.append(f"Update available: Ollama {new_version}; updating runtime...")
                    yield "Status: " + "\n".join(status_lines), ""
                    stop_ollama()
                    install_ollama_portable(settings, force=True)
                    start_msg = start_ollama(settings, requests_mod)
                    status_lines.append(start_msg)
                elif new_version:
                    status_lines.append(f"Update available: Ollama {new_version}")
                else:
                    status_lines.append("Ollama update check: current")
            except Exception as exc:
                status_lines.append(f"Update check skipped: {type(exc).__name__}: {exc}")
        if auto_pull and not model_present(model, settings, requests_mod):
            for progress in pull_model_stream(model, settings, requests_mod):
                yield "Status: " + "\n".join(status_lines + [f"Pulling {model}: {progress}"]), ""
        present = model_present(model, settings, requests_mod)
        if present:
            status_lines.append(f"Model ready: {model}")
        else:
            status_lines.append(f"Model not found locally: {model}. If you imported from Hugging Face, switch the Ollama Model field to `{settings.ollama_import_model_name}`. Otherwise click Pull Model or run: ollama pull {model}")
        settings.model = model
        settings.auto_pull_model = auto_pull
        save_settings(DEFAULT_SETTINGS_PATH, settings)
        yield "Status: " + "\n".join(status_lines), ""
    except Exception as exc:
        yield f"Status: Error — {type(exc).__name__}: {exc}", ""


def pull_model_ui(settings: Settings, requests_mod: Any, model: str) -> Generator[str, None, None]:
    try:
        start_ollama(settings, requests_mod)
        model = (model or settings.model).strip()
        if model_present(model, settings, requests_mod):
            settings.model = model
            save_settings(DEFAULT_SETTINGS_PATH, settings)
            yield f"Status: Model already exists locally: {model}. No pull needed."
            return
        for progress in pull_model_stream(model, settings, requests_mod):
            yield f"Status: Pulling {model}: {progress}"
        settings.model = model
        save_settings(DEFAULT_SETTINGS_PATH, settings)
    except Exception as exc:
        yield f"Status: Pull failed — {type(exc).__name__}: {exc}"


def save_settings_ui(settings: Settings, model: str, auto_download: bool, auto_update: bool, auto_pull: bool, ui_host: str, ui_port: int, agent_workspace: str, agent_max_steps: int, agent_allow_write: bool, agent_allow_shell: bool) -> str:
    settings.model = model
    settings.auto_download_ollama = bool(auto_download)
    settings.auto_update_ollama = bool(auto_update)
    settings.auto_pull_model = bool(auto_pull)
    settings.ui_host = ui_host or "127.0.0.1"
    settings.ui_port = int(ui_port)
    settings.agent_workspace = agent_workspace or "workspace"
    settings.agent_max_steps = int(agent_max_steps)
    settings.agent_allow_write = bool(agent_allow_write)
    settings.agent_allow_shell = bool(agent_allow_shell)
    save_settings(DEFAULT_SETTINGS_PATH, settings)
    return "Status: Settings saved."






def hf_headers(settings: Settings) -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or settings.hf_token or ""
    headers = {"User-Agent": "AI-Workstation-Hub"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_repo_api_url(repo: str) -> str:
    return f"https://huggingface.co/api/models/{repo}"


def hf_file_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def list_hf_gguf_files(repo: str, settings: Settings, requests_mod: Any) -> list[dict[str, Any]]:
    """Return GGUF files from a Hugging Face repo using the Hub metadata API."""
    log_event(f"Querying Hugging Face repo metadata: {repo}")
    r = requests_mod.get(hf_repo_api_url(repo), headers=hf_headers(settings), timeout=max(15, settings.request_timeout_seconds))
    r.raise_for_status()
    data = r.json()
    siblings = data.get("siblings") or []
    files = []
    for item in siblings:
        name = item.get("rfilename") or item.get("path") or ""
        if str(name).lower().endswith(".gguf"):
            files.append({
                "name": name,
                "size": item.get("size") or item.get("lfs", {}).get("size") or 0,
            })
    files.sort(key=lambda x: (str(x["name"]).lower()))
    log_event(f"Found {len(files)} GGUF file(s) in {repo}")
    return files


def choose_hf_gguf_file(files: list[dict[str, Any]], requested: str = "") -> str:
    if requested:
        return requested
    if not files:
        raise RuntimeError("No .gguf files found in Hugging Face repo metadata.")
    names = [str(f["name"]) for f in files]
    # Prefer a balanced quant. Fall back to any Q4-ish file, then the first GGUF.
    priorities = ["q4_k_m", "q4-k-m", "q4_0", "q4", "iq4", "q5_k_m", "q5"]
    lowered = [(n, n.lower()) for n in names]
    for token in priorities:
        for original, low in lowered:
            if token in low:
                return original
    return names[0]


def safe_repo_path(repo: str) -> Path:
    return Path(*[part for part in repo.replace("\\", "/").split("/") if part])


def download_hf_file_stream(repo: str, filename: str, settings: Settings, requests_mod: Any) -> Generator[str, None, Path]:
    """Download a HF file with resume support and yield status lines."""
    dest_dir = ROOT / settings.hf_download_dir / safe_repo_path(repo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = hf_file_url(repo, filename)
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = hf_headers(settings)
    if existing:
        headers["Range"] = f"bytes={existing}-"
    log_event(f"Downloading Hugging Face file: repo={repo}, file={filename}, dest={dest}")
    with requests_mod.get(url, headers=headers, stream=True, timeout=None, allow_redirects=True) as r:
        if r.status_code == 416 and tmp.exists():
            tmp.rename(dest)
            yield f"Already downloaded: {dest.name}"
            return dest
        r.raise_for_status()
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        total_header = r.headers.get("Content-Length")
        total = int(total_header) + existing if total_header and total_header.isdigit() else 0
        downloaded = existing
        last_emit = 0.0
        with tmp.open(mode + "") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_emit >= 1.0:
                    if total:
                        pct = downloaded * 100 / total
                        msg = f"HF download: {pct:.1f}% ({downloaded/1024/1024:.0f} / {total/1024/1024:.0f} MB)"
                    else:
                        msg = f"HF download: {downloaded/1024/1024:.0f} MB"
                    log_event(msg)
                    yield msg
                    last_emit = now
    tmp.rename(dest)
    log_event(f"HF download complete: {dest}")
    yield f"HF download complete: {dest.name}"
    return dest


def run_ollama_create_stream(model_name: str, gguf_path: Path, settings: Settings) -> Generator[str, None, None]:
    exe = ollama_exe(ROOT / settings.runtime_dir)
    if not exe.exists():
        raise RuntimeError(f"Ollama executable not found: {exe}")
    import_dir = ROOT / "models" / "ollama-imports" / model_name.replace(":", "_").replace("/", "_")
    import_dir.mkdir(parents=True, exist_ok=True)
    modelfile = import_dir / "Modelfile"
    # Use an absolute path and forward slashes; Ollama's Modelfile parser handles Windows absolute paths better this way.
    gguf_ref = str(gguf_path.resolve()).replace("\\", "/")
    modelfile.write_text(f"FROM {gguf_ref}\n", encoding="utf-8")
    log_event(f"Creating Ollama model '{model_name}' from {gguf_path}")
    log_event(f"Modelfile: {modelfile}")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{settings.host}:{settings.ollama_port}"
    models_dir = ROOT / settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("OLLAMA_MODELS", str(models_dir))
    proc = subprocess.Popen(
        [str(exe), "create", model_name, "-f", str(modelfile)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line:
            log_event(f"ollama create: {line}")
            yield line
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ollama create failed with exit code {rc}")
    log_event(f"Ollama model created: {model_name}")
    yield f"Imported as Ollama model: {model_name}"


def hf_download_import_ui(settings: Settings, requests_mod: Any, repo: str, hf_file: str, import_model_name: str) -> Generator[str, None, None]:
    try:
        start_ollama(settings, requests_mod)
        repo = (repo or settings.hf_repo).strip()
        import_model_name = (import_model_name or settings.ollama_import_model_name or "qwen3-14b-hf").strip()
        files = list_hf_gguf_files(repo, settings, requests_mod)
        chosen = choose_hf_gguf_file(files, (hf_file or "").strip())
        settings.hf_repo = repo
        settings.hf_file = chosen
        settings.ollama_import_model_name = import_model_name
        save_settings(DEFAULT_SETTINGS_PATH, settings)
        yield f"Status: Hugging Face file selected: {chosen}"
        gen = download_hf_file_stream(repo, chosen, settings, requests_mod)
        gguf_path = None
        while True:
            try:
                progress = next(gen)
                yield f"Status: {progress}"
            except StopIteration as stop:
                gguf_path = stop.value
                break
        if not gguf_path:
            raise RuntimeError("HF download did not return a local file path.")
        for line in run_ollama_create_stream(import_model_name, Path(gguf_path), settings):
            yield f"Status: {line}"
        settings.model = import_model_name
        save_settings(DEFAULT_SETTINGS_PATH, settings)
        yield f"Status: Ready. Use model `{import_model_name}` in the chat box."
    except Exception as exc:
        msg = f"HF download/import failed: {type(exc).__name__}: {exc}"
        log_event(msg)
        yield f"Status: {msg}"


# -----------------------------------------------------------------------------
# Workspace-bound coding agent
# -----------------------------------------------------------------------------

def workspace_root(settings: Settings, workspace_value: str = "") -> Path:
    raw = (workspace_value or settings.agent_workspace or "workspace").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_workspace_path(root: Path, relative_path: str) -> Path:
    rel = (relative_path or ".").replace("\\", "/").strip().lstrip("/")
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes workspace: {relative_path}")
    return candidate


def list_workspace_files(root: Path, max_files: int = 250) -> str:
    ignored_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".ai_workstation_venv"}
    rows: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        cur = Path(current)
        for name in sorted(files):
            p = cur / name
            try:
                rel = p.relative_to(root).as_posix()
                size = p.stat().st_size
            except Exception:
                continue
            rows.append(f"{rel} ({size} bytes)")
            if len(rows) >= max_files:
                rows.append(f"... truncated at {max_files} files")
                return "\n".join(rows)
    return "\n".join(rows) if rows else "Workspace is empty."


def read_workspace_file(root: Path, relative_path: str, max_chars: int = 24000) -> str:
    p = safe_workspace_path(root, relative_path)
    if not p.exists():
        return f"ERROR: file does not exist: {relative_path}"
    if not p.is_file():
        return f"ERROR: not a file: {relative_path}"
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > max_chars:
        return data[:max_chars] + f"\n\n... truncated after {max_chars} characters ..."
    return data


def write_workspace_file(root: Path, relative_path: str, content: str, allow_write: bool) -> str:
    if not allow_write:
        return "ERROR: file writes are disabled. Enable 'Agent can write files'."
    p = safe_workspace_path(root, relative_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content or "", encoding="utf-8", newline="")
    return f"WROTE: {p.relative_to(root).as_posix()} ({len(content or '')} characters)"


def run_workspace_command(root: Path, command: str, allow_shell: bool, timeout: int = 90) -> str:
    if not allow_shell:
        return "ERROR: shell commands are disabled. Enable 'Agent can run shell commands'."
    command = (command or "").strip()
    if not command:
        return "ERROR: command was empty."
    try:
        result = subprocess.run(
            command,
            cwd=str(root),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = result.stdout or ""
        if len(output) > 16000:
            output = output[-16000:]
        return f"EXIT_CODE: {result.returncode}\n{output}"
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return f"ERROR: command timed out after {timeout}s\n{out[-8000:]}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


VALID_AGENT_ACTIONS = {"list_files", "read_file", "write_file", "create_file", "run_command", "finish"}


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a single JSON object from an LLM response.

    Qwen-style models sometimes include reasoning or prose despite JSON mode. This
    parser accepts a clean object or the outermost object in the response, then
    validates that the result looks like an agent action instead of an unrelated
    manifest/config document.
    """
    text = (text or "").strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                action = str(obj.get("action") or "").strip().lower()
                if action in VALID_AGENT_ACTIONS:
                    obj["action"] = action
                    return obj
                if "action" not in obj:
                    raise ValueError("JSON object is missing required `action` key")
                raise ValueError(f"Unsupported action `{action}`")
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Model did not return a valid agent action JSON object: {last_error}. Raw response: {text[:1000]}")


def call_agent_model_json(settings: Settings, requests_mod: Any, model: str, messages: list[dict[str, str]], temperature: float = 0.15) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": float(temperature), "top_p": 0.9, "num_predict": 2048},
    }
    r = requests_mod.post(ollama_api_url(settings, "/api/chat"), json=payload, timeout=None)
    if r.status_code == 404:
        raise RuntimeError(f"Model `{model}` is not available locally. Use your imported model name, for example `{settings.ollama_import_model_name}`.")
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content") or ""
    return extract_json_object(content)


def ask_agent_for_action_with_retries(settings: Settings, requests_mod: Any, model: str, messages: list[dict[str, str]], retries: int = 3) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            return call_agent_model_json(settings, requests_mod, model, messages)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was rejected. Return exactly one valid JSON object with an `action` key. "
                    "Allowed actions: list_files, read_file, write_file, create_file, run_command, finish. "
                    "No markdown, no manifest, no dependency list, no explanation outside JSON. "
                    f"Parser error: {last_error}"
                ),
            })
            log_event(f"Agent JSON retry {attempt}/{retries}: {last_error}")
    raise ValueError(f"Model did not return valid agent action JSON after {retries} attempts. Last error: {last_error}")


def execute_agent_action(action: dict[str, Any], root: Path, allow_write: bool, allow_shell: bool) -> tuple[str, bool]:
    name = str(action.get("action") or "").strip().lower()
    if not name:
        return "ERROR: missing action", False
    if name == "finish":
        return str(action.get("summary") or "Finished."), True
    if name == "list_files":
        return list_workspace_files(root), False
    if name == "read_file":
        return read_workspace_file(root, str(action.get("path") or "")), False
    if name in {"write_file", "create_file"}:
        return write_workspace_file(root, str(action.get("path") or ""), str(action.get("content") or ""), allow_write), False
    if name == "run_command":
        return run_workspace_command(root, str(action.get("command") or ""), allow_shell), False
    return f"ERROR: unsupported action `{name}`. Supported actions: list_files, read_file, write_file, run_command, finish.", False


def run_coding_agent_ui(
    settings: Settings,
    requests_mod: Any,
    task: str,
    model: str,
    workspace: str,
    max_steps: int,
    allow_write: bool,
    allow_shell: bool,
) -> Generator[tuple[str, str], None, None]:
    """Autonomous but workspace-bound coding agent loop."""
    task = (task or "").strip()
    model = (model or settings.model or settings.ollama_import_model_name).strip()
    if not task:
        yield "Status: Agent task is empty.", tail_log()
        return
    try:
        start_ollama(settings, requests_mod)
        root = workspace_root(settings, workspace)
        settings.agent_workspace = str(Path(workspace or settings.agent_workspace or "workspace"))
        settings.agent_max_steps = int(max_steps)
        settings.agent_allow_write = bool(allow_write)
        settings.agent_allow_shell = bool(allow_shell)
        settings.model = model
        save_settings(DEFAULT_SETTINGS_PATH, settings)
        if not model_present(model, settings, requests_mod):
            yield f"Status: Agent cannot start. Model `{model}` is not local. Use `{settings.ollama_import_model_name}` if you imported from Hugging Face.", tail_log()
            return

        system = (
            "You are a local autonomous coding agent running inside a restricted workspace. "
            "You must solve the user's coding task by repeatedly choosing exactly one tool action. "
            "/no_think\nReturn ONLY valid JSON. No markdown. No prose outside JSON. "
            "Supported JSON actions:\n"
            "{\"thought\":\"brief plan\",\"action\":\"list_files\"}\n"
            "{\"thought\":\"why this file\",\"action\":\"read_file\",\"path\":\"relative/path.py\"}\n"
            "{\"thought\":\"what changed\",\"action\":\"write_file\",\"path\":\"relative/path.py\",\"content\":\"full file content\"}\n"
            "{\"thought\":\"validate\",\"action\":\"run_command\",\"command\":\"command to run\"}\n"
            "{\"thought\":\"done\",\"action\":\"finish\",\"summary\":\"what you changed and how to run it\"}\n"
            "When editing a file, write the complete intended file content. "
            "Stay within the workspace. Prefer reading existing files before overwriting them. "
            "If shell is disabled, do not rely on commands; inspect and edit files directly."
        )
        user = (
            f"Workspace: {root}\n"
            f"File writes enabled: {bool(allow_write)}\n"
            f"Shell commands enabled: {bool(allow_shell)}\n"
            f"Task:\n{task}\n"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        log_event(f"Agent started. model={model}; workspace={root}; max_steps={max_steps}; write={allow_write}; shell={allow_shell}")
        yield f"Status: Agent started in `{root}` with model `{model}`.", tail_log()

        transcript: list[str] = []
        for step in range(1, int(max_steps) + 1):
            log_event(f"Agent step {step}: asking model for next action")
            try:
                action = ask_agent_for_action_with_retries(settings, requests_mod, model, messages)
            except Exception as exc:
                msg = f"Agent failed while asking model: {type(exc).__name__}: {exc}"
                log_event(msg)
                yield f"Status: {msg}", tail_log()
                return
            thought = str(action.get("thought") or "").strip()
            act_name = str(action.get("action") or "").strip()
            path_hint = action.get("path") or action.get("command") or ""
            short = f"Step {step}: {act_name} {path_hint}".strip()
            if thought:
                short += f" — {thought[:180]}"
            log_event("Agent " + short)
            result, done = execute_agent_action(action, root, bool(allow_write), bool(allow_shell))
            log_event(f"Agent tool result step {step}: {result[:1000]}")
            transcript.append(f"{short}\n{result[:2000]}")
            yield "Status: Agent running...\n" + "\n\n".join(transcript[-6:]), tail_log()
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Tool result for step {step}:\n{result}\n\nChoose the next JSON action."})
            if done:
                yield f"Status: Agent finished.\n{result}", tail_log()
                return
        msg = f"Agent stopped after reaching max steps ({max_steps}). Increase max steps and run again if needed."
        log_event(msg)
        yield "Status: " + msg + "\n\n" + "\n\n".join(transcript[-8:]), tail_log()
    except Exception as exc:
        msg = f"Agent error: {type(exc).__name__}: {exc}"
        log_event(msg)
        yield f"Status: {msg}", tail_log()

def diagnostics_text(settings: Settings, requests_mod: Any) -> str:
    lines = [
        f"Log file: {LOG_FILE}",
        f"Ollama API: {settings.api_base}",
        f"Runtime dir: {ROOT / settings.runtime_dir}",
        f"Models dir setting: {ROOT / settings.models_dir}",
        f"HF repo: {settings.hf_repo}",
        f"HF file: {settings.hf_file or '(auto-select)'}",
        f"HF download dir: {ROOT / settings.hf_download_dir}",
        f"Ollama import model name: {settings.ollama_import_model_name}",
        f"Agent workspace: {ROOT / settings.agent_workspace}",
        f"Agent max steps: {settings.agent_max_steps}",
        f"Agent allow write: {settings.agent_allow_write}",
        f"Agent allow shell: {settings.agent_allow_shell}",
    ]
    try:
        r = requests_mod.get(ollama_api_url(settings, "/api/version"), timeout=3)
        lines.append(f"/api/version: HTTP {r.status_code} {r.text[:300]}")
    except Exception as exc:
        lines.append(f"/api/version failed: {type(exc).__name__}: {exc}")
    try:
        r = requests_mod.get(ollama_api_url(settings, "/api/tags"), timeout=5)
        lines.append(f"/api/tags: HTTP {r.status_code} {r.text[:1000]}")
    except Exception as exc:
        lines.append(f"/api/tags failed: {type(exc).__name__}: {exc}")
    return "\n".join(lines)

def build_ui(settings: Settings, mods: dict[str, Any]) -> Any:
    gr = mods["gradio"]
    requests_mod = mods["requests"]

    # Explicit generator functions are required. Do not wrap these in lambdas;
    # older/newer Gradio versions may otherwise receive the generator object
    # instead of streaming its yielded values.
    def chat_ui(message: str, history: Any, model_name: str, max_new: int, temp: float, p: float):
        yield from chat_stream(message, history, model_name, max_new, temp, p, settings, requests_mod)

    def start_ui(model_name: str, ap: bool):
        for st, gpu in start_connect_ui(settings, requests_mod, model_name, ap):
            yield st, gpu, tail_log()

    def pull_ui(model_name: str):
        for st in pull_model_ui(settings, requests_mod, model_name):
            yield st, tail_log()

    def hf_import_ui(repo: str, hf_file_name: str, import_name: str):
        selected_model = (import_name or settings.ollama_import_model_name or "qwen3-14b-hf").strip()
        for st in hf_download_import_ui(settings, requests_mod, repo, hf_file_name, selected_model):
            # Keep the visible Ollama Model textbox synchronized with the imported name.
            # Without this, the UI can keep showing the registry model (for example qwen3:14b),
            # and the next Start/Pull click will try the blocked Ollama registry again.
            yield st, tail_log(), selected_model

    def agent_ui(task: str, model_name: str, workspace_name: str, steps: int, aw: bool, sh: bool):
        for st, logs in run_coding_agent_ui(settings, requests_mod, task, model_name, workspace_name, int(steps), bool(aw), bool(sh)):
            yield st, logs

    def save_ui(m: str, ad: bool, au: bool, ap: bool, uh: str, up: int, workspace_name: str, steps: int, aw: bool, sh: bool):
        st = save_settings_ui(settings, m, ad, au, ap, uh, int(up), workspace_name, int(steps), bool(aw), bool(sh))
        log_event(st)
        return st, tail_log()

    def stop_ui():
        stop_ollama()
        return "Status: Managed Ollama stopped.", tail_log()

    def refresh_logs_ui():
        return tail_log()

    def diag_ui():
        text = diagnostics_text(settings, requests_mod)
        log_event("Diagnostics requested")
        return text + "\n\n--- LOG TAIL ---\n" + tail_log()

    with gr.Blocks(title=APP_NAME) as demo:
        # Flat layout restored: compact controls at left, chat at right.
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Textbox(label="Ollama Model", value=settings.model)
                auto_pull = gr.Checkbox(label="Auto-pull model if missing", value=settings.auto_pull_model)
                auto_download = gr.Checkbox(label="Auto-download portable Ollama", value=settings.auto_download_ollama)
                auto_update = gr.Checkbox(label="Daily auto-update Ollama", value=settings.auto_update_ollama)
                start_btn = gr.Button("🚀 Start / Connect", variant="primary")
                pull_btn = gr.Button("Download / Pull Model")
                hf_repo = gr.Textbox(label="HF GGUF Repo", value=settings.hf_repo)
                hf_file_name = gr.Textbox(label="HF GGUF File (blank = auto-select Q4)", value=settings.hf_file)
                hf_import_name = gr.Textbox(label="Import as Ollama Model", value=settings.ollama_import_model_name)
                hf_import_btn = gr.Button("HF Download + Ollama Import")
                agent_task = gr.Textbox(label="Coding Agent Task", value="", lines=4, placeholder="Example: Review this workspace and create a README with setup steps.")
                agent_workspace = gr.Textbox(label="Agent Workspace", value=settings.agent_workspace)
                agent_steps = gr.Slider(1, 40, value=settings.agent_max_steps, step=1, label="Agent max steps")
                agent_allow_write = gr.Checkbox(label="Agent can write files", value=settings.agent_allow_write)
                agent_allow_shell = gr.Checkbox(label="Agent can run shell commands", value=settings.agent_allow_shell)
                agent_btn = gr.Button("🤖 Run Coding Agent")
                save_btn = gr.Button("Save Settings")
                stop_btn = gr.Button("Stop Managed Ollama")
                status = gr.Markdown("Status: Idle")
                gpu_info = gr.Markdown(get_gpu_stats(mods))
                log_box = gr.Textbox(label="Runtime logs / diagnostics", value=tail_log(), lines=10, interactive=False)
                refresh_logs_btn = gr.Button("Refresh Logs")
                diag_btn = gr.Button("Diagnostics")
                max_tokens = gr.Slider(16, 8192, value=1024, step=16, label="Max new tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                ui_host = gr.Textbox(label="UI host", value=settings.ui_host, visible=False)
                ui_port = gr.Number(label="UI port", value=settings.ui_port, precision=0, visible=False)
            with gr.Column(scale=3):
                # No type="messages" so Gradio 3.x through 5.x can run.
                gr.ChatInterface(
                    fn=chat_ui,
                    additional_inputs=[model, max_tokens, temperature, top_p],
                )

        start_btn.click(start_ui, inputs=[model, auto_pull], outputs=[status, gpu_info, log_box])
        pull_btn.click(pull_ui, inputs=[model], outputs=[status, log_box])
        hf_import_btn.click(hf_import_ui, inputs=[hf_repo, hf_file_name, hf_import_name], outputs=[status, log_box, model])
        agent_btn.click(agent_ui, inputs=[agent_task, model, agent_workspace, agent_steps, agent_allow_write, agent_allow_shell], outputs=[status, log_box])
        save_btn.click(
            save_ui,
            inputs=[model, auto_download, auto_update, auto_pull, ui_host, ui_port, agent_workspace, agent_steps, agent_allow_write, agent_allow_shell],
            outputs=[status, log_box],
        )
        stop_btn.click(stop_ui, outputs=[status, log_box])
        refresh_logs_btn.click(refresh_logs_ui, outputs=[log_box])
        diag_btn.click(diag_ui, outputs=[log_box])
    return demo

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portable Windows Ollama launcher with flat Gradio UI.")
    p.add_argument("--settings", default=str(DEFAULT_SETTINGS_PATH))
    p.add_argument("--venv-dir", default=str(DEFAULT_VENV_DIR))
    p.add_argument("--model", default=None)
    p.add_argument("--ui-host", default=None)
    p.add_argument("--ui-port", type=int, default=None)
    p.add_argument("--ollama-port", type=int, default=None)
    p.add_argument("--runtime-dir", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--hf-repo", default=None)
    p.add_argument("--hf-file", default=None)
    p.add_argument("--hf-download-dir", default=None)
    p.add_argument("--ollama-import-model-name", default=None)
    p.add_argument("--hf-download-import", action="store_true", help="Download GGUF from Hugging Face and import it into Ollama, then exit.")
    p.add_argument("--agent-task", default="", help="Run the workspace-bound coding agent with this task, then exit.")
    p.add_argument("--agent-workspace", default=None)
    p.add_argument("--agent-max-steps", type=int, default=None)
    p.add_argument("--agent-allow-write", action="store_true")
    p.add_argument("--agent-allow-shell", action="store_true")
    p.add_argument("--auto-pull-model", action="store_true")
    p.add_argument("--no-auto-download", action="store_true")
    p.add_argument("--no-auto-update", action="store_true")
    p.add_argument("--download-ollama-only", action="store_true")
    p.add_argument("--update-ollama-now", action="store_true")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--wheelhouse", default="")
    p.add_argument("--force-install", action="store_true", help="Force dependency reinstall even when the managed venv already looks ready.")
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-browser", action="store_true")
    return p.parse_args()


def apply_args(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.model:
        settings.model = args.model
    if args.ui_host:
        settings.ui_host = args.ui_host
    if args.ui_port:
        settings.ui_port = args.ui_port
    if args.ollama_port:
        settings.ollama_port = args.ollama_port
    if args.runtime_dir:
        settings.runtime_dir = args.runtime_dir
    if args.models_dir:
        settings.models_dir = args.models_dir
    if args.hf_repo:
        settings.hf_repo = args.hf_repo
    if args.hf_file:
        settings.hf_file = args.hf_file
    if args.hf_download_dir:
        settings.hf_download_dir = args.hf_download_dir
    if args.ollama_import_model_name:
        settings.ollama_import_model_name = args.ollama_import_model_name
    if args.agent_workspace:
        settings.agent_workspace = args.agent_workspace
    if args.agent_max_steps:
        settings.agent_max_steps = int(args.agent_max_steps)
    if args.agent_allow_write:
        settings.agent_allow_write = True
    if args.agent_allow_shell:
        settings.agent_allow_shell = True
    if args.auto_pull_model:
        settings.auto_pull_model = True
    if args.no_auto_download:
        settings.auto_download_ollama = False
    if args.no_auto_update:
        settings.auto_update_ollama = False
    if args.no_browser:
        settings.open_browser = False
    return settings


def main() -> int:
    args = parse_args()
    bootstrap_if_needed(args)
    settings_path = Path(args.settings).resolve()
    global DEFAULT_SETTINGS_PATH
    DEFAULT_SETTINGS_PATH = settings_path
    settings = apply_args(load_settings(settings_path), args)
    save_settings(settings_path, settings)

    if args.download_ollama_only:
        print(install_ollama_portable(settings, force=False))
        return 0
    if args.update_ollama_now:
        stop_ollama()
        print(install_ollama_portable(settings, force=True))
        return 0

    mods = import_runtime_modules()
    try:
        # Startup should be useful, but UI should still open even if download/update fails.
        if settings.auto_download_ollama and not ollama_exe(ROOT / settings.runtime_dir).exists():
            print(install_ollama_portable(settings, force=False), flush=True)
        requests_mod = mods["requests"]
        start_ollama(settings, requests_mod)
    except Exception as exc:
        log_event(f"Startup warning: {type(exc).__name__}: {exc}")

    if args.hf_download_import:
        requests_mod = mods["requests"]
        for line in hf_download_import_ui(settings, requests_mod, settings.hf_repo, settings.hf_file, settings.ollama_import_model_name):
            print(line, flush=True)
        return 0

    if args.agent_task:
        requests_mod = mods["requests"]
        for st, logs in run_coding_agent_ui(
            settings,
            requests_mod,
            args.agent_task,
            settings.model,
            settings.agent_workspace,
            settings.agent_max_steps,
            settings.agent_allow_write,
            settings.agent_allow_shell,
        ):
            print(st, flush=True)
        return 0

    demo = build_ui(settings, mods)
    log_event(f"Launching {APP_NAME} UI at http://{settings.ui_host}:{settings.ui_port}")
    try:
        demo.launch(
            server_name=settings.ui_host,
            server_port=settings.ui_port,
            inbrowser=settings.open_browser,
            share=bool(args.share),
        )
    finally:
        stop_ollama()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
