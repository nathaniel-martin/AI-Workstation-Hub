# AI Workstation Hub

Portable, Windows-native local AI workstation for running a local Ollama model with a flat Gradio UI, Hugging Face GGUF import, diagnostics, and a workspace-scoped coding agent.

This project was designed for restricted corporate Windows environments where the following may be unavailable or blocked:

- local admin rights
- installers
- WSL2
- Docker
- native Python CUDA/AWQ/GGUF wheel builds
- direct Ollama registry model pulls

The preferred workflow is:

```text
Hugging Face GGUF download → local Ollama import → Gradio UI → workspace-scoped coding agent
```

## Features

- No-admin portable Ollama runtime management.
- Persistent `ai_workstation_settings.json` settings file.
- Flat Gradio layout.
- Hugging Face GGUF download and `ollama create` import.
- Avoids vLLM, Docker, WSL2, AutoAWQ, and Python GGUF bindings.
- Runtime logs and diagnostics panel.
- Workspace-scoped autonomous coding agent.
- Optional shell execution, disabled by default unless explicitly enabled.
- CLI support for app launch, HF import, and agent tasks.
- CI checks for linting, tests, import safety, and Windows executable build.

## Quick start

```powershell
python .\ai_workstation_launcher.py
```

The launcher will create a managed `.ai_workstation_venv`, install lightweight UI/client dependencies, create `ai_workstation_settings.json`, download portable Ollama when missing, start Ollama locally, and launch the UI at `http://127.0.0.1:7860`.

## Recommended model flow

Corporate networks may block Ollama registry blob downloads. Use the Hugging Face import path instead.

In the UI:

1. Set **HF GGUF Repo** to `Qwen/Qwen3-14B-GGUF`.
2. Leave **HF GGUF File** blank to auto-select a balanced Q4 file.
3. Set **Import as Ollama Model** to `qwen3-14b-hf`.
4. Click **HF Download + Ollama Import**.
5. Use `qwen3-14b-hf` in the **Ollama Model** field.

CLI equivalent:

```powershell
python .\ai_workstation_launcher.py `
  --hf-download-import `
  --hf-repo Qwen/Qwen3-14B-GGUF `
  --ollama-import-model-name qwen3-14b-hf
```

## Coding agent

The coding agent is intentionally bounded to a configured workspace. It can list, read, and write files inside that workspace. Shell command execution is optional and should only be enabled for trusted workspaces.

Example CLI agent run:

```powershell
python .\ai_workstation_launcher.py `
  --model qwen3-14b-hf `
  --agent-task "Review this workspace and create a README with setup instructions" `
  --agent-workspace .\workspace `
  --agent-max-steps 12 `
  --agent-allow-write
```

## Logs

```powershell
Get-Content .\logs\ai_workstation.log -Wait
Get-Content .\logs\ollama-runtime.log -Wait
```

## Build an executable

```powershell
.\scripts\build_exe.ps1
```

The build produces a PyInstaller onedir bundle under `dist\AIWorkstationHub`.

## Security model

The agent is not allowed to write outside its workspace path. Shell command execution is disabled unless explicitly enabled. Review `SECURITY.md` before using the agent against sensitive repositories.
