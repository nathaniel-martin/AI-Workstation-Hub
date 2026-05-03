# Operations

## Start the app

```powershell
python .\ai_workstation_launcher.py
```

## Import a model from Hugging Face

```powershell
python .\ai_workstation_launcher.py --hf-download-import --hf-repo Qwen/Qwen3-14B-GGUF --ollama-import-model-name qwen3-14b-hf
```

## Confirm local models

```powershell
.\runtime\ollama\ollama.exe list
```

## Watch logs

```powershell
Get-Content .\logs\ai_workstation.log -Wait
Get-Content .\logs\ollama-runtime.log -Wait
```

## Build executable

```powershell
.\scripts\build_exe.ps1
```

## Common issue: app tries to pull from Ollama registry

Use the imported model name, normally `qwen3-14b-hf`, in the **Ollama Model** field. Do not use `qwen3:14b` unless you want to pull from the Ollama registry.
