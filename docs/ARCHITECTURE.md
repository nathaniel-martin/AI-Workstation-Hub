# Architecture

```text
Gradio flat UI / CLI
        ↓
AI Workstation Launcher
        ↓
Portable Ollama runtime on 127.0.0.1:11434
        ↓
Imported local model, usually qwen3-14b-hf
```

## Design constraints

- Native Windows support.
- No administrator requirement.
- No WSL2 or Docker dependency.
- Avoid Python native CUDA/GGUF/AWQ builds.
- Prefer Hugging Face download/import when corporate networks block Ollama registry blobs.

## Runtime directories

- `runtime/ollama/` stores the portable Ollama executable.
- `runtime/ollama-models/` stores Ollama model manifests/blobs through `OLLAMA_MODELS`.
- `models/huggingface/` stores downloaded GGUF files.
- `models/ollama-imports/` stores generated Modelfiles.
- `logs/` stores app and runtime logs.
- `workspace/` is the default agent workspace.

## Agent loop

The coding agent requests one JSON action from the model at a time, executes a workspace-bound tool, returns the result to the model, and repeats until `finish` or max steps.

Supported actions:

- `list_files`
- `read_file`
- `write_file`
- `create_file`
- `run_command`
- `finish`
