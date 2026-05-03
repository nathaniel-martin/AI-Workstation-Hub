# Security Policy

## Supported use

AI Workstation Hub is intended for local, single-user workstation use. It starts local services on loopback by default.

## Agent safety model

The coding agent is workspace-scoped. File reads and writes are blocked when they resolve outside the configured workspace root. Shell command execution is disabled unless explicitly enabled.

Recommended defaults:

- keep UI host on `127.0.0.1`
- keep shell execution disabled unless you are working in a disposable or trusted workspace
- do not store secrets in `ai_workstation_settings.json`
- use `HF_TOKEN` environment variable rather than the settings file for Hugging Face tokens
- review all generated code before running it in privileged environments

## Reporting vulnerabilities

Open a private advisory or contact the repository owner. Do not publish exploit details until a fix is available.
