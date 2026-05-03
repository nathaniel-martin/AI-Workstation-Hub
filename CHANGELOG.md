# Changelog

All notable changes to this project will be documented in this file.

This project follows a practical subset of [Keep a Changelog](https://keepachangelog.com/) and uses semantic versioning for release tags.

## [0.1.0] - 2026-05-03

### Added

- Portable Windows-native AI Workstation Hub launcher.
- Flat Gradio UI for local chat, diagnostics, Hugging Face import, and coding-agent workflows.
- Portable Ollama runtime management with no-admin assumptions.
- Hugging Face GGUF download and local Ollama import flow.
- Persistent settings file support.
- Runtime and launcher log capture.
- Workspace-scoped coding agent with bounded file operations.
- Optional shell execution, disabled by default.
- PyInstaller Windows executable build script.
- GitHub Actions CI for lint, tests, CLI smoke check, and Windows executable artifact build.
- Security, operations, architecture, and contribution documentation.

### Security

- Agent file operations are constrained to the configured workspace.
- Shell execution is opt-in.
- Local services bind to loopback by default.
