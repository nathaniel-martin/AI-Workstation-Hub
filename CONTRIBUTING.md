# Contributing

1. Create a feature branch.
2. Keep the app Windows-native and no-admin friendly.
3. Avoid dependencies that require local native compilation unless they are optional.
4. Run checks before opening a PR:

```powershell
python -m pip install -r requirements-dev.txt
ruff check .
pytest
python -m py_compile ai_workstation_launcher.py
```

5. Update docs and tests for user-facing behavior changes.
