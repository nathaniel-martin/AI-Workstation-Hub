import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ai_workstation_launcher.py"

spec = importlib.util.spec_from_file_location("ai_workstation_launcher", MODULE_PATH)
launcher = importlib.util.module_from_spec(spec)
sys.modules["ai_workstation_launcher"] = launcher
assert spec.loader is not None
spec.loader.exec_module(launcher)


def test_default_model_prefers_hf_import_name():
    assert launcher.DEFAULT_MODEL == "qwen3-14b-hf"


def test_choose_hf_gguf_file_prefers_q4_k_m():
    files = [
        {"name": "Qwen3-14B-Q8_0.gguf", "size": 1},
        {"name": "Qwen3-14B-Q4_K_M.gguf", "size": 1},
        {"name": "Qwen3-14B-Q2_K.gguf", "size": 1},
    ]
    assert launcher.choose_hf_gguf_file(files) == "Qwen3-14B-Q4_K_M.gguf"


def test_extract_json_object_rejects_manifest_without_action():
    raw = json.dumps({"name": "MCP Server", "dependencies": {"Flask": "2.0.1"}})
    try:
        launcher.extract_json_object(raw)
    except ValueError as exc:
        assert "action" in str(exc)
    else:
        raise AssertionError("Expected manifest JSON without action to be rejected")


def test_extract_json_object_accepts_action_with_noise():
    raw = "thinking... {\"thought\": \"inspect\", \"action\": \"list_files\"}"
    obj = launcher.extract_json_object(raw)
    assert obj["action"] == "list_files"


def test_workspace_path_escape_is_blocked(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    try:
        launcher.read_workspace_file(root, "../secrets.txt")
    except ValueError as exc:
        assert "escapes workspace" in str(exc).lower()
    else:
        raise AssertionError("Expected escaped path to raise ValueError")
