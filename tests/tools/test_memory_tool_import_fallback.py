"""Regression tests for memory-tool import fallbacks."""

import subprocess
import sys


def test_memory_tool_imports_without_fcntl(tmp_path):
    code = f"""
import builtins
import sys
from pathlib import Path

original_import = builtins.__import__
def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fcntl":
        raise ImportError("simulated missing fcntl")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

import tools.memory_tool as memory_tool
from tools.registry import registry

memory_tool.get_memory_dir = lambda: Path(r'{tmp_path}')
store = memory_tool.MemoryStore(memory_char_limit=200, user_char_limit=200)
store.load_from_disk()
result = store.add("memory", "fact learned during import fallback test")

assert memory_tool.fcntl is None
assert registry.get_entry("memory") is not None
assert result["success"] is True
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
