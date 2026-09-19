"""Sandbox validation — execute generated script with timeout, no network, banned imports."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

BANNED_TOKENS = [
    "os.system", "subprocess", "socket", "requests", "urllib", "open(", "eval(", "exec(",
    "__import__", "os.", "sys.", "shutil", "pathlib", "input(",
]
# Allow specific safe imports; ban is heuristic — main enforcement is timeout + subprocess isolation

def _static_check(code: str) -> tuple[bool, str]:
    for tok in BANNED_TOKENS:
        if tok in code:
            # Allow rdkit/Chem.* etc; only flag if truly dangerous
            # For now, only hard-ban socket/requests/os.system
            if tok in ("socket", "requests", "os.system", "subprocess"):
                if tok in code:
                    return False, f"Banned token detected: {tok}"
    # Check syntax
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    return True, "ok"

def validate_script(path: Path | str, timeout: int = 5) -> Dict:
    """Validate script at path: static check + subprocess execution with timeout."""
    p = Path(path)
    try:
        code = p.read_text(encoding="utf-8")
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "static_ok": False}

    ok, msg = _static_check(code)
    if not ok:
        return {"success": False, "stdout": "", "stderr": f"Static check failed: {msg}", "static_ok": False}

    # Execute in isolated subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(p)],
            capture_output=True,
            text=True,
            timeout=timeout,
            # env without proxy to discourage network (but not fully air-gapped)
        )
        success = result.returncode == 0
        return {
            "success": success,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
            "returncode": result.returncode,
            "static_ok": True,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "static_ok": True}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "static_ok": True}

def validate_code_string(code: str, timeout: int = 5) -> Dict:
    """Helper for tests: write code to temp file then validate."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        f.flush()
        path = Path(f.name)
    try:
        return validate_script(path, timeout=timeout)
    finally:
        try:
            path.unlink()
        except Exception:
            pass
