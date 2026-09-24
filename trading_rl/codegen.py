"""Extracting and validating model-generated Strategy code."""

import ast
import re
import sys

import backtrader as bt

CLASS_HEADER = "class Strategy(bt.Strategy):"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_INIT_RE = re.compile(r'def\s+__init__\s*\([^)]*\)\s*:')
_NEXT_RE = re.compile(r'def\s+next\s*\([^)]*\)\s*:')
_STDLIB = {m.lower() for m in sys.stdlib_module_names}


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return _THINK_RE.sub("", text or "").strip()


def extract_function(text: str):
    """Extract the `class Strategy(bt.Strategy):` block from a fenced block or raw output."""
    text = strip_thinking(text)
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip().removeprefix("python\n")
        fx = fx[fx.find("class Strategy"):]
        if fx.startswith(CLASS_HEADER):
            return fx
    idx = text.find(CLASS_HEADER)
    return text[idx:] if idx != -1 else None


def has_required_functions(code) -> bool:
    code = code or ""
    return bool(_INIT_RE.search(code)) and bool(_NEXT_RE.search(code))


def check_python_modules(code: str):
    """Same contract as unsloth's check_python_modules: ok only if all imports are stdlib."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {"error": f"SyntaxError: {e}", "stdlib": [], "non_stdlib": []}

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            imports.add(node.module.split(".")[0])

    stdlib = sorted(m for m in imports if m.lower() in _STDLIB)
    non_stdlib = sorted(m for m in imports if m.lower() not in _STDLIB)
    return not non_stdlib, {"stdlib": stdlib, "non_stdlib": non_stdlib}


def function_works(code) -> bool:
    """True if the code parses and only imports stdlib modules."""
    if code is None:
        return False
    ok, info = check_python_modules(code)
    return ok and "error" not in info


def extract_strategy(code: str):
    """exec the code with `bt` in scope and return the Strategy class."""
    namespace = {'bt': bt}
    exec(code, namespace)
    return namespace['Strategy']
