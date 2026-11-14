"""Extracting and validating model-generated Strategy code."""

import ast
import re
import sys

import backtrader as bt

CLASS_HEADER = "class Strategy(bt.Strategy):"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_HEADER_RE = re.compile(r"^([ \t]*)class\s+Strategy\s*\(\s*bt\.Strategy\s*\)\s*:", re.MULTILINE)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_INIT_RE = re.compile(r'def\s+__init__\s*\([^)]*\)\s*:')
_NEXT_RE = re.compile(r'def\s+next\s*\([^)]*\)\s*:')
_STDLIB = {m.lower() for m in sys.stdlib_module_names}


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks produced by reasoning models."""
    return _THINK_RE.sub("", text or "").strip()


def _parses(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _cut_class(text):
    """The `class Strategy(bt.Strategy):` block in text, up to the first non-blank, non-comment
    line not indented deeper than the header (trailing prose, closing fence, next definition).
    The block is dedented so the header starts at column 0. None if there is no header."""
    m = _HEADER_RE.search(text)
    if not m:
        return None
    indent = len(m.group(1))
    lines = text[m.start():].splitlines()
    body = [lines[0]]
    for line in lines[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    code = "\n".join(line[indent:] for line in body).rstrip() + "\n"
    if not _parses(code):
        # e.g. a docstring line at column 0: fall back to everything after the header,
        # if that parses (otherwise keep the cut block; it fails validation either way).
        rest = "\n".join(line[indent:] for line in lines).rstrip() + "\n"
        if _parses(rest):
            return rest
    return code


def extract_function(text: str):
    """Extract the `class Strategy(bt.Strategy):` block. Fenced code blocks containing the
    class win (the first one that does); otherwise the raw text is searched. Anything after
    the class body (explanations, usage examples, a closing fence) is dropped."""
    text = strip_thinking(text)
    for block in _FENCE_RE.findall(text):
        code = _cut_class(block)
        if code:
            return code
    return _cut_class(text)


def _strategy_methods(code):
    """Method names defined directly in `class Strategy`, or None if the code doesn't parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Strategy":
            return {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return set()


def has_required_functions(code) -> bool:
    """True if class Strategy defines __init__ and next. Code with a syntax error falls back
    to a regex, so it is scored as invalid_code (-3) rather than missing_methods (-10)."""
    code = code or ""
    methods = _strategy_methods(code)
    if methods is None:
        return bool(_INIT_RE.search(code)) and bool(_NEXT_RE.search(code))
    return {"__init__", "next"} <= methods


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


def validate_code(code):
    """None if the code parses and only imports stdlib modules, else a short error message."""
    if code is None:
        return "No `class Strategy(bt.Strategy):` definition found in the output."
    ok, info = check_python_modules(code)
    if "error" in info:
        return info["error"]
    if not ok:
        return f"Imports are not allowed (bt is already available): {', '.join(info['non_stdlib'])}"
    if _RAW_FEED_RE.search(code):
        return ("Reading the raw data feed (dataname) is not allowed: it holds every bar of the "
                "window, including future ones. Use self.data.close[0], [-1], ... and indicators.")
    return None


# The pandas frame behind the feed (self.data.p.dataname / _dataname) holds the whole window,
# so reading it is lookahead. Positive line indices (self.data.close[1]) are blocked at run time
# instead: the backtester runs with preload=False, so future bars are not loaded yet.
_RAW_FEED_RE = re.compile(r'dataname')


def function_works(code) -> bool:
    """True if the code parses and only imports stdlib modules."""
    return validate_code(code) is None


def extract_strategy(code: str):
    """exec the code with `bt` in scope and return the Strategy class."""
    namespace = {'bt': bt}
    exec(code, namespace)
    return namespace['Strategy']


def describe_exception(e: BaseException) -> str:
    """'TypeError (line 7): ...' where the line is inside the generated code, if known."""
    import traceback
    lines = [f.lineno for f in traceback.extract_tb(e.__traceback__) if f.filename == '<string>']
    loc = f" (line {lines[-1]})" if lines else ""
    return f"{type(e).__name__}{loc}: {e}"[:300]
