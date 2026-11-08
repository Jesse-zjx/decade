"""ANE-r1 deterministic presentation decoder. No execution or LLM calls.

A protocol amendment, NOT a way to rewrite outcomes from the sealed study.
The existing assembler, task API checks and trusted evaluator remain authoritative.
"""
from dataclasses import dataclass
import ast
import hashlib
import re
from typing import Literal

Mode = Literal['FULL', 'REPLACE']

class EnvelopeError(ValueError):
    pass

@dataclass(frozen=True)
class Decoded:
    mode: Mode
    body: str
    route: str
    raw_sha256: str
    body_sha256: str
    fenced: bool

_MARKERS = {'# FUSE_FULL': 'FULL', '# FUSE_REPLACE': 'REPLACE'}
_OPEN = re.compile(r'^\s*(`{3,}|~{3,})([^\r\n]*)')

def _leading_marker(text: str):
    rows = text.splitlines(keepends=True)
    first = next((i for i, line in enumerate(rows) if line.strip()), None)
    if first is None:
        raise EnvelopeError('FORMAT_EMPTY')
    mode = _MARKERS.get(rows[first].strip())
    if mode is not None:
        # Remove only leading whitespace-only lines and the explicit metadata line.
        return mode, ''.join(rows[first + 1:])
    return None, ''.join(rows[first:])

def _obviously_full(body: str, entrypoint: str) -> bool:
    """Sufficient, not necessary condition: module imports and exact task entry.

    A pure function set without imports remains the intended C replacement.
    No semantic inference or trial execution determines the mode.
    """
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return False
    return (any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body)
            and any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == entrypoint for n in tree.body))

def decode(response: str, *, requested_scope: str, entrypoint: str) -> Decoded:
    if not isinstance(response, str):
        raise TypeError('response must be str')
    if requested_scope not in {'B', 'C', 'G'} or not entrypoint.isidentifier():
        raise ValueError('invalid declared task/scope')
    rows = response.splitlines(keepends=True)
    opened = None
    token = None
    spans = []
    for i, line in enumerate(rows):
        m = _OPEN.match(line)
        if not m:
            continue
        if opened is None:
            opened, token = i, m.group(1)
        elif (m.group(1)[0] == token[0] and len(m.group(1)) >= len(token)
              and not m.group(2).strip()):
            spans.append((opened, i))
            opened, token = None, None
    if opened is not None or len(spans) > 1:
        raise EnvelopeError('FORMAT_MULTIPLE_OR_UNCLOSED_FENCE')
    if spans:
        a, b = spans[0]
        inside, body = _leading_marker(''.join(rows[a + 1:b]))
        exterior = rows[:a] + rows[b + 1:]
        external_modes = [_MARKERS[line.strip()] for line in exterior if line.strip() in _MARKERS]
        if len(set(external_modes)) > 1:
            raise EnvelopeError('FORMAT_CONFLICTING_MODES')
        outside = external_modes[0] if external_modes else None
        if inside and outside and inside != outside:
            raise EnvelopeError('FORMAT_CONFLICTING_MODES')
        mode = inside or outside
        route = 'explicit_inside' if inside else ('explicit_outside' if outside else None)
    else:
        mode, body = _leading_marker(response)
        route = 'explicit_bare' if mode else None
    if not body.strip():
        raise EnvelopeError('FORMAT_EMPTY_CODE')
    if mode is None:
        if requested_scope == 'G':
            mode, route = 'FULL', 'declared_global_default'
        elif _obviously_full(body, entrypoint):
            mode, route = 'FULL', 'module_imports_and_entry'
        else:
            mode, route = 'REPLACE', 'declared_local_default'
    return Decoded(mode, body, route,
                   hashlib.sha256(response.encode()).hexdigest(),
                   hashlib.sha256(body.encode()).hexdigest(), bool(spans))

def canonical_for_legacy(decoded: Decoded) -> str:
    """Feed to the unchanged assembler. Do not treat compile success as validity."""
    return f'# FUSE_{decoded.mode}\n' + decoded.body
