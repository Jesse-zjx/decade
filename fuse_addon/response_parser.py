"""Shared faithful response extraction; never repairs or executes source.

The returned source is a contiguous, byte-identical UTF-8 span of the model's
content. Fenced syntax errors remain syntax errors for the common worker.
Unfenced extraction takes the entire suffix from the first exact, top-level
Python entry token; it never trims a tail until parsing happens to succeed.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import re

PARSER_VERSION = 'faithful-module-v1'
# Markdown opening/closing delimiters are metadata, not source bytes.
_OPEN = re.compile(r'^( {0,3})(`{3,}|~{3,})[ \t]*([^\r\n]*)[\r\n]*$')
_START = re.compile(r"^(?:(?:async[ \t]+)?def[ \t]+|class[ \t]+|import[ \t]+|from[ \t]+|(?:if|for|while|with|try|match)[ \t:]|@|[A-Za-z_]\w*[ \t]*(?:=(?!=)|:[ \t]*[^=\r\n]+=)|[rubfRUBF]{0,2}(?:\"{3}|'{3}))")


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ParsedResponse:
    status: str
    source: str | None
    description: str
    response_sha256: str
    source_sha256: str | None
    source_start: int | None
    source_end: int | None
    syntax_valid: bool | None
    syntax_error: str | None
    fence_count: int
    parser_version: str = PARSER_VERSION

    def evidence(self):
        value = asdict(self)
        value.pop('source')
        # Description is original text only; it is not an inferred insight.
        value['source_utf8_bytes'] = len(self.source.encode('utf-8')) if self.source is not None else None
        value['source_span_units'] = 'unicode_codepoints_in_decoded_response'
        value['source_is_exact_response_span'] = self.source is not None
        return value


def extract_response(response: str, entry_name='select_next_node') -> ParsedResponse:
    """Freeze extraction without AST rewriting or quality-informed selection.

    All nonempty Python/py/python3/unlabelled closed fences are considered.
    Duplicate byte-identical fences identify one source; distinct fences fail
    ambiguity rather than assembling fragments or choosing a candidate. An
    explicit unclosed Python fence fails; source syntax is diagnostic only.
    With no Python fence, only exact column-zero code tokens can start the
    module. The remaining suffix is indivisible, including any invalid tail.
    """
    if not isinstance(response, str):
        raise TypeError('response must be persisted decoded text')
    lines = response.splitlines(keepends=True)
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    spans = []
    fence_count = 0
    unclosed_python = False
    i = 0
    while i < len(lines):
        opening = _OPEN.match(lines[i])
        if not opening:
            i += 1
            continue
        marker, language = opening.group(2), opening.group(3).strip().lower()
        python = language in ('', 'python', 'py', 'python3')
        closing = re.compile(r'^ {0,3}' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}[ \t]*(?:\r?\n)?$')
        j = i + 1
        while j < len(lines) and not closing.match(lines[j]):
            j += 1
        if python:
            fence_count += 1
            if j == len(lines):
                unclosed_python = True
            else:
                start, end = offsets[i] + len(lines[i]), offsets[j]
                if response[start:end].strip():
                    spans.append((start, end, offsets[i]))
        i = j + 1
    status = None
    if unclosed_python:
        status = 'UNCLOSED_PYTHON_FENCE'
    elif spans:
        unique = {response[start:end] for start, end, _ in spans}
        if len(unique) != 1:
            status = 'AMBIGUOUS_CODE_MODULES'
    elif fence_count:
        status = 'EMPTY_PYTHON_FENCE'
    else:
        # Do not look inside a non-Python fenced block as an unfenced module.
        if any(_OPEN.match(line) for line in lines):
            status = 'NO_PYTHON_MODULE'
        else:
            found = next((n for n, line in enumerate(lines) if _START.match(line)), None)
            if found is None:
                status = 'NO_PYTHON_MODULE'
            else:
                # Include contiguous original comment/blank lines before the
                # first statement (but not prose or algorithm description).
                first = found
                while first and (not lines[first-1].strip() or lines[first-1].lstrip().startswith('#')):
                    first -= 1
                spans = [(offsets[first], len(response), offsets[first])]
    if status is not None:
        return ParsedResponse(status, None, '', _sha(response), None, None, None, None, None, fence_count)
    start, end, description_end = spans[0]
    source = response[start:end]
    description = response[:description_end].strip()
    try:
        tree = ast.parse(source)
        valid = True
        error = None
    except (SyntaxError, ValueError) as exc:
        tree = None
        valid = False
        error = f'{type(exc).__name__}: {exc}'
    # No entry signature, return, import, indentation or trailing source repair.
    # Actual ABI and source safety remain the common evaluator's responsibility.
    return ParsedResponse('EXTRACTED', source, description, _sha(response), _sha(source),
                          start, end, valid, error, fence_count)


def eoh_extract(response, *, parse_callback=None):
    result = extract_response(response or '')
    if parse_callback is not None:
        parse_callback(result)
    if result.source is None:
        return [], []
    # A missing description is recorded as the original empty string. A
    # faithful module remains usable; source dictionaries are never summaries.
    return [result.description], [result.source]


def hifo_get_alg(evolution, prompt_content, *, parse_callback=None):
    """HiFo adapter with its original initial + four extraction retry ceiling.

    UNKNOWN exceptions propagate before parsing and cannot be regenerated.
    Targeted slots enforce their existing lower maintenance ceiling in transport.
    """
    for _ in range(5):
        response = evolution.interface_llm.get_response(prompt_content)
        result = extract_response(response)
        if parse_callback is not None:
            parse_callback(result)
        if result.source is not None:
            return [result.source, result.description]
    raise ValueError('FAITHFUL_RESPONSE_EXTRACTION_FAILED: ' + result.status)
