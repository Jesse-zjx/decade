"""Deterministic envelope for a host-declared FULL action. No code execution.

Only G proposals and technical FULL repairs use this adapter. B/C still use
frozen ANE-r1 decoding. Wrapper metadata never chooses between candidate runs.
"""
from dataclasses import dataclass
import hashlib,re

@dataclass(frozen=True)
class GlobalEnvelope:
    code: str
    raw_sha256: str
    code_sha256: str
    discarded_mode_comments: tuple[str,...]
    route: str

def decode_global(raw: str) -> GlobalEnvelope:
    if not isinstance(raw,str) or not raw.strip():raise ValueError('EMPTY_GLOBAL_RESPONSE')
    lines=raw.splitlines(keepends=True)
    marks=[i for i,l in enumerate(lines) if l.strip().startswith('```')]
    if marks:
        if len(marks)!=2:raise ValueError('MULTIPLE_OR_UNCLOSED_FENCE')
        lo,hi=marks
        if not re.fullmatch(r'```(?:python|Python|py)?\s*',lines[lo].strip()) or lines[hi].strip()!='```':
            raise ValueError('INVALID_FENCE')
        body=lines[lo+1:hi];route='unique_fence_host_full'
    else:body=lines[:];route='raw_host_full'
    discarded=[]
    # Metadata outside a unique fence is not part of the returned executable body.
    for i,l in enumerate(lines):
        if l.strip() in ('# FUSE_FULL','# FUSE_REPLACE') and marks and (i<=marks[0] or i>=marks[1]):
            discarded.append(l.strip())
    # Inside the code body only a leading standalone mode-comment is metadata.
    first=next((i for i,l in enumerate(body) if l.strip()),None)
    if first is not None and body[first].strip() in ('# FUSE_FULL','# FUSE_REPLACE'):
        discarded.append(body[first].strip());body=body[:first]+body[first+1:]
    code=''.join(body)
    if not code.strip():raise ValueError('EMPTY_GLOBAL_CODE')
    return GlobalEnvelope(code,hashlib.sha256(raw.encode()).hexdigest(),hashlib.sha256(code.encode()).hexdigest(),tuple(discarded),route)
