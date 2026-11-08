"""Version-bound one/two-slot line envelopes, independent static assembly.

Thin adaptation of engineering/siblings/inherited_sibling_protocol.py. No
transport, evaluator or candidate invocation. A complete c1 survives c2 loss.
"""
from __future__ import annotations
import ast
import copy
import difflib
import hashlib
import re

PROTOCOL = 'FUSE_SIBLINGS_V2'
_BEGIN = re.compile(r'^\s*BEGIN_CANDIDATE\s+([A-Za-z0-9_-]+)\s*$')
_END = re.compile(r'^\s*END_CANDIDATE\s+([A-Za-z0-9_-]+)\s*$')
_FENCE = re.compile(r'^\s*(`{3,}|~{3,})([^\r\n]*)$')


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _segments(raw):
    segments, outside = [], []
    active, fence = None, None
    for line in raw.splitlines(keepends=True):
        stripped = line.rstrip('\r\n')
        mark = _FENCE.fullmatch(stripped)
        if fence is not None:
            if active is not None:
                active['lines'].append(line)
            else:
                outside.append(line)
            if mark and mark[1][0] == fence[0] and len(mark[1]) >= len(fence) and not mark[2].strip():
                fence = None
            continue
        begin, end = _BEGIN.fullmatch(stripped), _END.fullmatch(stripped)
        if begin:
            if active is not None:
                active['segment_error'] = 'MISSING_END_MARKER'
            active = dict(id=begin[1], lines=[line], ended=False, segment_error=None)
            segments.append(active)
        elif end:
            if active is None:
                outside.append(line)
            else:
                active['lines'].append(line)
                active['ended'] = end[1] == active['id']
                if not active['ended']:
                    active['segment_error'] = 'END_SLOT_ID_MISMATCH'
                active = None
        elif active is not None:
            active['lines'].append(line)
            if mark:
                fence = mark[1]
        else:
            outside.append(line)
            if mark:
                fence = mark[1]
    return segments, ''.join(outside)


def _body(segment):
    headers, code, after = [], [], []
    fence, opened, closed, language = None, False, False, None
    for line in segment['lines'][1:]:
        s = line.rstrip('\r\n')
        if closed:
            if not _END.fullmatch(s):
                after.append(line)
            continue
        marker = _FENCE.fullmatch(s)
        if not opened:
            if marker:
                opened = True
                fence, language = marker[1], marker[2].strip()
            elif not _END.fullmatch(s):
                headers.append(line)
        elif marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
            closed = True
        else:
            code.append(line)
    return ''.join(headers), ''.join(code), ''.join(after), opened, closed, language


def decode_response(raw, pre, request_id):
    """Return each pre['slot_order'] slot, with raw_output/body/repairable/status.

    Additional slot IDs never become candidates. Duplicate expected IDs make
    only that slot ambiguous. Text inside code fences cannot start/end a slot.
    A body must exist for a format failure to be technically repairable.
    """
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('LOGICAL_REQUEST_ID_REQUIRED')
    required = ('parent_sha', 'context_sha', 'archive_snapshot_sha', 'archive_source_hashes', 'slots', 'slot_order')
    if any(k not in pre for k in required):
        raise ValueError('INCOMPLETE_HOST_BINDING')
    order = pre['slot_order']
    if not order or len(order) > 2 or len(set(order)) != len(order) or any(x not in ('c1', 'c2') for x in order):
        raise ValueError('INVALID_FROZEN_SLOT_ORDER')
    text = raw if isinstance(raw, str) else ''
    segments, outside = _segments(text)
    result = dict(protocol=PROTOCOL, request_id=request_id, response_sha=sha(text), raw_response=raw,
                  slots_consumed=len(order), slots=[], physical_request_charge_count=1,
                  per_candidate_shared_token_allocation=None,
                  archive_source_hashes=list(pre['archive_source_hashes']),
                  extra_selector_calls=0, extra_slot_ids=[s['id'] for s in segments if s['id'] not in order],
                  outside_text=outside, extra_output_fee_preserved=True)
    for cid in order:
        frozen = pre['slots'][cid]
        slot = dict(candidate_id=cid, slot_id=cid, normal_index=frozen['normal_index'],
                    proposal_id=request_id + '/' + cid, parent_sha=pre['parent_sha'], context_sha=pre['context_sha'],
                    archive_snapshot_sha=pre['archive_snapshot_sha'], status='INVALID', error=None,
                    quality=None, attribution=None, evaluation_id=request_id + '/' + cid + '/feedback',
                    slot_consumed=True, body='', raw_output='', repairable=False, truncated=False,
                    request_id=request_id, shared_cost_request_id=request_id)
        result['slots'].append(slot)
        occurrences = [s for s in segments if s['id'] == cid]
        if not occurrences:
            slot['error'] = 'MISSING_CANDIDATE_NO_BODY'
            continue
        slot['raw_output'] = ''.join(''.join(s['lines']) for s in occurrences)
        parsed = [_body(s) for s in occurrences]
        slot['body'] = parsed[0][1]
        slot['has_implementation_body'] = any(bool(p[1].strip()) for p in parsed)
        slot['repairable'] = slot['has_implementation_body']
        if len(occurrences) != 1:
            slot['error'] = 'DUPLICATE_SLOT_ID_AMBIGUOUS'
            continue
        segment = occurrences[0]
        header, code, after, opened, closed, language = parsed[0]
        slot['truncated'] = bool(opened and not closed or not segment['ended'])
        try:
            if segment['segment_error']:
                raise ValueError(segment['segment_error'])
            if not opened or language not in ('python', 'py'):
                raise ValueError('SINGLE_PYTHON_CODE_FENCE_REQUIRED')
            if not closed:
                raise ValueError('TRUNCATED_CODE_BODY')
            if not segment['ended']:
                raise ValueError('MISSING_END_MARKER')
            if after.strip():
                raise ValueError('EXTRA_OUTPUT_IN_SLOT')
            fields = {}
            for line in header.splitlines():
                if not line.strip():
                    continue
                name, sep, value = line.partition(':')
                name = name.strip()
                if not sep or name not in ('TARGET', 'MODE', 'EVIDENCE', 'INTENT') or name in fields:
                    raise ValueError('EXACT_UNAMBIGUOUS_SLOT_HEADERS_REQUIRED')
                fields[name] = value.strip()
            if set(fields) != {'TARGET', 'MODE', 'EVIDENCE', 'INTENT'}:
                raise ValueError('FOUR_SLOT_HEADERS_REQUIRED')
            if not fields['INTENT'] or len(fields['INTENT'].split()) > 35 or '\n' in fields['INTENT']:
                raise ValueError('INTENT_ONE_LINE_MAX_35_WORDS')
            if not code.strip():
                raise ValueError('EMPTY_IMPLEMENTATION_BODY')
            target, mode = fields['TARGET'], fields['MODE']
            if target not in frozen['menu']:
                raise ValueError('UNREGISTERED_TARGET')
            unit = frozen['menu'][target]
            if frozen['parent_sha'] != pre['parent_sha'] or unit.get('parent_sha') != pre['parent_sha']:
                raise ValueError('MENU_PARENT_BINDING_MISMATCH')
            if mode not in ('REPLACE', 'FULL') or ((mode == 'FULL') != (target == 'GLOBAL')):
                raise ValueError('TARGET_MODE_CONFLICT')
            if frozen['host_full'] and mode != 'FULL':
                raise ValueError('HOST_REQUIRES_FULL')
            refs = [] if fields['EVIDENCE'] == 'NONE' else [s.strip() for s in fields['EVIDENCE'].split(',')]
            valid = {c['id'] for c in frozen['memory']['cases']}
            if len(set(refs)) != len(refs) or any(r not in valid for r in refs):
                raise ValueError('INVALID_EVIDENCE_REFERENCE')
            proposal = dict(candidate_id=cid, parent_sha=pre['parent_sha'], target=target, target_id=target,
                            mode=mode, evidence=refs, evidence_ids=refs, intent=fields['INTENT'], code=code)
            slot.update(status='BOUND', error=None, repairable=False, proposal=proposal, unit=copy.deepcopy(unit),
                        body_sha=sha(code), attribution_scope='WHOLE_PROGRAM_ONLY' if mode == 'FULL' else 'PAIR_PENDING')
        except (ValueError, TypeError) as exc:
            slot['error'] = str(exc)
    return result


def assemble_plans(batch, parent, assembler):
    """Pure assembly: original P for every child; no imports/execute of candidate.

    STATIC duplicates are classified against the pre-request archive first, then
    against c1. A duplicate may reference an exact cached quality but must never
    create an independent measured edit. Execute failures may be marked repairable
    by the caller only after the original evaluator has returned a true error.
    """
    result = copy.deepcopy(batch)
    seen = {}
    for slot in result['slots']:
        if slot['status'] != 'BOUND':
            continue
        try:
            if sha(parent) != slot['parent_sha']:
                raise ValueError('HOST_PARENT_BYTES_MISMATCH')
            region = slot['unit']
            if region.get('parent_sha') != slot['parent_sha']:
                raise ValueError('MENU_PARENT_BINDING_MISMATCH')
            lines = parent.splitlines(keepends=True)
            previous = 0
            for a, b in region['spans']:
                if not (type(a) is int and type(b) is int and previous < a <= b <= len(lines)):
                    raise ValueError('INVALID_REGISTERED_SPAN_UNION')
                previous = b
            registered = ''.join(''.join(lines[a - 1:b]) for a, b in region['spans'])
            if sha(registered) != region['region_sha']:
                raise ValueError('REGISTERED_REGION_BYTES_MISMATCH')
            proposal = slot['proposal']
            child, support = assembler(parent, copy.deepcopy(region), '# FUSE_' + proposal['mode'] + '\n' + proposal['code'])
            parent_bytes, child_bytes = parent.encode(), child.encode()
            changes = [op for op in difflib.SequenceMatcher(None, parent_bytes, child_bytes, autojunk=False).get_opcodes() if op[0] != 'equal']
            removed = sum(b - a for _, a, b, _, _ in changes)
            added = sum(d - c for _, _, _, c, d in changes)
            support = copy.deepcopy(support)
            support.update(changed_parent_bytes=removed, changed_child_bytes=added,
                           changed_byte_ratio=max(removed, added) / max(1, len(parent_bytes)),
                           changed_byte_ratio_definition='max(deleted_parent_bytes,inserted_child_bytes)/parent_utf8_bytes',
                           parent_top_level_functions=[n.name for n in ast.parse(parent).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                           child_top_level_functions=[n.name for n in ast.parse(child).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            slot.update(child_source=child, child_sha=sha(child), support=support,
                        status='ASSEMBLED_NOT_EVALUATED', error=None, repairable=False)
            if slot['child_sha'] in result['archive_source_hashes']:
                slot.update(status='DUPLICATE_ARCHIVE_REQUIRES_EXACT_OBSERVATION', duplicate_of=slot['child_sha'])
            elif slot['child_sha'] in seen:
                slot.update(status='DUPLICATE_SIBLING_PENDING_OBSERVATION', duplicate_of=seen[slot['child_sha']])
            else:
                seen[slot['child_sha']] = slot['proposal_id']
        except (ValueError, SyntaxError, TypeError, KeyError) as exc:
            slot.update(status='ASSEMBLY_FAILED', error=str(exc), repairable=bool(slot.get('body', '').strip()))
    return result


def unique_physical_tokens(request_rows):
    """Whole-response bill once; unknown bills remain unknown (no slot allocation)."""
    unique = {}
    for row in request_rows:
        rid = row['physical_request_id']
        if rid in unique and unique[rid] != row:
            raise ValueError('CONFLICTING_REQUEST_RECEIPT')
        unique[rid] = row
    missing = sum(row.get('tokens') is None for row in unique.values())
    known = sum(row['tokens'] for row in unique.values() if row.get('tokens') is not None)
    return dict(physical_requests=len(unique), tokens_known=known, usage_missing=missing,
                tokens=None if missing else known)
