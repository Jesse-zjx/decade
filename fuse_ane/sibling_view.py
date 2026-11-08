"""Frozen single/shared-parent compact views; no transport or candidate execution.

The lossless contrast compiler is the frozen inherited compiler supplied by this
study's authority. Candidate-slot schedules advance independently of HTTP calls.
Host state and records retain rich metadata; messages deliberately transmit only
current parent/elite means plus visibility-controlled compact edit contrasts.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import random
from pathlib import Path

from fuse_addon.evaluation import digest, encoded
from .memory_view import full_catalog
from .edit_evidence import match_case
from .reactive_policy import FocusChain

AUTHORITY = Path(__file__).resolve().parents[1] / 'authority' / 'shared_edit_siblings'
_SPEC = importlib.util.spec_from_file_location('_siblings_frozen_contrast', AUTHORITY / 'reference' / 'inherited_contrast_compiler.py')
_CONTRAST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRAST)
SCHEDULE = ('B', 'B', 'C', 'C', 'ALT', 'MIX')


def _key(region):
    return region['scope'], tuple(map(tuple, region['spans']))


def prepare(io, evidence, slot_numbers, *, repair_of=None):
    """Freeze one or two normal slots against one current Feedback elite.

    Example: pre=prepare(io, records, [1,2]); messages(io,pre).
    slot_numbers are one-based normal candidate positions, NOT request numbers.
    Repairs use prepare_repair(existing_pre,slot_id), never a newly read archive.
    """
    if repair_of is not None:
        raise ValueError('USE_PREPARE_REPAIR_ON_FROZEN_PARENT')
    numbers = list(slot_numbers)
    if len(numbers) not in (1, 2) or any(type(i) is not int or i < 1 for i in numbers):
        raise ValueError('ONE_OR_TWO_VALID_NORMAL_SLOTS_REQUIRED')
    if len(numbers) == 2 and (numbers[0] % 2 != 1 or numbers[1] != numbers[0] + 1):
        raise ValueError('BATCH_MUST_BE_ONE_ORDERED_NORMAL_PAIR')
    if io.arm not in ('F1', 'F2', 'N2'):
        raise ValueError('SIBLING_ARM_REQUIRED')
    if (io.arm == 'F1') != (len(numbers) == 1):
        raise ValueError('ARM_PAIR_SIZE_CONFLICT')
    programs = io.state['programs']
    parent = min(programs, key=lambda h: (programs[h]['mean'], h))
    source = io.source(parent)
    if digest(source) != parent:
        raise ValueError('PARENT_BYTES_MISMATCH')
    index = numbers[0]
    rows = full_catalog(io, parent, index)
    planned = [SCHEDULE[(n - 1) % len(SCHEDULE)] for n in numbers]
    scope = planned[0] if planned[0] in ('B', 'C') else 'G'
    pool = [r for r in rows if r['scope'] == scope]
    if not pool:
        scope = 'C' if scope == 'B' and any(r['scope'] == 'C' for r in rows) else 'G'
        pool = [r for r in rows if r['scope'] == scope]
    rng = random.Random(io.rng('edit-menu', index))
    rng_before = copy.deepcopy(rng.getstate())
    matches, omitted = [], []
    for record in evidence:
        match, reason = match_case(record, source, rows, io.context, index, programs)
        if match:
            matches.append(match)
        else:
            omitted.append(dict(evidence_id=record['evidence_id'], reason=reason))
    matches.sort(key=lambda m: (m['status'] == 'HISTORICAL_LINEAGE', -m['record']['episode'],
                                -m['record']['attempt'], m['record']['evidence_id']))
    domain = pool if scope != 'G' else [r for r in rows if r['scope'] in ('B', 'C')]
    selected = []
    for match in matches:
        region = match['region']
        if _key(region) in {_key(r) for r in domain} and _key(region) not in {_key(r) for r in selected}:
            selected.append(region)
            if len(selected) == 2:
                break
    rest = [r for r in domain if _key(r) not in {_key(s) for s in selected}]
    selected += rng.sample(rest, min(6 - len(selected), len(rest)))
    menu = {f'u{i:02d}': copy.deepcopy(r) for i, r in enumerate(selected)}
    menu['GLOBAL'] = copy.deepcopy(next(r for r in rows if r['scope'] == 'G'))
    ids = {_key(r): uid for uid, r in menu.items()}
    ordered, used = [], set()
    for match in matches:
        key = _key(match['region'])
        if key in ids and key not in used:
            ordered.append(match)
            used.add(key)
    ordered += [m for m in matches if _key(m['region']) in ids and m not in ordered]
    host_cards, compact_cards, rich_charge = [], [], 0
    for match in ordered:
        if len(host_cards) == 4:
            break
        record = match['record']
        card = dict(id=f'e{len(host_cards):02d}', evidence_id=record['evidence_id'],
                    target=ids[_key(match['region'])], context_status=match['status'],
                    matched_side=match['matched_side'], current_effect_unknown=match['current_effect_unknown'],
                    target_credit=copy.deepcopy(match['current_target_credit']),
                    effect=copy.deepcopy(record['effect']))
        compact = _CONTRAST.compact_case(source, menu, card, record, True)
        charge = len(json.dumps(compact, ensure_ascii=False, separators=(',', ':')))
        if rich_charge + charge > 8192:
            omitted.append(dict(evidence_id=record['evidence_id'], reason='WHOLE_CONTRAST_OVER_MEMORY_CAP'))
            continue
        # Admission has already used the full visible-effect card, for ALL arms.
        if io.arm == 'N2':
            compact['observed_comparison'] = None
            card['effect'] = None
            card['target_credit'] = None
        host_cards.append(card)
        compact_cards.append(compact)
        rich_charge += charge
    memory = dict(cases=host_cards, status='AVAILABLE' if host_cards else 'EMPTY_MEMORY',
                  compact_cases=compact_cards, rich_card_char_charge=rich_charge, omitted=omitted)
    quality = dict(parent=programs[parent]['mean'], elite=min(r['mean'] for r in programs.values()),
                   scale=io.reference['mean'])
    donor, donor_sampling = None, None
    if 'MIX' in planned:
        donor_index = numbers[planned.index('MIX')]
        donor, donor_sampling = FocusChain(io.rng('edit-donor', donor_index)).donor(programs, parent)
    slots = {}
    for normal_index, scheduled in zip(numbers, planned):
        sid = 'c1' if normal_index % 2 else 'c2'
        operator = 'MIX' if scheduled == 'MIX' else 'ALT' if scope == 'G' else 'REFINE'
        slot_menu = copy.deepcopy(menu)
        slot_donor = donor if operator == 'MIX' else None
        for uid, region in slot_menu.items():
            region.update(parent_sha=parent, donor_sha=slot_donor, operator=operator, operation=operator,
                          action_id=uid, local_key=digest(parent + '|' + region['region_sha'] + '|' + operator))
        slots[sid] = dict(slot_id=sid, normal_index=normal_index, episode=normal_index,
                          parent_sha=parent, menu=slot_menu, memory=copy.deepcopy(memory),
                          planned_scope=scheduled, effective_scope=scope, operator=operator,
                          host_full=scope == 'G', donor_sha=slot_donor,
                          donor_sampling=copy.deepcopy(donor_sampling) if slot_donor else None,
                          quality_before=copy.deepcopy(quality), target_selected_pre_evaluation=False,
                          selector_extra_http=0, menu_rng_before=copy.deepcopy(rng_before),
                          menu_rng_after=copy.deepcopy(rng.getstate()))
    return dict(protocol='FUSE_SIBLINGS_V2', arm=io.arm, normal_slot_numbers=numbers,
                batch_id='/'.join(str(io.context[k]) for k in ('study_id','phase','task','group','arm'))+f'/step-{numbers[0]:03d}',
                repair_pair_index=(numbers[0]-1)//2,
                slot_order=list(slots), slots=slots, parent_sha=parent, parent_source=source,
                context=copy.deepcopy(io.context), context_sha=digest(encoded(io.context)),
                archive_source_hashes=sorted(programs), archive_snapshot_sha=digest(encoded(programs)),
                menu_rng_before=rng_before, menu_rng_after=copy.deepcopy(rng.getstate()),
                quality_before=quality, memory=memory, donor_sha=donor,
                donor_source=io.source(donor) if donor else None,
                physical_request_charge_count=1, sibling_independent_statistical_samples=False)


def legacy_pre(pre, slot_id):
    """Deep copied single-slot pre containing menu/memory/quality/episode for legacy evidence seams."""
    return copy.deepcopy(pre['slots'][slot_id])


def prepare_repair(pre, slot_id):
    """Narrow the existing frozen batch to one FULL repair; do not consult IO state."""
    result = copy.deepcopy(pre)
    slot = result['slots'][slot_id]
    slot['host_full'] = True
    slot['repair_of'] = slot_id
    # All original cases and menu references remain; only GLOBAL may execute.
    result['slots'] = {slot_id: slot}
    result['slot_order'] = [slot_id]
    result['repair_of'] = slot_id
    result['normal_slot_numbers'] = [slot['normal_index']]
    return result


def _operation_instructions(slot, donor_available):
    # Exact substantive operation paragraphs inherited from the existing RNE
    # templates; omit their repeated parent, historical numeric feedback and
    # obsolete output wrappers. No donor score or donor ranking is transmitted.
    if slot['operator'] == 'REFINE':
        text = (Path(__file__).with_name('prompts') / 'REFINE.txt').read_text()
        return text[text.index('Improve the algorithmic logic'):text.index('Return a complete replacement')].strip()
    name = 'G_ALT' if slot['operator'] == 'ALT' else 'G_MIX' if donor_available else 'G_BACKBONE_SINGLE'
    text = (Path(__file__).with_name('prompts') / (name + '.txt')).read_text()
    text = text[text.index('Operation:'):]
    stop = text.find('The host has')
    if name == 'G_BACKBONE_SINGLE':
        stop = text.find('Return only')
    return text[:stop].strip() if stop >= 0 else text.strip()


def messages(io, pre, *, failed_slot=None, error=None, failed_output=None):
    """Serialize P once and per-slot compact contracts, with no recent-quality side channels.

    For a repair pass prepare_repair(pre,sid), failed_slot=sid and the ORIGINAL
    slot-only raw output plus its technical error. Never pass a batch's results.
    """
    repair = failed_slot is not None
    if repair and (pre.get('repair_of') != failed_slot or pre['slot_order'] != [failed_slot]):
        raise ValueError('REPAIR_REQUIRES_FROZEN_NARROWED_PRE')
    if digest(pre['parent_source']) != pre['parent_sha']:
        raise ValueError('FROZEN_PARENT_SOURCE_MISMATCH')
    name = 'TECHNICAL_REPAIR' if repair else 'PROPOSAL_SINGLE' if len(pre['slot_order']) == 1 else 'PROPOSAL_BATCH'
    instruction = (AUTHORITY / 'prompts' / (name + '.txt')).read_text()
    if repair:
        instruction = instruction.replace('<FAILED_SLOT>', failed_slot)
    elif len(pre['slot_order']) == 1:
        instruction = instruction.replace('<SLOT_ID>', pre['slot_order'][0])
    views = {}
    for sid in pre['slot_order']:
        slot = pre['slots'][sid]
        views[sid] = dict(SLOT_ID=sid, operation=slot['operator'], planned_scope=slot['planned_scope'],
                          requires_FULL=slot['host_full'], menu=_CONTRAST.compact_menu(slot['menu']),
                          allowed_evidence_ids=[c['id'] for c in slot['memory']['compact_cases']])
        if not repair:
            views[sid]['operation_instructions'] = _operation_instructions(slot, bool(slot['donor_sha']))
    ordinary = dict(current_parent_mean=pre['quality_before']['parent'], current_elite_mean=pre['quality_before']['elite'])
    payload = dict(TASK_AND_API=io.task_description, CURRENT_PROGRAM=pre['parent_source'],
                   EVALUATION_CONTEXT=ordinary, SLOT_CONTRACTS=views,
                   EDIT_CASES=pre['memory']['compact_cases'])
    if not repair and pre.get('donor_source'):
        payload['MIX_DONOR_PROGRAM'] = pre['donor_source']
    if repair:
        payload.update(FAILED_SLOT=failed_slot, ACTUAL_TECHNICAL_ERROR=copy.deepcopy(error),
                       ORIGINAL_SLOT_RESPONSE=failed_output)
    # One-shot json serialization never interprets source-code placeholder text.
    return [dict(role='system', content='Generate executable algorithms under the supplied task API and frozen edit contract.'),
            dict(role='user', content=instruction + '\nFROZEN_INPUT\n' + json.dumps(payload, ensure_ascii=False, sort_keys=True))]
