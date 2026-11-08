"""Pure source-linked contrast compiler; never imports or executes candidate code.

Byte ranges are UTF-8 half-open ranges. Human view uses zero-based character
spans within a cited current region, and complete old/new strings for each
changed span. Kept fields are bounded, not silently truncated.
"""
from __future__ import annotations
import copy
import difflib
import hashlib
import json
from typing import Any


def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def delta(current: str, other: str) -> dict[str, Any]:
    """Lossless character patch from current to a concrete alternate version."""
    ops=[]
    a=current.splitlines(True); b=other.splitlines(True)
    ap=[0]; bp=[0]
    for line in a: ap.append(ap[-1]+len(line))
    for line in b: bp.append(bp[-1]+len(line))
    for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag!='equal':
            ops.append({'start':ap[i],'end':ap[j],'old':current[ap[i]:ap[j]],'new':other[bp[k]:bp[l]]})
    obj={'index_unit':'unicode_character','base_sha256':sha(current),'other_sha256':sha(other),'operations':ops}
    full={'index_unit':'unicode_character','base_sha256':sha(current),'other_sha256':sha(other),
          'whole_region_replacement':other}
    if len(json.dumps(full,ensure_ascii=False))<len(json.dumps(obj,ensure_ascii=False)): obj=full
    if apply_delta(current,obj)!=other:
        raise ValueError('INTERNAL_PATCH_ROUNDTRIP')
    return obj


def apply_delta(current: str, patch: dict[str, Any]) -> str:
    if patch.get('base_sha256')!=sha(current):
        raise ValueError('PATCH_BASE_MISMATCH')
    if 'whole_region_replacement' in patch:
        result=patch['whole_region_replacement']
        if sha(result)!=patch['other_sha256']: raise ValueError('PATCH_RESULT_MISMATCH')
        return result
    cursor=0; pieces=[]
    for op in patch['operations']:
        i,j=op['start'],op['end']
        if not isinstance(i,int) or not isinstance(j,int) or not cursor<=i<=j<=len(current):
            raise ValueError('PATCH_RANGE')
        if current[i:j]!=op['old']:
            raise ValueError('PATCH_OLD_MISMATCH')
        pieces += [current[cursor:i],op['new']]; cursor=j
    result=''.join(pieces)+current[cursor:]
    if sha(result)!=patch['other_sha256']:
        raise ValueError('PATCH_RESULT_MISMATCH')
    return result


def compact_case(parent: str, menu: dict, card: dict, record: dict,
                 show_effect: bool) -> dict:
    """All alternate bytes come from a measured record, never from an LLM summary."""
    target=card['target']; region=menu[target]; side=card['matched_side']
    if side not in ('before','after'):
        raise ValueError('UNSUPPORTED_MATCH_SIDE')
    other_side='after' if side=='before' else 'before'
    parts=record['parts']; ranges=region['byte_ranges']
    if len(parts)!=len(ranges):
        raise ValueError('PART_RANGE_MISMATCH')
    encoded=parent.encode(); patches=[]
    for (start,end),part in zip(ranges,parts):
        current=encoded[start:end].decode('utf-8')
        if current!=part[side]:
            raise ValueError('CURRENT_REGION_NOT_EXACT_MATCH')
        patch=delta(current,part[other_side])
        patches.append({'current_parent_utf8_range':[start,end], 'alternate_patch':patch})
    effect=record['effect']
    summary={k:effect[k] for k in ('credit_pp','se_pp_conditional','n','positive_instances',
                                  'negative_instances','zero_instances_at_tolerance')}
    return {'id':card['id'],'target':target,'context_status':card['context_status'],
            'current_implementation_is':side,
            'comparison_orientation':'measured BEFORE -> AFTER; positive credit means improvement',
            'patches_current_to_alternate':patches,
            'observed_comparison':summary if show_effect else None,
            'current_effect_status':'UNKNOWN' if card['context_status']=='HISTORICAL_LINEAGE' else 'OBSERVED_PAIR',
            'warning':'An observed substitution effect is not the expected benefit of another edit.'}


def compact_menu(menu: dict) -> dict:
    """Code appears once in the parent. Both byte and original line spans remain."""
    result={}
    for uid,r in menu.items():
        result[uid]={'scope':r['scope']}
        if uid!='GLOBAL':
            result[uid].update(unit_path=r.get('unit_path'),line_spans=r.get('spans'),
                               parent_utf8_ranges=r['byte_ranges'])
    return result


def compile_view(snapshot: dict, show_effect: bool, memory_char_cap: int=8192) -> dict:
    """Admission is identical across visibility arms, budgeted using rich card size."""
    parent=snapshot['parent_source'];menu=snapshot['pre']['menu'];byid=snapshot['evidence']
    kept=[]; omitted=[];budget_used=0
    for card in snapshot['pre']['memory']['cases']:
        full=compact_case(parent,menu,card,byid[card['evidence_id']],True)
        n=len(json.dumps(full,ensure_ascii=False,separators=(',',':')))
        if budget_used+n>memory_char_cap:
            omitted.append({'id':card['id'],'reason':'WHOLE_CONTRAST_OVER_MEMORY_CAP'})
            continue
        if not show_effect:full['observed_comparison']=None
        kept.append(full);budget_used+=n
    return {'parent_source':parent,'menu':compact_menu(menu),'cases':kept,
            'omitted':omitted,'rich_card_char_charge':budget_used,
            'ordinary_feedback':{'parent_mean':snapshot['pre']['quality_before']['parent'],
                                 'elite_mean':snapshot['pre']['quality_before']['elite'],
                                 'historical_edit_scores':'NOT_TRANSMITTED_IN_ANY_CELL'}}


def legacy_view(snapshot: dict, show_effect: bool) -> dict:
    """Legacy serialization, not legacy policy replication; no numeric side channel."""
    pre=copy.deepcopy(snapshot['pre']); byid=snapshot['evidence']
    for card in pre['memory']['cases']:
        record=byid[card['evidence_id']]
        card['effect']=copy.deepcopy(record['effect']) if show_effect else None
        card['target_credit']=(copy.deepcopy(record['effect']) if show_effect and
                               card['context_status']!='HISTORICAL_LINEAGE' else None)
    return {'parent_source':snapshot['parent_source'],'menu':pre['menu'],
            'memory':pre['memory'],
            'ordinary_feedback':{'parent_mean':pre['quality_before']['parent'],
                                 'elite_mean':pre['quality_before']['elite'],
                                 'historical_edit_scores':'NOT_TRANSMITTED_IN_ANY_CELL'}}
