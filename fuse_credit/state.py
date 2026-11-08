"""Complete program archive and current-background paired credit; no execution."""
from __future__ import annotations
import copy
import hashlib
import json
import random
import statistics
from pathlib import Path
from .factors import factors, compatible_reference, natural_pair
from .controller import credit_key
from fuse_addon.evaluation import freeze, freeze_bytes, digest, encoded

REFERENCE_SOURCE = '''import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    return min((int(x) for x in unvisited_nodes), key=lambda x: (float(distance_matrix[current_node, x]), x))
'''


def fresh_state(context):
    return dict(schema='credit-action-state-v1', context=context, programs={}, qualities={},
                credits={}, credit_history=[], occurrences={}, occurrence_counter=0, history=[], parent_visits={}, target_visits={}, op_attempts={},
                op_valid={}, completion_history={}, cf_last_processed={}, cf_events=[],
                noninit_completed=0, init_completed=0, archive_counter=0, scale=None,
                reference_hash=digest(REFERENCE_SOURCE), controllers={}, failures=[])


def register_program(state, source, quality, parents, origin, position, algorithm='', *, admit=True):
    if not quality.get('complete') or quality.get('mean') is None:
        raise ValueError('ONLY_COMPLETE_QUALITY_ADMITTED')
    h = digest(source)
    state['qualities'][h] = quality
    if not admit:
        return h
    # A new actual occurrence has its own predecessor identities even when its
    # complete source matches an earlier archived program. No source-hash cycle.
    parent_occurrences = [state['programs'][p]['occurrence_id'] for p in parents]
    occurrence_id = f"occurrence-{state['occurrence_counter']:06d}"
    state['occurrence_counter'] += 1
    state['occurrences'][occurrence_id] = dict(id=occurrence_id, source_hash=h,
        parents=parent_occurrences, origin=origin, position=position, quality_available=True)
    if h not in state['programs']:
        state['programs'][h] = dict(hash=h, source=source, mean=quality['mean'],
            factors=factors(source, state['context']), parents=list(parents),
            origin=origin, first_position=position, first_index=state['archive_counter'],
            algorithm=algorithm, quality_available=True, occurrence_id=occurrence_id)
        state['archive_counter'] += 1
    else:
        state['programs'][h].update(parents=list(parents),occurrence_id=occurrence_id,
            latest_origin=origin, latest_position=position, algorithm=algorithm)
    # Retain known utility only if its actual reference is still the nearest
    # distinct compatible predecessor in this current occurrence lineage.
    for f in state['programs'][h]['factors']:
        key=credit_key(h,f)
        old=state['credits'].get(key)
        if old is None:continue
        ref=compatible_reference(f,ancestors(state,h),current_source=source)
        if ref is None or digest(ref['replacement_source']) != old['reference_hash']:
            state['credit_history'].append(dict(old,invalidated_by_occurrence=occurrence_id))
            del state['credits'][key]
    return h


def active_pool(state, rng_seed):
    """Algorithm R on the current nonelite archive, independent stream each slate.

    Each pool refresh is a new exact uniform four-item reservoir over all current
    nonelites in admission order. No score is used in reservoir admission.
    """
    ordered = sorted(state['programs'].values(), key=lambda p: (p['mean'], p['hash']))
    elites = ordered[:8]
    rest = sorted(ordered[8:], key=lambda p: (p['first_index'], p['hash']))
    rng = random.Random(rng_seed)
    reservoir = []
    for i, p in enumerate(rest):
        if i < 4:
            reservoir.append(p)
        else:
            j = rng.randrange(i + 1)
            if j < 4:
                reservoir[j] = p
    return sorted(elites + reservoir, key=lambda p: (p['mean'], p['hash']))


def ancestors(state, program_hash):
    occurrence = state['occurrences'][state['programs'][program_hash]['occurrence_id']]
    frontier = list(occurrence['parents'])
    seen = {occurrence['id']}
    result = []
    depth = 1
    while frontier:
        following = []
        layer=[]
        for oid in sorted(set(frontier)):
            if oid in seen:continue
            seen.add(oid)
            o=state['occurrences'][oid]
            h=o['source_hash'];p=state['programs'][h]
            layer.append(dict(source=p['source'],program_hash=h,occurrence_id=oid,
                              quality_available=True,depth=depth))
            following.extend(o['parents'])
        result.extend(sorted(layer,key=lambda x:(x['program_hash'],x['occurrence_id'])))
        frontier=following;depth+=1
    return result


def matched_credit(state, parent_hash, factor, reference_hash, origin, *, evidence=None):
    p = state['qualities'][parent_hash]
    r = state['qualities'][reference_hash]
    if not p['complete'] or not r['complete']:
        return None
    if p.get('execution_binding') != r.get('execution_binding'):
        raise RuntimeError('CREDIT_ENDPOINT_EXECUTION_CONFLICT')
    if p['panel_sha256'] != r['panel_sha256']:
        raise RuntimeError('CREDIT_ENDPOINT_PANEL_CONFLICT')
    pi = [x['instance_id'] for x in p['rows']]
    ri = [x['instance_id'] for x in r['rows']]
    if pi != ri:
        raise RuntimeError('CREDIT_ENDPOINT_INSTANCE_CONFLICT')
    values = [100 * (b['length']-a['length']) / state['scale']
              for a, b in zip(p['rows'], r['rows'])]
    c = dict(mean=statistics.mean(values), se=statistics.stdev(values)/len(values)**.5
             if len(values)>1 else None, values=values, parent_hash=parent_hash,
             factor_version=factor['version'], background=factor['background'],
             reference_hash=reference_hash, origin=origin,
             occurrence_id=state['programs'][parent_hash]['occurrence_id'],
             panel_sha256=p['panel_sha256'], context=state['context'], evidence=evidence)
    key = credit_key(parent_hash, factor)
    # Repeated exact current endpoint facts agree; no historical utility migration.
    if key in state['credits']:
        old = state['credits'][key]
        if old['reference_hash'] != reference_hash or old['values'] != values:
            raise RuntimeError('CURRENT_CREDIT_REFERENCE_CONFLICT')
        return old
    state['credits'][key] = c
    return c


def admit_natural_credit(state, parent_hash, child_hash):
    if parent_hash == child_hash:
        return None
    parent, child = state['programs'][parent_hash], state['programs'][child_hash]
    certificate = natural_pair(parent['source'], child['source'], state['context'])
    if certificate is None:
        return None
    nearest = compatible_reference(certificate['child_factor'], ancestors(state, child_hash), current_source=child['source'])
    if nearest is None or digest(nearest['replacement_source']) != parent_hash:
        return None
    # Only the lineage-defined nearest reference can supply this current credit.
    return matched_credit(state, child_hash, certificate['child_factor'], parent_hash,
                          'natural_exact_single_function_pair', evidence=certificate)


def cf_candidate(state, pool, rng_seed):
    choices = []
    for p in pool:
        refs = []
        lineage = ancestors(state, p['hash'])
        for f in p['factors']:
            if credit_key(p['hash'], f) in state['credits']:
                continue
            ref = compatible_reference(f, lineage, current_source=p['source'])
            if ref is not None:
                refs.append((f, ref))
        if refs:
            choices.append((state['cf_last_processed'].get(p['hash'], -1),
                            p['first_index'], p['hash'], refs))
    if not choices:
        return None
    _, _, h, refs = min(choices, key=lambda x:x[:3])
    f, ref = random.Random(rng_seed).choice(refs)
    return dict(program_hash=h, factor=f, reference=ref)


def freeze_state(path, state):
    return freeze(path, state)
