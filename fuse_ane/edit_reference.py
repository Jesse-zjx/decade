"""Pure reference primitives, NOT a candidate evaluator or a deployed search method.
All evaluations supplied by the caller must come from the trusted runner.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from math import fsum, isfinite, sqrt
from typing import Sequence


def digest(value: bytes) -> str:
    return sha256(value).hexdigest()


@dataclass(frozen=True)
class Replacement:
    start: int
    end: int
    before: bytes
    after: bytes


def apply_exact(parent: bytes, replacements: Sequence[Replacement]) -> bytes:
    """Apply exact half-open byte replacements; reject ambiguity and overlap."""
    ordered = sorted(replacements, key=lambda r: (r.start, r.end))
    if not ordered:
        raise ValueError('NO_REPLACEMENT')
    result: list[bytes] = []
    cursor = 0
    for r in ordered:
        if not (0 <= r.start < r.end <= len(parent)) or r.start < cursor:
            raise ValueError('INVALID_OR_OVERLAPPING_RANGE')
        if parent[r.start:r.end] != r.before:
            raise ValueError('BEFORE_BYTES_MISMATCH')
        result.extend((parent[cursor:r.start], r.after))
        cursor = r.end
    result.append(parent[cursor:])
    return b''.join(result)


def paired_effect(before: Sequence[float], after: Sequence[float], scale: float) -> dict:
    if len(before) < 2 or len(before) != len(after):
        raise ValueError('UNPAIRED_OR_INSUFFICIENT_MEASUREMENTS')
    if not isfinite(scale) or scale <= 0:
        raise ValueError('INVALID_FIXED_SCALE')
    if any(not isfinite(x) or x < 0 for x in (*before, *after)):
        raise ValueError('INVALID_MEASUREMENT')
    ds = [100.0 * (p-q) / scale for p,q in zip(before, after)]
    n = len(ds)
    mean = fsum(ds)/n
    se = sqrt(fsum((x-mean)**2 for x in ds)/(n*(n-1)))
    ordered = sorted(ds)
    median = (ordered[(n-1)//2]+ordered[n//2])/2
    return {'n': n, 'credit_pp': mean, 'median_pp': median,
            'se_pp_conditional': se,
            'positive_instances': sum(x>1e-9 for x in ds),
            'negative_instances': sum(x< -1e-9 for x in ds),
            'zero_instances_at_tolerance': sum(abs(x)<=1e-9 for x in ds),
            'before_mean': fsum(before)/n, 'after_mean': fsum(after)/n,
            'raw_instance_differences_pp': ds}


def local_record(*, parent: bytes, child: bytes, replacements: Sequence[Replacement],
                 scope: str, actual_mode: str,
                 parent_ids: Sequence[str], child_ids: Sequence[str],
                 parent_losses: Sequence[float], child_losses: Sequence[float],
                 parent_runtime: str, child_runtime: str,
                 panel_sha: str, event_id: str, scale: float) -> dict:
    """Verify the supplied exact edit & measurement binding, not executability.

    The caller must also certify that all ranges belong to ONE registered B/C
    unit and both full programs were valid under the trusted runtime.
    """
    if actual_mode != 'REPLACE' or scope not in ('B', 'C'):
        raise ValueError('NOT_A_CONTROLLED_LOCAL_EDIT')
    if parent == child:
        raise ValueError('DUPLICATE_SOURCE')
    if not panel_sha or not parent_runtime or parent_runtime != child_runtime:
        raise ValueError('RUNTIME_OR_PANEL_BINDING_MISMATCH')
    if (list(parent_ids) != list(child_ids)
        or len(parent_ids) != len(parent_losses)
        or len(child_ids) != len(child_losses)
        or len(set(parent_ids)) != len(parent_ids)):
        raise ValueError('INSTANCE_BINDING_MISMATCH')
    if apply_exact(parent, replacements) != child:
        raise ValueError('OUTSIDE_CHANGE_OR_UNRECORDED_EDIT')
    effect = paired_effect(parent_losses, child_losses, scale)
    return {'event_id':event_id, 'type':'LOCAL_PAIRED', 'scope':scope,
            'parent_sha':digest(parent), 'child_sha':digest(child),
            'panel_sha':panel_sha, 'runtime_sha':parent_runtime,
            'fixed_scale':scale, 'effect':effect,
            'replacements': [dict(start=r.start,end=r.end,
                                  before_sha=digest(r.before),after_sha=digest(r.after))
                             for r in replacements],
            'causal_generalization_claim':False}


def context_status(record: dict, current: bytes, *, historical_unique_match: bool=False) -> str:
    h=digest(current)
    if h == record['parent_sha']:
        return 'CURRENT_EXACT_BEFORE'
    if h == record['child_sha']:
        return 'CURRENT_EXACT_AFTER'
    return 'HISTORICAL_LINEAGE' if historical_unique_match else 'UNAVAILABLE'


def effect_views(code_cards: list[dict], effects: list[dict], arm: str) -> dict:
    """Input cards are pre-sanitized code/provenance only. N has no fake labels."""
    if arm not in ('F', 'N') or len(code_cards) != len(effects):
        raise ValueError('INVALID_VIEW_INPUT')
    forbidden = {'effect', 'credit_pp', 'before_mean', 'after_mean',
                 'elite_gain', 'reward', 'scores'}
    if any(forbidden & set(c) for c in code_cards):
        raise ValueError('UNSANITIZED_CODE_CARD')
    if arm=='F':
        return {'cases':[dict(c,effect=e) for c,e in zip(code_cards,effects)],
                'unbound_outcomes':[]}
    # The control retains code examples but has no per-edit numeric annotation.
    # Current parent/archive quality is supplied separately by the shared runner.
    return {'cases':[dict(c, effect=None) for c in code_cards],
            'unbound_outcomes':[]}
