"""FUSE-RNE finite-state reference. Pure controller: no model or evaluator.

The credit is observed progress of an entire proposal+repair episode, not
counterfactual credit of a code block. All controls use the same generator.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal
import math

Scope = Literal['B','C','G']
MODES = ('B','C','G')

@dataclass(frozen=True)
class Observation:
    episode_id: str
    planned_scope: Scope
    complete: bool
    valid: bool
    novel: bool
    elite_gain_pp: float | None
    technical_failure: bool = False
    realized_scope: str | None = None

    def check(self) -> None:
        if not self.episode_id or self.planned_scope not in MODES:
            raise ValueError('invalid episode binding')
        if not self.complete:
            raise ValueError('unresolved infrastructure episode must not update the policy')
        if self.valid and (self.elite_gain_pp is None or not math.isfinite(self.elite_gain_pp)):
            raise ValueError('valid outcome requires finite measured progress')
        if self.valid and self.technical_failure:
            raise ValueError('successful repair is valid, not a final technical failure')

@dataclass
class ReactiveController:
    scope: Scope = 'B'
    no_progress: int = 0
    global_calls: int = 0
    completed: int = 0
    threshold_pp: float = 0.01
    failure_patience: int = 2
    last_episode_id: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in MODES or self.threshold_pp <= 0 or self.failure_patience < 1:
            raise ValueError('invalid configuration')

    def next(self, available: tuple[str,...] = ('B','C','G')) -> dict:
        """Resolve empty regions without sending a request or incrementing a step.

        The adapter must freeze this returned action before generation. Calling
        next twice without observe is idempotent. A hash/version change invalidates
        the adapter's region binding, not this task-level controller state.
        """
        if 'G' not in available:
            raise ValueError('full-program generation must always be available')
        while self.scope not in available:
            self.scope = MODES[min(2, MODES.index(self.scope)+1)]
            self.no_progress = 0
        kind = ('ALT' if self.global_calls % 2 == 0 else 'MIX') if self.scope == 'G' else 'REFINE'
        return dict(scope=self.scope,operator=kind,completed=self.completed,global_calls=self.global_calls)

    def observe(self, obs: Observation) -> dict:
        obs.check()
        if obs.episode_id == self.last_episode_id:
            raise ValueError('duplicate commit; reconcile in adapter, do not update twice')
        if obs.planned_scope != self.scope:
            raise ValueError('outcome is not bound to the dispatched stage')
        before=asdict(self)
        progressed=bool(obs.valid and obs.novel and obs.elite_gain_pp >= self.threshold_pp)
        if self.scope == 'G':
            self.global_calls += 1
        if progressed:
            # Actual FULL escaping a local request is legal, but is not evidence
            # that the requested local region itself caused the improvement.
            self.scope='B';self.no_progress=0
        elif obs.technical_failure:
            self.scope=MODES[min(2,MODES.index(self.scope)+1)];self.no_progress=0
        else:
            self.no_progress += 1
            if self.no_progress >= self.failure_patience:
                self.scope=MODES[min(2,MODES.index(self.scope)+1)];self.no_progress=0
        self.completed += 1;self.last_episode_id=obs.episode_id
        return dict(before=before,after=asdict(self),material_progress=progressed)

@dataclass
class FixedController:
    """Same-capability control S: a frozen B,B,C,C,G,G cycle.

    Outcome quality still updates the common archive/parent in the adapter;
    only stage scheduling is independent of episode progress.
    """
    completed: int = 0
    global_calls: int = 0
    last_episode_id: str | None = None
    pending: dict | None = None

    def next(self, available: tuple[str,...] = ('B','C','G')) -> dict:
        if 'G' not in available: raise ValueError('G missing')
        target=('B','B','C','C','G','G')[self.completed % 6]
        while target not in available: target=MODES[min(2,MODES.index(target)+1)]
        kind=('ALT' if self.global_calls % 2 == 0 else 'MIX') if target=='G' else 'REFINE'
        self.pending=dict(scope=target,operator=kind,completed=self.completed,global_calls=self.global_calls)
        return dict(self.pending)

    def observe(self, obs: Observation) -> dict:
        obs.check()
        if self.pending is None or obs.planned_scope != self.pending['scope']:
            raise ValueError('dispatch binding mismatch')
        if obs.episode_id == self.last_episode_id:raise ValueError('duplicate commit')
        if obs.planned_scope=='G':self.global_calls += 1
        self.completed += 1;self.last_episode_id=obs.episode_id;self.pending=None
        return dict(completed=self.completed,global_calls=self.global_calls)
