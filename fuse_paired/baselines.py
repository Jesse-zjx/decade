"""V2 ledger/selection wrapper around the frozen complete EoH/HiFo engines.

Native prompts, parent/population policy, duplicate generation, rounding,
insight/navigation, auxiliary requests and due maintenance remain unchanged.
Only fresh I/O, 128-position stopping and raw Search prefix selection differ.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
from pathlib import Path
import threading
from unittest.mock import patch

from fuse_addon.evaluation import digest, encoded, freeze, read
from fuse_credit import baselines as native
from .full_runtime import strategy_cost
from .runtime import NN_ANCHOR_SOURCE, STUDY_ID

_ORIGINAL_BASELINE_IO = native.BaselineIO
_ORIGINAL_SOURCE_MANIFEST = native.source_manifest
_NATIVE_SCOPE_LOCK = threading.RLock()


def source_manifest(arm):
    original = _ORIGINAL_SOURCE_MANIFEST(arm)
    result = copy.deepcopy(original)
    result['v1_adapter_manifest_sha256'] = digest(encoded(original))
    result['study_id'] = STUDY_ID
    result['v2_policy_changes'] = []
    result['v2_wrapper_changes'] = [
        'fresh V2 phase/group streams and shared frozen group NN reference',
        '128 main generation positions including native initialization/duplicates',
        'raw Search best complete program per requested prefix, then locked independent Val/Test',
        'full group reference charged to each standalone strategy; physical reference work counted once',
        'every confirmed main or auxiliary response identity checked; original requests reused on recovery',
    ]
    result['adaptations'] = [
        ('all valid complete source artifacts enter Search archive; raw Search best is locked before Val/Test'
         if 'final selection is external Val' in item else item)
        for item in result['adaptations']]
    return result


def _selection(search):
    if search.state['programs']:
        best = min(search.state['programs'].values(),
                   key=lambda row: (row['mean'], row['first_position'], row['hash']))
        return dict(source_hash=best['hash'], mean=best['mean'], complete=True,
                    first_position=best['first_position'],
                    fallback=best.get('origin') == 'all_initialization_failed_reference')
    h = digest(NN_ANCHOR_SOURCE)
    quality = search.state['qualities'][h]
    return dict(source_hash=h, mean=quality['mean'], complete=quality['complete'],
                first_position=0, fallback=True)


class BaselineIO(_ORIGINAL_BASELINE_IO):
    """Same native generation/auxiliary methods, V2-only prefix/final ledgers."""
    def write_prefix(self, budget, *, status='PREFIX_COMPLETE'):
        path = self.search.root/'prefixes'/f'{budget:03d}'/'PREFIX.json'
        selection = _selection(self.search)
        if path.exists():
            result = read(path)
            if (result['archive_hashes'] != sorted(self.search.state['programs']) or
                    result['selection'] != selection or result['completed_positions'] != self.position):
                raise RuntimeError('REPLAY_PREFIX_ARCHIVE_OR_SELECTION_CONFLICT')
            self.prefix_results[str(budget)] = result
            return result
        self.search.checkpoint(f'prefix-{budget:03d}')
        cost = strategy_cost(self.search)
        result = dict(budget=budget, completed_positions=self.position, status=status,
            archive_hashes=sorted(self.search.state['programs']),
            fallback_program_hash=selection['source_hash'] if selection['fallback'] else None,
            state_path=str(self.search.root/'states'/f'prefix-{budget:03d}.json'),
            cost=cost, own_physical_cost=cost['own_physical_cost'],
            reference_cost=cost['reference_cost'], reference_charged_in_full=True,
            selection=selection, search_best_hash=selection['source_hash'], search_best_mean=selection['mean'],
            selection_rule='raw Search mean,first_position,source_hash', search_locked=True)
        freeze(path, result)
        self.prefix_results[str(budget)] = result
        return result

    def finish(self, population, native_state=None, *, status='NATIVE_BUDGET_COMPLETE'):
        from fuse_credit.state import register_program
        if not self.search.state['programs']:
            h = digest(NN_ANCHOR_SOURCE)
            register_program(self.search.state, NN_ANCHOR_SOURCE, self.search.state['qualities'][h],
                             [], 'all_initialization_failed_reference', 0)
        self.end_batch(population, native_state)
        for point in self.prefixes:
            if str(point) not in self.prefix_results:
                if self.position < point:
                    self.write_prefix(point, status='NATIVE_INITIALIZATION_FAILED_FALLBACK')
                else:
                    raise RuntimeError('NATIVE_PREFIX_WAS_NOT_CAPTURED')
        self.search.checkpoint('FINAL')
        path = self.search.root/'SEARCH_RESULT.json'
        if path.exists():
            return read(path)
        selection = _selection(self.search)
        cost = strategy_cost(self.search)
        result = dict(study_id=STUDY_ID, phase=self.search.phase, group=self.search.group, arm=self.search.arm,
            search_best_hash=selection['source_hash'], search_best_mean=selection['mean'],
            selection=selection, selection_rule='raw Search mean,first_position,source_hash',
            planned_positions=self.budget, completed_positions=self.position,
            unattempted_positions=self.budget-self.position, termination=status,
            init_completed=self.search.state['init_completed'], noninit_completed=self.search.state['noninit_completed'],
            cost=cost, own_physical_cost=cost['own_physical_cost'], reference_cost=cost['reference_cost'],
            reference_charged_in_full=True, historical_common_cost=None,
            state_path=str(self.search.root/'states/FINAL.json'),
            program_count=len(self.search.state['programs']), credit_count=0,
            failure_count=len(self.search.state['failures']), prefixes=self.prefix_results, search_locked=True)
        return freeze(path, result)


@contextmanager
def native_io_scope(io):
    """Bound module adaptation; native runners must retain their own process.

The yielded BaselineIO is also usable by the root's complete ReEvo wrapper.
The original V1 files and implementations are never edited.
"""
    from fuse_credit import search as old_search
    original_cost = old_search.cost_snapshot
    def scoped_cost(root):
        return strategy_cost(io) if Path(root).resolve() == io.root.resolve() else original_cost(root)
    with _NATIVE_SCOPE_LOCK, ExitStack() as stack:
        stack.enter_context(patch.object(old_search, 'cost_snapshot', scoped_cost))
        stack.enter_context(patch.object(native, 'BaselineIO', BaselineIO))
        stack.enter_context(patch.object(native, 'source_manifest', source_manifest))
        yield BaselineIO


def run_native(io, budget=128, prefixes=(128,)):
    if io.arm not in ('E', 'H'):
        raise ValueError('E_OR_H_REQUIRED')
    path = io.root/'SEARCH_RESULT.json'
    if path.exists():
        return read(path)
    # Mature native loops rebuild their own population/insight state. Exact
    # persisted logical requests and evaluation roles supply the old facts.
    io.replay_native_from_start()
    with native_io_scope(io):
        return native.run_native_baseline(io, budget=budget, prefixes=prefixes)
