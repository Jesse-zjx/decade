"""Finite token/position boundary; no transport or candidate execution of its own.

A response crossing a threshold records the *then available* incumbent before
its candidate is evaluated. The eventual terminal candidate remains separately
available in the final policy result. Snapshots must use a logical replay cost
frontier: later durable receipts must not enter an earlier replay boundary.
"""
from __future__ import annotations
import ast
import copy
import math
from pathlib import Path

from fuse_addon.evaluation import digest, encoded, freeze, freeze_bytes, read
from fuse_addon.response_parser import extract_response
from fuse_credit.baselines import MODULE_FORMAT, _native_parents

TOKEN_PREFIXES = (100000, 200000, 300000, 400000)
POSITION_CAP = 256
CVRP_MODULE_FORMAT = (
    "\nExecution/output contract: select_next_node(current_node, depot, unvisited_nodes, "
    "rest_capacity, demands, distance_matrix) returns an integer node ID, never bool. "
    "Choose an unserved customer from unvisited_nodes with demand no larger than rest_capacity; "
    "Away from depot you may return depot to close the current route; when none is feasible return depot. At depot, return a feasible customer if one exists. "
    "Inputs are integer node IDs, remaining capacity, NumPy demand and distance arrays. "
    "Do not mutate inputs. Output exactly one complete Python module in one "
    "```python code block, including all imports and helper definitions it needs. "
    "All top-level functions execute in one module namespace. You may add, remove, "
    "merge or split helpers and change internal composition. Do not output fragments "
    "or multiple alternatives. No external I/O, files or network are available."
)


EXECUTION_CAPABILITIES = (
    " Ordinary Python imports, helpers, classes, closures, NumPy, instance-local state and clocks are allowed. "
    "The full current-instance API arrays are provided. Private container scratch files are available. "
    "No other instance, hidden panel, host files or network are available. "
    "An external trusted controller constructs and scores the route and enforces immutable deadlines."
)

def module_format(search):
    task = str(search.context.get('task', 'tsp')).lower()
    if task == 'tsp':
        return MODULE_FORMAT.replace("No external I/O, files or network are available.", "") + EXECUTION_CAPABILITIES
    if task == 'cvrp':
        return CVRP_MODULE_FORMAT.replace("No external I/O, files or network are available.", "") + EXECUTION_CAPABILITIES
    raise ValueError('UNKNOWN_NATIVE_TASK')


class DispatchStop(BaseException):
    """Escape native broad Exception handlers without generating replacement."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def token_count(cost):
    value = cost.get('tokens')
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or cost.get('tokens_missing', 0)):
        return None
    return value


def selection(search):
    programs = search.state.get('programs', {})
    if programs:
        row = min(programs.values(), key=lambda x: (x['mean'], x.get('first_position', 0), x['hash']))
        return dict(source_hash=row['hash'], mean=row['mean'], complete=True,
                    first_position=row.get('first_position', 0), fallback=False)
    reference = search.reference
    return dict(source_hash=reference['source_hash'], mean=reference['mean'],
                complete=reference['quality']['complete'], first_position=0, fallback=True)


def admit(search, source, quality, parents, origin, position, algorithm=''):
    """Use runtime's V3 admission; never create legacy factor/controller state."""
    if hasattr(search, 'register_program'):
        return search.register_program(source, quality, parents, origin, position, algorithm)
    raise RuntimeError('V3_RUNTIME_REGISTER_PROGRAM_REQUIRED')


class TokenBoundary:
    def __init__(self, search, *, target=None, position_cap=POSITION_CAP,
                 initial_positions=0, continuation_limit=None):
        self.search = search
        self.target = target if target is not None else (300000 if search.arm in ('J', 'A', 'W', 'I30') else 400000)
        self.budget = int(position_cap)
        self.position = initial_positions
        self.initial_positions = initial_positions
        self.continuation_limit = continuation_limit
        self.prefixes = tuple(p for p in TOKEN_PREFIXES if p <= self.target)
        self.prefix_results = {}
        self.event_counter = 0
        self.last_individual = self.last_receipt = None
        self._pending_position = None
        self.pending_prefix = None
        search.state.setdefault('native_auxiliary', [])
        search.state.setdefault('failures', [])

    def ensure_reference(self):
        reference = self.search.reference
        if not reference['quality'].get('complete'):
            raise RuntimeError('PUBLIC_REFERENCE_INVALID')
        self.search.save_program(reference['source'])
        self.search.state['qualities'][reference['source_hash']] = copy.deepcopy(reference['quality'])
        self.search.state['scale'] = reference['mean']
        return reference['quality']

    def before_dispatch(self, *, generation=False):
        cost = self.search.cost_snapshot()
        tokens = token_count(cost)
        if tokens is None:
            raise DispatchStop('UNKNOWN_USAGE')
        if tokens >= self.target:
            raise DispatchStop('TOKEN_TARGET_REACHED')
        if self.position >= self.budget:
            raise DispatchStop('POSITION_CAP_REACHED')
        # At the structural pause no further native maintenance is dispatched.
        if (self.continuation_limit is not None and
                self.position - self.initial_positions >= self.continuation_limit):
            raise DispatchStop('PAUSED_EARLY_CHECK')

    def observe(self, kind, **detail):
        self.event_counter += 1
        cost = self.search.cost_snapshot()
        chosen = selection(self.search)
        row = dict(event=self.event_counter, boundary=kind, position=self.position,
                   pending_position=self._pending_position, selection=chosen,
                   archive_hashes=sorted(self.search.state.get('programs', {})),
                   cost=cost, **detail)
        path = self.search.root/'budget_boundaries'/f'{self.event_counter:06d}.json'
        # On logical replay, keep original wall timings; compare only scientific
        # state/cost accounting. Runtime restores the exact event frontier cost.
        if path.exists():
            prior = read(path)
            for key in ('boundary', 'position', 'pending_position', 'selection', 'archive_hashes'):
                if prior.get(key) != row.get(key):
                    raise RuntimeError('BUDGET_BOUNDARY_REPLAY_CONFLICT: '+key)
            for key in ('tokens', 'tokens_known', 'tokens_missing', 'http'):
                if prior['cost'].get(key) != cost.get(key):
                    raise RuntimeError('BUDGET_COST_REPLAY_CONFLICT: '+key)
            row = prior
        else:
            freeze(path, row)
        tokens = token_count(row['cost'])
        if tokens is not None:
            for threshold in self.prefixes:
                if threshold not in self.prefix_results and tokens >= threshold:
                    prefix = dict(threshold_tokens=threshold, status='REACHED',
                        reached=True, first_boundary_event=self.event_counter,
                        boundary=kind, completed_positions=self.position,
                        pending_position=self._pending_position, selection=row['selection'],
                        search_best_hash=row['selection']['source_hash'],
                        search_best_mean=row['selection']['mean'], cost=row['cost'],
                        archive_hashes=row['archive_hashes'], search_locked=True,
                        actual_tokens=tokens, overshoot_tokens=tokens-threshold)
                    target = self.search.root/'prefixes'/str(threshold)/'PREFIX.json'
                    if target.exists() and read(target) != prefix:
                        raise RuntimeError('TOKEN_PREFIX_REPLAY_CONFLICT')
                    freeze(target, prefix)
                    self.prefix_results[threshold] = prefix
        return row

    def inherit_prefixes(self, shared):
        for raw, prefix in shared.get('prefixes', {}).items():
            threshold = int(raw)
            if threshold not in self.prefixes or prefix.get('reached') is not True:
                continue
            inherited = dict(copy.deepcopy(prefix), inherited_shared_initialization=True,
                             shared_initialization_sha256=digest(encoded(shared)))
            freeze(self.search.root/'prefixes'/str(threshold)/'PREFIX.json', inherited)
            self.prefix_results[threshold] = inherited

    def auxiliary(self, prompt, kind, parents=None, metadata=None):
        self.before_dispatch()
        parent_records = _native_parents(parents)
        meta = dict(kind=kind, position=self._pending_position or self.position,
                    native_auxiliary=True, parents=[p['hash'] for p in parent_records], **(metadata or {}))
        result = self.search.transport(prompt, meta)
        record = dict(kind=kind, position=self.position, prompt_sha256=digest(prompt),
                      parents=parent_records, response_available=result is not None)
        if getattr(self.search.transport, 'records', None):
            record['terminal'] = self.search.transport.records[-1]
        self.search.state['native_auxiliary'].append(record)
        self.observe('response', request_kind=kind, candidate_response=False)
        return result

    def generate(self, prompt, operator, parents=None, metadata=None, *,
                 defer_prefix=False, after_parse=None, append_module_format=True):
        self.before_dispatch(generation=True)
        position = self.position + 1
        self._pending_position = position
        parent_records = _native_parents(parents)
        parent_hashes = [p['hash'] for p in parent_records]
        prompt = prompt + module_format(self.search) if append_module_format else prompt
        folder = self.search.root/'positions'/f'native-{position:03d}'
        pre = dict(position=position, operator=operator, parents=parent_records,
                   metadata=metadata or {}, prompt_sha256=digest(prompt),
                   native_method=self.search.arm, candidate_unknown=True,
                   archive_hashes=sorted(self.search.state['programs']))
        freeze(folder/'PRE.json', pre)
        initialization = operator == 'i1' or (metadata or {}).get('native_phase') == 'initialization'
        response = self.search.transport(prompt, dict(kind='initialization' if initialization else 'generation',
            position=position, operator=operator, parents=parent_hashes, native_method=self.search.arm,
            predecision_sha256=digest(encoded(pre))))
        self.observe('response', request_kind=operator, candidate_response=True)
        parsed = extract_response(response) if response is not None else None
        source = parsed.source if parsed else None
        error = quality = None
        individual = dict(algorithm=parsed.description if parsed else '', code=source, objective=None, other_inf=None)
        if parsed:
            freeze(folder/'PARSE.json', parsed.evidence())
            if source is not None:
                freeze_bytes(folder/'response_source.py', source.encode('utf-8'))
        if source is not None:
            try:
                ast.parse(source)
            except (ValueError, SyntaxError) as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
            else:
                if after_parse is not None:
                    after_parse(individual)
                    if individual['code'] != source:
                        raise RuntimeError('NATIVE_ALIGNMENT_CANNOT_REWRITE_SOURCE')
                self.search.save_program(source)
                quality = self.search.evaluate(source, f'native-{position:03d}')
        valid = bool(quality and quality.get('complete'))
        h = digest(source) if source is not None else None
        seen = h in self.search.state['programs'] if h else False
        if valid:
            admit(self.search, source, quality, parent_hashes, f'{self.search.arm}_{operator}', position, individual['algorithm'])
            import numpy as np
            individual['objective'] = float(np.round(quality['mean'], 5))
        else:
            individual['code'] = None
        individual['_quality_result'] = quality
        individual['_source_hash'] = h
        individual['_evaluation_timeout'] = bool(quality and (quality.get('timed_out') or quality.get('timeout') or
            any(any(tag in str(r.get(field, '')).upper() for tag in ('TIMEOUT', 'DEADLINE')
                    for field in ('status', 'error_type')) for r in quality.get('rows', []))))
        receipt = dict(position=position, op=operator, parents=parent_hashes, source_hash=h,
            valid=valid, candidate_quality=quality['mean'] if valid else None,
            native_rounded_objective=individual['objective'], error=error,
            response_available=response is not None, parser_status=parsed.status if parsed else None,
            evaluation=quality, restores_seen_program=seen, new_source=bool(h and not seen),
            original_algorithm=parsed.description if parsed else None,
            returned_algorithm=individual['algorithm'], other_inf=individual['other_inf'])
        if not valid:
            self.search.state['failures'].append(receipt)
        key = 'init_completed' if initialization else 'noninit_completed'
        self.search.state[key] = self.search.state.get(key, 0) + 1
        self.position, self._pending_position = position, None
        self.last_individual, self.last_receipt = individual, receipt
        freeze(folder/'RESULT.json', receipt)
        self.observe('evaluation' if quality is not None else 'candidate_unavailable', request_kind=operator,
                     source_hash=h, candidate_valid=valid)
        return individual

    def end_batch(self, population, native_state=None):
        self.search.state['native_population'] = copy.deepcopy(population)
        if native_state is not None:
            self.search.state['native_state'] = copy.deepcopy(native_state)

    def finish(self, population, native_state=None, *, status='TOKEN_TARGET_REACHED', paused=False):
        self.end_batch(population, native_state)
        if not paused:
            self.observe('terminal', termination=status)
        chosen, cost = selection(self.search), self.search.cost_snapshot()
        prefixes = {str(k): v for k, v in self.prefix_results.items()}
        if not paused:
            for point in self.prefixes:
                if str(point) not in prefixes:
                    prefix = dict(threshold_tokens=point, status='NOT_REACHED', reached=False,
                                  reason=status, actual_tokens=token_count(cost), selection=None, cost=None)
                    freeze(self.search.root/'prefixes'/str(point)/'PREFIX.json', prefix)
                    prefixes[str(point)] = prefix
        tag = f'PAUSE-{self.position:03d}' if paused else 'FINAL'
        self.search.checkpoint(tag)
        result = dict(study_id=self.search.context['study_id'], phase=self.search.phase,
            group=self.search.group, arm=self.search.arm, termination=status,
            completed_positions=self.position, planned_positions=self.budget,
            unattempted_positions=self.budget-self.position,
            init_completed=self.search.state.get('init_completed', 0),
            noninit_completed=self.search.state.get('noninit_completed', 0),
            target_tokens=self.target, actual_tokens=token_count(cost),
            selection=chosen, search_best_hash=chosen['source_hash'], search_best_mean=chosen['mean'],
            cost=cost, own_physical_cost=self.search.own_cost_snapshot(), prefixes=prefixes,
            search_locked=not paused, state_path=str(self.search.root/'states'/f'{tag}.json'),
            program_count=len(self.search.state['programs']), failure_count=len(self.search.state['failures']))
        path = self.search.root/(f'PAUSE-{self.position:03d}.json' if paused else 'SEARCH_RESULT.json')
        return freeze(path, result)
