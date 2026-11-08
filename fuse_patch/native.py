"""V3 bounded adapters for pinned complete ReEvo/EoH/HiFo.

ReEvo initialization is physically executed once, then native population, elite,
reflection and NumPy state are hydrated for R. A paused R resumes by replaying
only its continuation through durable request/evaluation identities, never by
rerunning shared initialization. All native algorithm operators remain upstream.
"""
from __future__ import annotations
import copy
from contextlib import ExitStack
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fuse_addon.evaluation import digest, encoded, freeze, freeze_bytes, read
from fuse_credit import baselines as legacy
from fuse_credit.baselines import MODULE_FORMAT, _NumpyProxy
from fuse_paired.reevo import (UPSTREAM, UPSTREAM_COMMIT, NATIVE_METHODS,
    _load_native, _json_state, _FrozenText, adapted_seed)
from fuse_paired.reevo import source_manifest as reevo_source_manifest
from fuse_paired.full_runtime import flatten_native_messages
from .budget import TokenBoundary, DispatchStop, admit, selection, token_count, module_format, CVRP_MODULE_FORMAT

CVRP_SEED = '''import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    feasible = [int(node) for node in unvisited_nodes if demands[int(node)] <= rest_capacity]
    if not feasible:
        return int(depot)
    return min(feasible, key=lambda node: (float(distance_matrix[current_node, node]), node))
'''
from .cvrp_contract import CVRP_SIGNATURE, CVRP_DESCRIPTION, CVRP_FUNC_DESCRIPTION
TSP_DESCRIPTION = ('Solving Traveling Salesman Problem (TSP) with constructive heuristics. TSP requires finding the shortest path that visits all given nodes and returns to the starting node.')


def task_name(io):
    task = str(io.context.get('task', 'tsp')).lower()
    if task not in ('tsp', 'cvrp'):
        raise ValueError('UNKNOWN_NATIVE_TASK')
    return task


def source_manifest(arm, task='tsp'):
    if task not in ('tsp', 'cvrp'):
        raise ValueError('UNKNOWN_NATIVE_TASK')
    result = copy.deepcopy(reevo_source_manifest() if arm in ('R', 'I30') else legacy.source_manifest(arm))
    result.update(adapter='V3 executable-patch study with V3.1 execution contract', task=task,
        target_tokens=300000 if arm == 'I30' else 400000, maximum_generated_positions=30 if arm == 'I30' else 256,
        budget_contract='first available response/evaluation boundary; no new dispatch after target or unknown usage',
        full_task_adaptation='public TSP V2 ABI' if task == 'tsp' else 'V3 preregistered CVRP constructive API, not official ReEvo CVRP solver',
        cvrp_seed_sha256=digest(CVRP_SEED), cvrp_module_format_sha256=digest(CVRP_MODULE_FORMAT), code_repair=False)
    result['inherited_adapter_description'] = result.pop('adaptations', [])
    result['adaptations'] = [
        'exact upstream parent/population selection, prompt builders, reflection and native operators',
        'common durable transport at uniform temperature 1.0, no native connectivity probe',
        'same nonrepairing full-response parser; V3.1 external trusted per-instance stepwise RPC evaluator',
        'all valid complete sources compete by raw Search quality, locked before hidden evaluation',
        'per-run Python/NumPy RNG, immutable original response/evaluation receipts',
        'CVRP replaces only frozen task prompt/API/public seed, not the native algorithm' if task == 'cvrp' else 'unchanged V2 TSP public seed/API projection',
    ]
    result['v3_changes'] = [
        'shared ReEvo30 physical initialization with full independent strategy accounting for J/A/W/R',
        'first-cross token snapshots and separate final actual policy state',
        'hard dispatch stop before every native request, including due maintenance',
        '256 position cap including initialization; unknown usage stops without replacement',
        'early R pause resumes exact persisted continuation without shared initialization replay',
        'V3.1 terminal main quality; original first-cross baseline candidates; all actual overshoot charged',
        'V3.1 ordinary Python clocks allowed; immutable host deadlines and no clock-free claim',
    ]
    return result


def _restore_json(value):
    if isinstance(value, list):
        return [_restore_json(x) for x in value]
    if isinstance(value, dict):
        result = {k: _restore_json(v) for k, v in value.items() if k != 'native_obj_sentinel'}
        if value.get('native_obj_sentinel') == 'positive_infinity':
            result['obj'] = float('inf')
        return result
    return value


def _numpy_state(rng):
    state = rng.get_state()
    return [state[0], state[1].tolist(), *state[2:]]


class _ReEvoClient:
    temperature = 1.
    def __init__(self, owner, kind, generation=False):
        self.owner, self.kind, self.generation = owner, kind, generation

    def multi_chat_completion(self, messages_list, n=1, temperature=None):
        owner, boundary = self.owner, self.owner.boundary
        expanded = [copy.deepcopy(messages) for messages in messages_list for _ in range(n)]
        responses = []
        for index, messages in enumerate(expanded):
            boundary.before_dispatch(generation=self.generation)
            parents = (owner.parents[index] if self.kind in ('initialization', 'crossover', 'short_reflection')
                       else owner.parents[0])
            meta = dict(native_phase='initialization' if self.kind == 'initialization' else 'evolution',
                native_stage=self.kind, native_iteration=owner.engine.iteration,
                native_batch_index=index, temperature=1., native_requested_temperature=temperature,
                native_parent_objectives=[p['obj'] for p in parents])
            owner.inflight = dict(stage=self.kind, batch_index=index, batch_size=len(expanded),
                                  completed_in_current_batch=len(responses))
            if self.generation:
                last_user = max(i for i, row in enumerate(messages) if row['role'] == 'user')
                messages[last_user]['content'] += module_format(owner.search)
            prompt = flatten_native_messages(messages)
            with owner.search.native_messages(messages):
                if self.generation:
                    item = boundary.generate(prompt, 'i1' if self.kind == 'initialization' else self.kind,
                        parents, meta, append_module_format=False)
                    quality = item['_quality_result']
                    valid = bool(quality and quality.get('complete'))
                    raw = quality['mean'] if valid else None
                    native_item = dict(code=item['code'], response_id=index,
                        code_path=str(owner.search.root/'positions'/f'native-{boundary.position:03d}'/'response_source.py'),
                        stdout_filepath=None, exec_success=valid, obj=raw if valid else float('inf'), objective=raw,
                        source_hash=item['_source_hash'], position=boundary.position,
                        traceback_msg=None if valid else str(boundary.last_receipt.get('error') or 'Invalid response or Search evaluation'),
                        quality_mean=raw)
                    freeze(owner.search.root/'positions'/f'native-{boundary.position:03d}'/'REEVO_OBJECTIVE.json',
                        dict(position=boundary.position, source_hash=item['_source_hash'], exec_success=valid,
                             native_objective=raw, selection_objective='raw Search mean',
                             common_receipt_rounded_objective_unused=item['objective'],
                             invalid_native_sentinel=None if valid else 'positive_infinity'))
                    responses.append(native_item)
                else:
                    response = boundary.auxiliary(prompt, self.kind, parents, meta)
                    if response is None:
                        raise DispatchStop('REFLECTION_RESPONSE_UNAVAILABLE:'+self.kind)
                    responses.append(response)
            owner.partial_responses = copy.deepcopy(responses)
            owner.inflight['completed_in_current_batch'] = len(responses)
            owner.inflight['next_batch_index'] = index + 1
        owner.inflight = None
        owner.partial_responses = []
        return responses


def _reevo(io, *, shared=None, initialization_only=False, continuation_limit=None):
    result_path = io.root/('SHARED_INITIALIZATION.json' if initialization_only else 'SEARCH_RESULT.json')
    if result_path.exists():
        return read(result_path)
    if not initialization_only and io.arm != 'R':
        raise ValueError('REEVO_CONTINUATION_REQUIRES_R')
    if initialization_only and shared is not None:
        raise ValueError('INITIALIZATION_CANNOT_INHERIT_INITIALIZATION')
    io.replay_native_from_start()
    if shared is not None:
        io.attach_shared(shared)
    boundary = TokenBoundary(io, target=300000 if initialization_only else 400000,
        position_cap=30 if initialization_only else 256,
        initial_positions=0 if initialization_only else shared['completed_positions'],
        continuation_limit=continuation_limit)
    boundary.ensure_reference()
    if shared is not None:
        boundary.inherit_prefixes(shared)
    manifest = source_manifest('I30' if initialization_only else 'R', task_name(io))
    freeze(io.root/'REEVO_SOURCE_MANIFEST.json', manifest)
    native = _load_native('_patch_reevo_'+digest(str(io.root))[:12])
    nrng = np.random.RandomState(io.rng('native-numpy', 0) % (2**32))
    native.np = _NumpyProxy(np, nrng)
    native.open = lambda name, mode='r', encoding=None: _FrozenText(io.root/'native'/Path(name).name)
    owner = SimpleNamespace(search=io, boundary=boundary, parents=[], engine=None, stages=[],
                            inflight=None, partial_responses=[])

    def snapshot(reason=None):
        engine = owner.engine
        fields = ('population', 'elitist', 'iteration', 'function_evals', 'best_obj_overall',
            'best_code_overall', 'best_code_path_overall', 'long_term_reflection_str',
            'external_knowledge', 'seed_ind', 'print_crossover_prompt', 'print_mutate_prompt',
            'print_short_term_reflection_prompt', 'print_long_term_reflection_prompt')
        return _json_state(dict(upstream_commit=UPSTREAM_COMMIT,
            attributes={key: copy.deepcopy(getattr(engine, key, None)) for key in fields},
            completed_stages=owner.stages, stop_reason=reason, generated_positions=boundary.position,
            numpy_rng_state=_numpy_state(nrng), inflight=owner.inflight, partial_responses=owner.partial_responses,
            recovery='hydrate shared state; deterministic exact-request continuation replay'))

    class Adapted(native.ReEvo):
        def init_prompt(self):
            super().init_prompt()
            if task_name(io) == 'tsp':
                self.seed_func = adapted_seed()
                self.func_signature = self.func_signature.replace('select_next_node_v{version}', 'select_next_node').replace('unvisited_nodes: set', 'unvisited_nodes: np.ndarray')
                self.func_desc = self.func_desc.replace('a set of unvisited nodes', 'an ndarray of unvisited nodes')
            else:
                self.seed_func, self.func_signature, self.func_desc = CVRP_SEED, CVRP_SIGNATURE, CVRP_FUNC_DESCRIPTION
                self.external_knowledge = self.long_term_reflection_str = ''
            self.user_generator_prompt = (UPSTREAM/'prompts/common/user_generator.txt').read_text().format(
                func_name=self.func_name, problem_desc=self.problem_desc, func_desc=self.func_desc)
            self.seed_prompt = (UPSTREAM/'prompts/common/seed.txt').read_text().replace('{func_name}_v2', '{func_name}').format(
                seed_func=self.seed_func, func_name=self.func_name)
            self.crossover_prompt = self.crossover_prompt.replace('{func_name}_v2', '{func_name}')
            self.mutation_prompt = self.mutation_prompt.replace('{func_name}_v2', '{func_name}')

        def init_population(self):
            owner.engine = self
            if shared is not None:
                if shared.get('native_state', {}).get('upstream_commit') != UPSTREAM_COMMIT:
                    raise RuntimeError('SHARED_NATIVE_COMMIT_CONFLICT')
                for key, value in _restore_json(shared['native_state']['attributes']).items():
                    setattr(self, key, value)
                rs = shared['native_state']['numpy_rng_state']
                nrng.set_state((rs[0], np.asarray(rs[1], dtype=np.uint32), *rs[2:]))
                owner.stages = copy.deepcopy(shared['native_state']['completed_stages'])
                return
            code = self.seed_func
            h = io.save_program(code)
            quality = io.evaluate(code, 'reevo-native-seed')
            valid = bool(quality.get('complete'))
            raw = quality['mean'] if valid else None
            freeze_bytes(io.root/'native'/'seed.py', code.encode())
            self.seed_ind = dict(code=code, response_id=0, code_path=str(io.root/'native'/'seed.py'),
                stdout_filepath=None, exec_success=valid, obj=raw if valid else float('inf'),
                objective=raw, source_hash=h, position=0, quality_mean=raw)
            self.population = [self.seed_ind]
            self.function_evals += 1
            freeze(io.root/'native'/'SEED.json', dict(source_hash=h, generated_position=False,
                complete=valid, native_objective=raw, evaluation=quality,
                source_adaptation='frozen public TSP V2 projection' if task_name(io) == 'tsp' else 'preregistered public feasible-nearest-neighbor CVRP seed'))
            if valid:
                admit(io, code, quality, [], 'R_public_native_seed', 0)
            boundary.observe('evaluation', request_kind='public_native_seed', source_hash=h, candidate_valid=valid)
            if not valid:
                raise DispatchStop('NATIVE_SEED_INVALID')
            self.update_iter()
            owner.stages.append(dict(stage='seed_update', position=0))
            messages = [{'role': 'system', 'content': self.system_generator_prompt},
                {'role': 'user', 'content': self.user_generator_prompt+'\n'+self.seed_prompt+'\n'+self.long_term_reflection_str}]
            owner.parents = [[self.seed_ind]] * self.cfg.init_pop_size
            responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size,
                temperature=self.generator_llm.temperature+.3)
            self.population = self.evaluate_population([self.response_to_individual(r, i) for i, r in enumerate(responses)])
            self.update_iter()
            owner.stages.append(dict(stage='initialization_update', position=boundary.position))
            boundary.end_batch(_json_state(self.population), snapshot())

        def response_to_individual(self, response, response_id, file_name=None):
            return dict(response, response_id=response_id)

        def evaluate_population(self, population):
            self.function_evals += len(population)
            return population

        def _run_code(self, individual, response_id):
            raise AssertionError('NATIVE_EXECUTION_MUST_USE_COMMON_EVALUATOR')

        def evolve(self):
            while True:
                boundary.before_dispatch()
                if all(not item['exec_success'] for item in self.population):
                    raise DispatchStop('NATIVE_ALL_INDIVIDUALS_INVALID')
                pool = self.population if self.elitist is None or self.elitist in self.population else [self.elitist]+self.population
                selected = self.random_select(pool)
                if selected is None:
                    raise DispatchStop('NATIVE_SELECTION_FAILED')
                owner.parents = [selected[i:i+2] for i in range(0, len(selected), 2)]
                reflected = self.short_term_reflection(selected)
                owner.stages.append(dict(stage='short_reflection', position=boundary.position))
                crossed = self.crossover(reflected)
                self.population = self.evaluate_population(crossed)
                self.update_iter()
                owner.stages.append(dict(stage='crossover_update', position=boundary.position))
                owner.parents = [selected]
                self.long_term_reflection(list(reflected[0]))
                owner.stages.append(dict(stage='long_reflection', position=boundary.position))
                owner.parents = [[self.elitist]]
                mutated = self.mutate()
                self.population.extend(self.evaluate_population(mutated))
                self.update_iter()
                owner.stages.append(dict(stage='mutation_update', position=boundary.position, dispatched=len(mutated)))
                boundary.end_batch(_json_state(self.population), snapshot())

    cfg = SimpleNamespace(pop_size=10, init_pop_size=30, mutation_rate=.5, max_fe=257, timeout=500,
        problem=SimpleNamespace(problem_name='tsp_constructive', problem_type='constructive', obj_type='min',
            problem_size=50, func_name='select_next_node', description=TSP_DESCRIPTION if task_name(io) == 'tsp' else CVRP_DESCRIPTION))
    status = 'SHARED_INITIALIZATION_COMPLETE'
    try:
        engine = Adapted(cfg, str(UPSTREAM), _ReEvoClient(owner, 'initialization', True),
            short_reflector_llm=_ReEvoClient(owner, 'short_reflection'),
            long_reflector_llm=_ReEvoClient(owner, 'long_reflection'),
            crossover_llm=_ReEvoClient(owner, 'crossover', True),
            mutation_llm=_ReEvoClient(owner, 'mutation', True))
        if not initialization_only:
            engine.evolve()
    except DispatchStop as exc:
        status = exc.reason
    engine = owner.engine
    state = snapshot(status)
    population = _json_state(getattr(engine, 'population', []))
    if initialization_only:
        boundary.end_batch(population, state)
        io.checkpoint('SHARED_INITIALIZATION')
        value = dict(schema='v3-shared-reevo-initialization', completed_positions=boundary.position,
            complete=boundary.position == 30 and status == 'SHARED_INITIALIZATION_COMPLETE',
            termination=status, task=task_name(io), phase=io.phase, group=io.group,
            archive_state=copy.deepcopy(io.state), native_state=state,
            selection=selection(io), cost=io.cost_snapshot(), own_physical_cost=io.own_cost_snapshot(),
            reference=copy.deepcopy(io.reference), prefixes={str(k): v for k, v in boundary.prefix_results.items()},
            boundary_events=boundary.event_counter, source_manifest=manifest,
            physical_owner=str(io.root), init_root=str(io.root),
            evaluation_cache_root=str(io.root/'evaluator'),
            initialization_positions=[read(io.root/'positions'/f'native-{i:03d}'/'RESULT.json')
                                      for i in range(1, boundary.position+1)],
            public_seed=read(io.root/'native'/'SEED.json'), initialization_requested_once=True)
        return freeze(result_path, value)
    paused = status == 'PAUSED_EARLY_CHECK'
    return boundary.finish(population, state, status=status, paused=paused)


def run_shared_initialization(io):
    return _reevo(io, initialization_only=True)


def run_reevo(io, shared, continuation_limit=None):
    if not shared.get('complete'):
        io.replay_native_from_start()
        io.attach_shared(shared)
        boundary = TokenBoundary(io, target=400000, initial_positions=shared['completed_positions'])
        boundary.inherit_prefixes(shared)
        return boundary.finish(shared['native_state']['attributes'].get('population') or [],
            shared['native_state'], status='SHARED_INITIALIZATION_INCOMPLETE:'+shared['termination'])
    return _reevo(io, shared=shared, continuation_limit=continuation_limit)


class _CVRPPrompts:
    def get_task(self): return CVRP_DESCRIPTION
    def get_func_name(self): return 'select_next_node'
    def get_func_inputs(self): return ['current_node', 'depot', 'unvisited_nodes', 'rest_capacity', 'demands', 'distance_matrix']
    def get_func_outputs(self): return ['next_node']
    def get_inout_inf(self): return CVRP_FUNC_DESCRIPTION
    def get_other_inf(self): return 'Use numpy arrays as supplied; return int node, never bool. Do not mutate inputs.'


def run_native(search):
    """Complete E/H native policy, adapted solely at existing common I/O seams."""
    arm = search.arm
    if arm not in ('E', 'H'):
        raise ValueError('E_OR_H_REQUIRED')
    path = search.root/'SEARCH_RESULT.json'
    if path.exists():
        return read(path)
    search.replay_native_from_start()
    boundary = TokenBoundary(search, target=400000)
    boundary.ensure_reference()
    modules = legacy._native_modules(arm, '_patch_native_'+arm+'_'+digest(str(search.root))[:12])
    hm, im, evolution = (modules[name] for name in ('engine', 'interface', 'evolution'))
    paras = modules['parameters'].Paras()
    paras.set_paras()
    original_horizon = paras.ec_n_pop
    # Bound is a dispatch cap, not a replacement native policy/hyperparameter.
    import math
    paras.ec_n_pop = max(1, math.ceil((256-2*paras.ec_pop_size)/(paras.ec_pop_size*len(paras.ec_operators))))
    paras.exp_output_path = str(search.root/'native')
    for folder in ('results/pops', 'results/pops_best'):
        (Path(paras.exp_output_path)/folder).mkdir(parents=True, exist_ok=True)
    paras.llm_api_endpoint, paras.llm_api_key = 'durable-common-transport', 'not-a-credential'
    paras.llm_model = getattr(search.transport, 'binding', {}).get('model_settings', {}).get('model')
    rng = random.Random(search.rng('native-python', 0))
    nrng = np.random.RandomState(search.rng('native-numpy', 0) % (2**32))
    proxy = _NumpyProxy(np, nrng)
    owner = SimpleNamespace(current=None, interface=None, population=[], batches=0)
    Base = im.InterfaceEC

    def snapshot():
        ec = owner.interface
        result = dict(completed_native_batches=owner.batches, rng_state=rng.getstate(), numpy_rng_state=_numpy_state(nrng))
        if arm == 'H' and ec is not None:
            result.update(best_fitness_history=ec.best_fitness_history, avg_fitness_history=ec.avg_fitness_history,
                diversity_history=ec.diversity_history, insight_tips=list(ec.insight_pool.tips),
                insight_stats=ec.insight_pool.tip_stats, insight_generation=ec.insight_pool.current_generation,
                navigator=vars(ec.navigator))
        return result

    class Bridge:
        def __init__(self, *args, **kwargs): pass
        def get_response(self, prompt):
            try:
                return boundary.auxiliary(prompt, 'insight_extraction', owner.population)
            except Exception as exc:
                # Upstream insight parsing catches Exception; infrastructure
                # corruption must escape instead of becoming an absent tip.
                raise legacy._InfrastructureStop(exc) from exc

    def extract_once(evol, prompt):
        current = owner.current
        individual = boundary.generate(prompt, current['operator'], current['parents'], current['metadata'])
        return [individual['code'], individual['algorithm']]

    class HookedEC(Base):
        def __init__(ec, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owner.interface = ec
            for name in ('i1', *paras.ec_operators):
                original = getattr(ec.evol, name)
                def dispatch(*args, _name=name, _original=original, **kwargs):
                    owner.current['parents'] = None if _name == 'i1' else args[0]
                    if arm == 'H':
                        offset = 0 if _name == 'i1' else 1
                        owner.current['metadata'] = {key: copy.deepcopy(args[offset+j] if len(args)>offset+j else kwargs.get(key))
                            for j, key in enumerate(('insights', 'design_directive', 'regime'))}
                    return _original(*args, **kwargs)
                setattr(ec.evol, name, dispatch)

        def get_offspring(ec, pop, operator):
            owner.current = dict(operator=operator, parents=[], metadata={})
            def attempt():
                parents, offspring = ec._get_alg(pop, operator)
                if arm == 'H' and 'timestamp' in offspring.get('metadata', {}):
                    path = search.root/'positions'/f'native-{boundary.position:03d}'/'NATIVE_METADATA.json'
                    if path.exists(): offspring['metadata']['timestamp'] = read(path)['timestamp']
                    freeze(path, offspring['metadata'])
                return parents, offspring
            parents, off = attempt()
            if off['code'] is not None and ec.check_duplicate(pop, off['code']):
                boundary.before_dispatch(generation=True)
                parents, off = attempt()
            evaluated = boundary.last_individual
            off.update({key: evaluated[key] for key in ('code', 'algorithm', 'objective', 'other_inf')})
            if arm == 'H' and off['objective'] is not None:
                ec.update_insight_feedback(off, pop)
            return parents, off

        def get_algorithm(ec, pop, operator):
            boundary.before_dispatch()
            owner.population = pop
            if arm == 'H': ec.insight_pool.update_generation(len(ec.best_fitness_history))
            results = []
            for _ in range(ec.pop_size):
                boundary.before_dispatch(generation=True)
                results.append(ec.get_offspring(pop, operator))
            # Upstream two-second scheduling delay changes no scientific state;
            # retained by runtime sleep outside candidate execution.
            import time
            time.sleep(2)
            if arm == 'H' and pop:
                ec.update_population_metrics(pop)
                if nrng.random() < .8:
                    ec.extract_insights_from_population(pop)
            owner.batches += 1
            if operator == 'i1': boundary.end_batch(pop, snapshot())
            return [p for p, _ in results], [off for _, off in results]

    original_manage = modules['manage'].population_management
    def managed(pop, size):
        result = original_manage(pop, size)
        owner.population = result
        boundary.end_batch(result, snapshot())
        if not result: raise DispatchStop('NATIVE_INITIALIZATION_FAILED')
        return result

    freeze(search.root/'NATIVE_SOURCE_CONFIG.json', dict(source_manifest(arm, task_name(search)),
        actual_horizon_generations=paras.ec_n_pop, original_horizon_generations=original_horizon,
        actual_parameters=vars(paras), rng_seed_python=search.rng('native-python', 0), rng_seed_numpy=search.rng('native-numpy', 0)))
    status = 'NATIVE_HORIZON_EXHAUSTED'
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(evolution, 'InterfaceLLM', Bridge))
            stack.enter_context(patch.object(evolution.Evolution, '_get_alg', extract_once))
            stack.enter_context(patch.object(hm, 'InterfaceEC', HookedEC))
            stack.enter_context(patch.object(modules['manage'], 'population_management', managed))
            for module in modules.values():
                if hasattr(module, 'random'): stack.enter_context(patch.object(module, 'random', rng))
                if hasattr(module, 'np'): stack.enter_context(patch.object(module, 'np', proxy))
            prompts = modules['prompts'].GetPrompts() if task_name(search) == 'tsp' else _CVRPPrompts()
            engine = (hm.EOH if arm == 'E' else hm.HiFo)(paras, SimpleNamespace(prompts=prompts), modules['select'], modules['manage'])
            rng.seed(search.rng('native-python', 0))
            engine.run()
    except DispatchStop as exc:
        status = exc.reason
    except legacy._InfrastructureStop as exc:
        raise exc.cause
    return boundary.finish(owner.population, snapshot(), status=status)
