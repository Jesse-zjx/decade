"""Pinned full ReEvo, with common scientific I/O and a generated-position cap.

The upstream selection, reflection, crossover, mutation and update methods are
called directly. Only task ABI, durable I/O and the finite dispatch boundary are
adapted. Generated sources always go through the common non-repairing parser.
"""
from __future__ import annotations

import copy
import importlib.util
import io as text_io
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from fuse_addon.evaluation import digest, freeze, freeze_bytes, read
from fuse_credit.baselines import MODULE_FORMAT, _NumpyProxy
from fuse_credit.state import register_program
from .baselines import BaselineIO, native_io_scope
from .full_runtime import flatten_native_messages

UPSTREAM = Path(__file__).resolve().parents[1] / 'vendor/upstream_reevo'
UPSTREAM_COMMIT = '6dce18257da5e11db2d138e417a2fffc5c72d05f'
NATIVE_METHODS = ('random_select', 'gen_short_term_reflection_prompt',
                  'short_term_reflection', 'long_term_reflection', 'crossover',
                  'mutate', 'update_iter', 'mark_invalid_individual')


def adapted_seed():
    """Fixed public seed adaptation only; never called on a generated response."""
    source = (UPSTREAM/'prompts/tsp_constructive/seed_func.txt').read_text()
    return 'import numpy as np\n\n' + source.replace(
        'select_next_node_v1', 'select_next_node').replace(
        'unvisited_nodes: set', 'unvisited_nodes: np.ndarray')


def source_manifest():
    manifest = read(UPSTREAM/'SOURCE_MANIFEST.json')
    if manifest['upstream_commit'] != UPSTREAM_COMMIT:
        raise RuntimeError('REEVO_UPSTREAM_COMMIT_CONFLICT')
    for name, expected in manifest['files'].items():
        if digest((UPSTREAM/name).read_bytes()) != expected:
            raise RuntimeError('REEVO_UPSTREAM_SOURCE_CONFLICT: ' + name)
    return dict(arm='R', upstream=manifest, direct_native_methods=list(NATIVE_METHODS),
        defaults=dict(init_pop_size=30, pop_size=10, mutation_rate=.5),
        adapted_seed_sha256=digest(adapted_seed()),
        objective='raw complete Search mean; no rounding in native selection',
        temperature=1., seed_search_separately_charged=True,
        initialization_in_generated_budget=True,
        adaptations=[
            'public select_next_node name and ndarray ABI in fixed seed and request prompts',
            'fixed seed numpy import; generated responses never repaired',
            'common model temperature 1 including initialization (native +0.3 overridden)',
            'durable role-preserving transport, common parser and isolated Search evaluator',
            'seed paid separately; common NN supplies scale and operational fallback',
            'generated-position cap clips last dispatch; already-due reflection and update retained',
            'per-run NumPy RNG; native invalid infinity represented by null plus sentinel in JSON',
            'native stop retains unattempted positions; no replacement search policy',
        ])


def _load_native(namespace):
    """Load exact source with typing/config-only dependencies isolated from E/H."""
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded

    hydra = ModuleType('hydra')
    omega = ModuleType('omegaconf')
    omega.DictConfig = SimpleNamespace
    base = ModuleType('utils.llm_client.base')
    base.BaseClient = object
    # None of Hydra's configuration/client factories is used: actual clients
    # below enter the common durable transport. These imports are annotations.
    with patch.dict(sys.modules, {'hydra': hydra, 'omegaconf': omega}):
        utils = module(namespace+'_utils', UPSTREAM/'utils/utils.py')
        with patch.dict(sys.modules, {'utils': ModuleType('utils'), 'utils.utils': utils,
                                     'utils.llm_client': ModuleType('utils.llm_client'),
                                     'utils.llm_client.base': base}):
            return module(namespace, UPSTREAM/'reevo.py')


def _json_state(value):
    """Only serialization changes the native invalid +inf sentinel to null."""
    if isinstance(value, dict):
        result = {k: _json_state(v) for k, v in value.items()}
        if isinstance(value.get('obj'), float) and math.isinf(value['obj']):
            result['native_obj_sentinel'] = 'positive_infinity'
        return result
    if isinstance(value, (list, tuple)):
        return [_json_state(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _NativeStop(Exception):
    """Native failure with an explicit incomplete-budget result, never rescue."""


class _FrozenText(text_io.StringIO):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def close(self):
        if not self.closed:
            freeze_bytes(self.path, self.getvalue().encode())
        super().close()


class _Client:
    temperature = 1.

    def __init__(self, owner, kind, generation=False):
        self.owner, self.kind, self.generation = owner, kind, generation

    def multi_chat_completion(self, messages_list, n=1, temperature=None):
        owner, boundary = self.owner, self.owner.boundary
        expanded = [copy.deepcopy(messages) for messages in messages_list for _ in range(n)]
        if self.generation:
            expanded = expanded[:max(0, boundary.budget-boundary.position)]
        responses = []
        for index, messages in enumerate(expanded):
            parents = owner.parents[index] if self.kind in ('initialization', 'crossover', 'short_reflection') else owner.parents[0]
            meta = dict(native_phase='initialization' if self.kind == 'initialization' else 'evolution',
                        native_stage=self.kind, native_iteration=owner.engine.iteration,
                        native_batch_index=index, temperature=1.,
                        native_requested_temperature=temperature,
                        native_parent_objectives=[p['obj'] for p in parents])
            if self.generation:
                # Preserve roles, appending only the common complete-module ABI.
                last_user = max(i for i, row in enumerate(messages) if row['role'] == 'user')
                messages[last_user]['content'] += MODULE_FORMAT
            prompt = flatten_native_messages(messages)
            with owner.search.native_messages(messages):
                if self.generation:
                    item = boundary.generate(prompt, 'i1' if self.kind == 'initialization' else self.kind,
                        parents, meta, append_module_format=False, defer_prefix=True)
                    quality = item['_quality_result']
                    valid = bool(quality and quality.get('complete'))
                    raw = quality['mean'] if valid else None
                    native_item = dict(code=item['code'], response_id=index,
                        code_path=str(owner.search.root/'positions'/f'native-{boundary.position:03d}'/'response_source.py'),
                        stdout_filepath=None, exec_success=valid,
                        obj=raw if valid else float('inf'), objective=raw,
                        source_hash=item['_source_hash'], position=boundary.position,
                        traceback_msg=None if valid else (boundary.last_receipt.get('error') or 'Invalid response or Search evaluation'),
                        quality_mean=raw)
                    freeze(owner.search.root/'positions'/f'native-{boundary.position:03d}'/'REEVO_OBJECTIVE.json',
                        dict(position=boundary.position, source_hash=item['_source_hash'],
                             exec_success=valid, native_objective=raw, selection_objective='raw Search mean',
                             common_receipt_rounded_objective_unused=item['objective'],
                             invalid_native_sentinel=None if valid else 'positive_infinity'))
                    responses.append(native_item)
                else:
                    response = boundary.auxiliary(prompt, self.kind, parents, meta)
                    if response is None:
                        raise _NativeStop('REFLECTION_RESPONSE_UNAVAILABLE: ' + self.kind)
                    responses.append(response)
        return responses


def run_reevo(io, budget=128, prefixes=(128,)):
    """Run complete official ReEvo in its own worker process through FreshSearchIO."""
    search = io
    if search.arm != 'R':
        raise ValueError('REEVO_REQUIRES_R_ARM')
    if any(0 < p < budget for p in prefixes):
        raise ValueError('REEVO_ONLY_FINAL_PREFIX_IS_PREREGISTERED')
    result_path = search.root/'SEARCH_RESULT.json'
    if result_path.exists():
        return read(result_path)
    manifest = source_manifest()
    search.replay_native_from_start()
    with native_io_scope(search):
        boundary = BaselineIO(search, budget, prefixes)
        boundary.ensure_reference()
        freeze(search.root/'REEVO_SOURCE_MANIFEST.json', manifest)
        native = _load_native('_paired_reevo_'+digest(str(search.root))[:12])
        nrng = np.random.RandomState(search.rng('native-numpy', 0) % (2**32))
        native.np = _NumpyProxy(np, nrng)
        # Native long-term reflection keeps its exact write calls, redirected
        # to immutable per-run records without process-wide cwd changes.
        native.open = lambda name, mode='r', encoding=None: _FrozenText(search.root/'native'/Path(name).name)
        owner = SimpleNamespace(search=search, boundary=boundary, parents=[], engine=None, stages=[])

        def snapshot(reason=None):
            engine = owner.engine
            state = dict(upstream_commit=UPSTREAM_COMMIT, completed_stages=owner.stages,
                stop_reason=reason, generated_positions=boundary.position,
                seed_evaluation_count=1 if hasattr(engine, 'seed_ind') else 0,
                native_function_evals=engine.function_evals,
                iteration=engine.iteration, elitist=engine.elitist,
                best_obj_overall=engine.best_obj_overall,
                best_code_overall=engine.best_code_overall,
                long_term_reflection=engine.long_term_reflection_str,
                external_knowledge=engine.external_knowledge,
                seed_individual=getattr(engine, 'seed_ind', None),
                numpy_rng_state=[nrng.get_state()[0], nrng.get_state()[1].tolist(), *nrng.get_state()[2:]])
            return _json_state(state)

        class Adapted(native.ReEvo):
            def init_prompt(self):
                super().init_prompt()
                self.seed_func = adapted_seed()
                self.func_signature = self.func_signature.replace('select_next_node_v{version}', 'select_next_node').replace(
                    'unvisited_nodes: set', 'unvisited_nodes: np.ndarray')
                self.func_desc = self.func_desc.replace('a set of unvisited nodes', 'an ndarray of unvisited nodes')
                self.user_generator_prompt = (UPSTREAM/'prompts/common/user_generator.txt').read_text().format(
                    func_name=self.func_name, problem_desc=self.problem_desc, func_desc=self.func_desc)
                self.seed_prompt = (UPSTREAM/'prompts/common/seed.txt').read_text().replace(
                    '{func_name}_v2', '{func_name}').format(seed_func=self.seed_func, func_name=self.func_name)
                self.crossover_prompt = self.crossover_prompt.replace('{func_name}_v2', '{func_name}')
                self.mutation_prompt = self.mutation_prompt.replace('{func_name}_v2', '{func_name}')

            def init_population(self):
                owner.engine = self
                code = self.seed_func
                h = search.save_program(code)
                quality = search.evaluate(code, 'reevo-native-seed')
                valid = bool(quality.get('complete'))
                raw = quality['mean'] if valid else None
                freeze_bytes(search.root/'native'/'seed.py', code.encode())
                self.seed_ind = dict(code=code, response_id=0, code_path=str(search.root/'native'/'seed.py'),
                    stdout_filepath=None, exec_success=valid, obj=raw if valid else float('inf'),
                    objective=raw, source_hash=h, position=0, quality_mean=raw)
                self.population = [self.seed_ind]
                self.function_evals += 1
                freeze(search.root/'native'/'SEED.json', dict(source_hash=h, generated_position=False,
                    complete=valid, native_objective=raw, evaluation=quality,
                    source_adaptation='fixed public name, ndarray annotation, numpy import only'))
                if not valid:
                    raise _NativeStop('NATIVE_SEED_INVALID')
                register_program(search.state, code, quality, [], 'R_public_native_seed', 0)
                self.update_iter()
                owner.stages.append(dict(stage='seed_update', position=0))
                messages = [{'role': 'system', 'content': self.system_generator_prompt},
                    {'role': 'user', 'content': self.user_generator_prompt+'\n'+self.seed_prompt+'\n'+self.long_term_reflection_str}]
                owner.parents = [[self.seed_ind]] * self.cfg.init_pop_size
                responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size,
                                                                     temperature=self.generator_llm.temperature+.3)
                self.population = self.evaluate_population([self.response_to_individual(response, i)
                                                            for i, response in enumerate(responses)])
                self.update_iter()
                owner.stages.append(dict(stage='initialization_update', position=boundary.position))
                boundary.end_batch(_json_state(self.population), snapshot())

            def response_to_individual(self, response, response_id, file_name=None):
                # The common parser/evaluator has produced this exact record.
                return dict(response, response_id=response_id)

            def evaluate_population(self, population):
                self.function_evals += len(population)
                return population

            def _run_code(self, individual, response_id):
                raise AssertionError('NATIVE_EXECUTION_MUST_USE_COMMON_EVALUATOR')

            def evolve(self):
                while boundary.position < boundary.budget:
                    if all(not individual['exec_success'] for individual in self.population):
                        raise _NativeStop('NATIVE_ALL_INDIVIDUALS_INVALID')
                    pool = self.population if self.elitist is None or self.elitist in self.population else [self.elitist]+self.population
                    selected = self.random_select(pool)
                    if selected is None:
                        raise _NativeStop('NATIVE_SELECTION_FAILED')
                    owner.parents = [selected[i:i+2] for i in range(0, len(selected), 2)]
                    reflected = self.short_term_reflection(selected)
                    owner.stages.append(dict(stage='short_reflection', position=boundary.position))
                    # All native reflections already issued remain charged. Only
                    # unissued candidate dispatch is clipped at the final batch.
                    count = min(self.cfg.pop_size, boundary.budget-boundary.position)
                    old_size = self.cfg.pop_size
                    self.cfg.pop_size = count
                    try:
                        crossed = self.crossover(tuple(values[:count] for values in reflected))
                    finally:
                        self.cfg.pop_size = old_size
                    self.population = self.evaluate_population(crossed)
                    self.update_iter()
                    owner.stages.append(dict(stage='crossover_update', position=boundary.position))
                    owner.parents = [selected]
                    self.long_term_reflection([response for response in reflected[0]])
                    owner.stages.append(dict(stage='long_reflection', position=boundary.position))
                    owner.parents = [[self.elitist]]
                    mutated = self.mutate()
                    self.population.extend(self.evaluate_population(mutated))
                    self.update_iter()
                    owner.stages.append(dict(stage='mutation_update', position=boundary.position,
                                             dispatched=len(mutated)))
                    boundary.end_batch(_json_state(self.population), snapshot())
                return self.best_code_overall, self.best_code_path_overall

        cfg = SimpleNamespace(pop_size=10, init_pop_size=30, mutation_rate=.5,
            max_fe=budget+1, timeout=500, problem=SimpleNamespace(problem_name='tsp_constructive',
                problem_type='constructive', obj_type='min', problem_size=50, func_name='select_next_node',
                description='Solving Traveling Salesman Problem (TSP) with constructive heuristics. TSP requires finding the shortest path that visits all given nodes and returns to the starting node.'))
        status, reason = 'NATIVE_BUDGET_COMPLETE', None
        try:
            engine = Adapted(cfg, str(UPSTREAM), _Client(owner, 'initialization', True),
                short_reflector_llm=_Client(owner, 'short_reflection'),
                long_reflector_llm=_Client(owner, 'long_reflection'),
                crossover_llm=_Client(owner, 'crossover', True),
                mutation_llm=_Client(owner, 'mutation', True))
            engine.evolve()
        except _NativeStop as exc:
            reason, status = str(exc), 'NATIVE_STOP_INCOMPLETE_BUDGET'
        engine = owner.engine
        freeze(search.root/'native'/'TERMINATION.json', dict(status=status, reason=reason,
            completed_positions=boundary.position, planned_positions=boundary.budget,
            unattempted_positions=boundary.budget-boundary.position,
            no_replacement_search_policy=True))
        return boundary.finish(_json_state(engine.population), snapshot(reason), status=status)
