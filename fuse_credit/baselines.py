"""Complete pinned legacy EoH/HiFo with bounded common scientific I/O.

Native prompt builders, parent selection, population management, initialization,
and HiFo's insight/navigation methods are called from the actual pinned modules.
Every fresh candidate response occupies its own main position, including native
duplicate regenerations. Parsing never repairs code or requests another answer.
Each runner must live in its own process; module aliases prevent EoH API mixing.
"""
from __future__ import annotations

import ast
import copy
from contextlib import ExitStack
import importlib
import importlib.util
import math
from pathlib import Path
import random
import sys
import time
import types
from unittest.mock import patch

from fuse_addon.evaluation import digest, encoded, freeze, freeze_bytes, read
from fuse_addon.response_parser import extract_response
from .generation import MODULE_FORMAT

REPO = Path(__file__).resolve().parents[1]
PREFIXES = (64, 96, 128, 160, 192)
SOURCE_PINS = {"E": "4b2338fc2317c743cec50c5c7a2e4684707af973",
               "H": "e64ce9edbfb4c8ebffd652b785b0c87261785586"}


class BudgetComplete(Exception):
    """No next candidate may be started at the frozen main-position limit."""


class _InitializationFailed(BaseException):
    pass


class _InfrastructureStop(BaseException):
    """Escape upstream broad Exception handlers without masking host failures."""
    def __init__(self, cause):
        self.cause = cause


def _root(arm):
    if arm == "E":
        return REPO / "vendor/upstream_eoh_v01/eoh/src/eoh"
    if arm == "H":
        return REPO / "vendor/upstream_hifo/hifo/src/hifo"
    raise ValueError("Native E/H adapter only")


def _file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_defaults(arm):
    """Read the actual Paras implementation; no API or candidate is executed."""
    module = _file_module("_credit_native_defaults_" + arm, _root(arm) / "utils/getParas.py")
    paras = module.Paras()
    paras.set_paras()
    return vars(paras).copy()


def native_batch_plan(arm, budget=192):
    """Fixed creation order before any duplicate attempts; useful freeze audit."""
    if budget <= 0:
        raise ValueError("POSITIVE_BUDGET_REQUIRED")
    defaults = native_defaults(arm)
    size = defaults["ec_pop_size"]
    remaining = budget
    batches = []
    for phase, op in [("initialization", "i1")] * 2:
        count = min(size, remaining)
        if count:
            batches.append({"phase": phase, "operator": op, "count": count})
        remaining -= count
    while remaining:
        for op in defaults["ec_operators"]:
            count = min(size, remaining)
            if count:
                batches.append({"phase": "evolution", "operator": op, "count": count})
            remaining -= count
    return batches


def source_manifest(arm):
    root = _root(arm)
    method = "eoh" if arm == "E" else "hifo"
    paths = ["utils/getParas.py", f"methods/{method}/{method}.py",
             f"methods/{method}/{method}_interface_EC.py", f"methods/{method}/{method}_evolution.py",
             "methods/selection/prob_rank.py", "methods/management/pop_greedy.py",
             "problems/optimization/tsp_greedy/prompts.py"]
    if arm == "H":
        paths += ["methods/hifo/insight_pool.py", "methods/hifo/evolutionary_navigator.py"]
    return {"arm": arm, "upstream_commit": SOURCE_PINS[arm], "defaults": native_defaults(arm),
            "files": [{"path": str((root / p).relative_to(REPO)),
                       "sha256": digest((root / p).read_bytes())} for p in paths],
            "adaptations": [
                "external per-run Python/NumPy RNG replaces fixed seed 2024",
                "durable common API replaces constructor connectivity probe and native API",
                "common complete-module appendix plus faithful whole-response extraction",
                "no parser regeneration, return synthesis, import injection, numba rewrite or algorithm repair",
                "native duplicate regeneration remains a NEW main candidate position, never a response choice within one position",
                "native batch population remains frozen; candidates execute sequentially at native exp_n_proc=1",
                "remote isolated common evaluator replaces local ThreadPool candidate execution",
                "native ec_n_pop stopping horizon is replaced by frozen main-position budget including initialization",
                "all valid complete source artifacts enter Search archive; final selection is external Val after Search lock",
                "prefix costs include maintenance already due at that position; unfinished native batches preserve frozen parents",
            ]}


def _native_parents(parents):
    if parents is None:
        return []
    if isinstance(parents, dict):
        parents = [parents]
    return [{"hash": digest(p.get("source", p.get("code"))),
             "algorithm": p.get("algorithm") or "",
             "source": p.get("source", p.get("code")), "objective": p.get("objective")}
            for p in parents]


class BaselineIO:
    """Shared budget/transport/evaluation adapter for complete native methods.

    ``search`` is an initialized common Search object, not an already run search.
    ``generate`` returns a native individual; ``after_parse`` may modify only its
    natural-language algorithm/metadata, never its source. Auxiliary responses
    are raw text and consume real costs without consuming a main position.
    """
    def __init__(self, search, budget=192, prefixes=PREFIXES):
        self.search = search
        self.budget = int(budget)
        if self.budget <= 0:
            raise ValueError("POSITIVE_BUDGET_REQUIRED")
        self.prefixes = tuple(sorted(set(int(p) for p in prefixes if p <= budget)))
        self.position = 0
        self.pending_prefix = None
        self.prefix_results = {}
        self.last_individual = None
        self.last_receipt = None
        self._pending_position = None
        self.search.controllers = {}
        self.search.state["controllers"] = {}
        self.search.state["native_method"] = search.arm
        self.search.state.setdefault("native_auxiliary", [])

    def ensure_reference(self):
        from .state import REFERENCE_SOURCE
        self.search.save_program(REFERENCE_SOURCE)
        quality = self.search.evaluate(REFERENCE_SOURCE, "reference")
        if not quality["complete"]:
            raise RuntimeError("NAIVE_REFERENCE_NOT_FEASIBLE")
        self.search.state["scale"] = quality["mean"]
        return quality

    def auxiliary(self, prompt, kind, parents=None, metadata=None):
        parent_records = _native_parents(parents)
        position = self._pending_position or self.position
        meta = dict(kind=kind, position=position, native_auxiliary=True,
                    parents=[p["hash"] for p in parent_records], **(metadata or {}))
        result = self.search.transport(prompt, meta)
        record = dict(kind=kind, position=position, prompt_sha256=digest(prompt),
                      parents=parent_records, response_available=result is not None)
        if getattr(self.search.transport, "records", None):
            record["terminal"] = self.search.transport.records[-1]
        self.search.state["native_auxiliary"].append(record)
        return result

    def generate(self, prompt, operator, parents=None, metadata=None, *,
                 defer_prefix=False, after_parse=None, append_module_format=True):
        from .state import register_program
        if self.position >= self.budget:
            raise BudgetComplete()
        if self.pending_prefix is not None:
            raise RuntimeError("PENDING_PREFIX_MAINTENANCE_NOT_FINISHED")
        position = self.position + 1
        self._pending_position = position
        parent_records = _native_parents(parents)
        parent_hashes = [p["hash"] for p in parent_records]
        prompt = prompt + MODULE_FORMAT if append_module_format else prompt
        folder = self.search.root / "positions" / f"native-{position:03d}"
        pre = dict(position=position, operator=operator, parents=parent_records,
                   metadata=metadata or {}, prompt_sha256=digest(prompt),
                   native_method=self.search.arm, candidate_unknown=True,
                   archive_hashes=sorted(self.search.state["programs"]))
        freeze(folder / "PRE.json", pre)
        initialization = operator == "i1" or (metadata or {}).get("native_phase") == "initialization"
        response = self.search.transport(prompt, dict(kind="initialization" if initialization else "generation",
            position=position, operator=operator, parents=parent_hashes, native_method=self.search.arm,
            predecision_sha256=digest(encoded(pre))))
        parsed = extract_response(response) if response is not None else None
        source = parsed.source if parsed is not None else None
        error, quality = None, None
        individual = dict(algorithm=parsed.description if parsed is not None else "",
                          code=source, objective=None, other_inf=None)
        if parsed is not None:
            freeze(folder / "PARSE.json", parsed.evidence())
            if source is not None:
                freeze_bytes(folder / "response_source.py", source.encode("utf-8"))
        if source is not None:
            try:
                ast.parse(source)
            except (ValueError, SyntaxError) as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
            else:
                if after_parse is not None:
                    after_parse(individual)
                    if individual["code"] != source:
                        raise RuntimeError("NATIVE_ALIGNMENT_CANNOT_REWRITE_SOURCE")
                self.search.save_program(source)
                quality = self.search.evaluate(source, f"native-{position:03d}")
        valid = bool(quality and quality["complete"])
        h = digest(source) if source is not None else None
        seen = h in self.search.state["programs"] if h else False
        if valid:
            register_program(self.search.state, source, quality, parent_hashes,
                             f"{self.search.arm}_{operator}", position, individual["algorithm"])
            # Actual native E/H management uses five-decimal objectives. Raw
            # full quality remains in archive/results for all final selection.
            import numpy as np
            individual["objective"] = float(np.round(quality["mean"], 5))
        else:
            individual["code"] = None
        individual["_quality_result"] = quality
        individual["_source_hash"] = h
        individual["_evaluation_timeout"] = bool(quality and (
            quality.get("timed_out") or quality.get("timeout") or
            any(any(tag in str(row.get(field, "")).upper()
                    for tag in ("TIMEOUT", "DEADLINE") for field in ("status", "error_type"))
                for row in quality.get("rows", []))))
        receipt = dict(position=position, op=operator, parents=parent_hashes, source_hash=h,
                       valid=valid, candidate_quality=quality["mean"] if valid else None,
                       native_rounded_objective=individual["objective"], error=error,
                       response_available=response is not None,
                       parser_status=parsed.status if parsed else None,
                       evaluation=quality, restores_seen_program=seen,
                       new_source=bool(h and not seen), occurrence_parents=parent_hashes,
                       archive_lineage_retained=seen,
                       original_algorithm=parsed.description if parsed else None,
                       returned_algorithm=individual["algorithm"], other_inf=individual["other_inf"],
                       postparse_metadata={k: v for k, v in individual.items()
                           if k not in {"algorithm", "code", "objective", "other_inf", "_quality_result"}})
        if not valid:
            self.search.state["failures"].append(receipt)
        if initialization:
            self.search.state["init_completed"] += 1
        else:
            self.search.state["noninit_completed"] += 1
        self.position = position
        self._pending_position = None
        self.last_individual, self.last_receipt = individual, receipt
        freeze(folder / "RESULT.json", receipt)
        if position in self.prefixes:
            if defer_prefix:
                self.pending_prefix = position
            else:
                self.write_prefix(position)
        return individual

    def write_prefix(self, budget, *, status="PREFIX_COMPLETE"):
        from .search import cost_snapshot
        from .state import REFERENCE_SOURCE
        path = self.search.root / "prefixes" / f"{budget:03d}" / "PREFIX.json"
        if path.exists():
            result = read(path)
            if result["archive_hashes"] != sorted(self.search.state["programs"]):
                raise RuntimeError("REPLAY_PREFIX_ARCHIVE_CONFLICT")
            self.prefix_results[str(budget)] = result
            return result
        self.search.checkpoint(f"prefix-{budget:03d}")
        cost = cost_snapshot(self.search.root)
        cost["wall_seconds"] = time.time() - self.search.started
        result = dict(budget=budget, completed_positions=self.position, status=status,
                      archive_hashes=sorted(self.search.state["programs"]),
                      fallback_program_hash=(digest(REFERENCE_SOURCE) if not self.search.state["programs"] else None),
                      state_path=str(self.search.root / "states" / f"prefix-{budget:03d}.json"),
                      cost=cost, search_locked=True)
        freeze(path, result)
        self.prefix_results[str(budget)] = result
        return result

    def end_batch(self, population, native_state=None):
        self.search.state["native_population"] = copy.deepcopy(population)
        if native_state is not None:
            self.search.state["native_state"] = copy.deepcopy(native_state)
        if self.pending_prefix is not None:
            point = self.pending_prefix
            self.pending_prefix = None
            self.write_prefix(point)

    def finish(self, population, native_state=None, *, status="NATIVE_BUDGET_COMPLETE"):
        from .search import cost_snapshot
        from .state import REFERENCE_SOURCE, register_program
        if not self.search.state["programs"]:
            h = digest(REFERENCE_SOURCE)
            register_program(self.search.state, REFERENCE_SOURCE, self.search.state["qualities"][h],
                             [], "all_initialization_failed_reference", 0)
        self.end_batch(population, native_state)
        for point in self.prefixes:
            if str(point) not in self.prefix_results:
                if self.position < point:
                    self.write_prefix(point, status="NATIVE_INITIALIZATION_FAILED_FALLBACK")
                else:
                    raise RuntimeError("NATIVE_PREFIX_WAS_NOT_CAPTURED")
        self.search.checkpoint("FINAL")
        path = self.search.root / "SEARCH_RESULT.json"
        if path.exists():
            return read(path)
        best = min(self.search.state["programs"].values(),
                   key=lambda p: (p["mean"], p["first_position"], p["hash"]))
        cost = cost_snapshot(self.search.root)
        cost["wall_seconds"] = time.time() - self.search.started
        result = dict(study_id=self.search.state["context"].get("study_id", "fuse-credit-action-v1-20260926"),
                      phase=self.search.phase, group=self.search.group, arm=self.search.arm,
                      search_best_hash=best["hash"], search_best_mean=best["mean"],
                      planned_positions=self.budget, completed_positions=self.position,
                      unattempted_positions=self.budget - self.position, termination=status,
                      init_completed=self.search.state["init_completed"],
                      noninit_completed=self.search.state["noninit_completed"], cost=cost,
                      own_physical_cost=cost, common_cost_charged_in_full=False,
                      state_path=str(self.search.root / "states/FINAL.json"),
                      program_count=len(self.search.state["programs"]), credit_count=0,
                      failure_count=len(self.search.state["failures"]), prefixes=self.prefix_results,
                      search_locked=True)
        return freeze(path, result)


class _NumpyProxy:
    def __init__(self, module, rng):
        self.module, self.random = module, rng

    def __getattr__(self, name):
        return getattr(self.module, name)


def _native_modules(arm, namespace):
    root = _root(arm)
    spec = importlib.util.spec_from_file_location(namespace, root / "__init__.py",
                                                 submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    spec.loader.exec_module(package)
    method = "eoh" if arm == "E" else "hifo"
    modules = {name: importlib.import_module(namespace + "." + suffix) for name, suffix in {
        "engine": f"methods.{method}.{method}", "interface": f"methods.{method}.{method}_interface_EC",
        "evolution": f"methods.{method}.{method}_evolution", "parameters": "utils.getParas",
        "select": "methods.selection.prob_rank", "manage": "methods.management.pop_greedy",
        "prompts": "problems.optimization.tsp_greedy.prompts",
    }.items()}
    if arm == "H":
        modules["insight"] = importlib.import_module(namespace + ".methods.hifo.insight_pool")
        modules["navigator"] = importlib.import_module(namespace + ".methods.hifo.evolutionary_navigator")
    return modules


def run_native_baseline(search, budget=192, prefixes=PREFIXES):
    """Run full pinned E/H in its own process, with root Search I/O adapters."""
    import numpy as np
    arm = search.arm
    if arm not in ("E", "H"):
        raise ValueError("E_OR_H_REQUIRED")
    io = BaselineIO(search, budget, prefixes)
    io.ensure_reference()
    modules = _native_modules(arm, "_credit_native_" + arm + "_" + digest(str(search.root))[:12])
    hm, im, ev = (modules[name] for name in ("engine", "interface", "evolution"))
    paras = modules["parameters"].Paras()
    paras.set_paras()
    native_n_pop = paras.ec_n_pop
    paras.ec_n_pop = max(1, math.ceil(max(0, budget - 2 * paras.ec_pop_size) /
                                     (paras.ec_pop_size * len(paras.ec_operators))))
    paras.exp_output_path = str(search.root / "native")
    for folder in ("results/pops", "results/pops_best"):
        (Path(paras.exp_output_path) / folder).mkdir(parents=True, exist_ok=True)
    paras.llm_api_endpoint, paras.llm_api_key = "durable-common-transport", "not-a-credential"
    paras.llm_model = getattr(search.transport, "binding", {}).get("model_settings", {}).get("model")
    rng = random.Random(search.rng("native-python", 0))
    nrng = np.random.RandomState(search.rng("native-numpy", 0) % (2 ** 32))
    proxy = _NumpyProxy(np, nrng)
    owner = types.SimpleNamespace(current=None, interface=None, population=[], defer=False, batches=0)
    Base = im.InterfaceEC

    def snapshot():
        ec = owner.interface
        if arm == "E" or ec is None:
            return {"completed_native_batches": owner.batches, "rng_state": rng.getstate(),
                    "numpy_rng_state": [nrng.get_state()[0], nrng.get_state()[1].tolist(),
                                        *nrng.get_state()[2:]]}
        return dict(completed_native_batches=owner.batches,
                    best_fitness_history=ec.best_fitness_history, avg_fitness_history=ec.avg_fitness_history,
                    diversity_history=ec.diversity_history, insight_tips=list(ec.insight_pool.tips),
                    insight_stats=ec.insight_pool.tip_stats,
                    insight_generation=ec.insight_pool.current_generation,
                    navigator=vars(ec.navigator), rng_state=rng.getstate(),
                    numpy_rng_state=[nrng.get_state()[0], nrng.get_state()[1].tolist(), *nrng.get_state()[2:]])

    class Bridge:
        def __init__(self, *args, **kwargs):
            pass  # Native constructor connectivity probe is not a generation.

        def get_response(self, prompt):
            try:
                return io.auxiliary(prompt, "insight_extraction", owner.population)
            except Exception as exc:
                raise _InfrastructureStop(exc) from exc

    def extract_once(evolution, prompt):
        current = owner.current
        individual = io.generate(prompt, current["operator"], current["parents"],
                                 current["metadata"], defer_prefix=True)
        return [individual["code"], individual["algorithm"]]

    class HookedEC(Base):
        def __init__(ec, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owner.interface = ec
            for name in ("i1", *paras.ec_operators):
                original = getattr(ec.evol, name)

                def dispatch(*args, _name=name, _original=original, **kwargs):
                    owner.current["parents"] = None if _name == "i1" else args[0]
                    if arm == "H":
                        offset = 0 if _name == "i1" else 1
                        owner.current["metadata"] = {
                            key: copy.deepcopy(args[offset + j] if len(args) > offset + j else kwargs.get(key))
                            for j, key in enumerate(("insights", "design_directive", "regime"))}
                    return _original(*args, **kwargs)
                setattr(ec.evol, name, dispatch)

        def get_offspring(ec, pop, operator):
            owner.current = dict(operator=operator, parents=[], metadata={})
            def native_attempt():
                parents, offspring = ec._get_alg(pop, operator)
                if arm == "H" and "timestamp" in offspring.get("metadata", {}):
                    path = search.root / "positions" / f"native-{io.position:03d}" / "NATIVE_METADATA.json"
                    if path.exists():
                        offspring["metadata"]["timestamp"] = read(path)["timestamp"]
                    freeze(path, offspring["metadata"])
                return parents, offspring
            parents, off = native_attempt()
            # The pinned native implementation regenerates a duplicate once.
            # It is a fresh recorded candidate position, not an extraction retry.
            if (off["code"] is not None and ec.check_duplicate(pop, off["code"])
                    and io.position < io.budget):
                # A prefix reached by the first attempt has no batch-end
                # maintenance due yet if a native duplicate attempt follows.
                if io.pending_prefix is not None:
                    io.end_batch(pop, snapshot())
                parents, off = native_attempt()
            evaluated = io.last_individual
            off.update({key: evaluated[key] for key in ("code", "algorithm", "objective", "other_inf")})
            if arm == "H" and off["objective"] is not None:
                ec.update_insight_feedback(off, pop)
            if io.pending_prefix is not None and not owner.defer and io.position < io.budget:
                io.end_batch(pop, snapshot())
            return parents, off

        def get_algorithm(ec, pop, operator):
            if io.position >= io.budget:
                raise BudgetComplete()
            owner.population = pop
            if arm == "H":
                ec.insight_pool.update_generation(len(ec.best_fitness_history))
            results = []
            for index in range(ec.pop_size):
                if io.position >= io.budget:
                    break
                owner.defer = index == ec.pop_size - 1 or io.position + 1 == io.budget
                results.append(ec.get_offspring(pop, operator))
            time.sleep(2)  # Existing upstream inter-batch scheduling delay.
            if arm == "H" and pop:
                ec.update_population_metrics(pop)
                if nrng.random() < 0.8:
                    ec.extract_insights_from_population(pop)
            owner.batches += 1
            if operator == "i1":
                # Native population_generation concatenates both i1 batches
                # before management, so initialization prefixes contain an
                # unfinished native population but all completed archive rows.
                io.end_batch(pop, snapshot())
            # Native admission remains after the entire batch. Prefix capture
            # occurs in managed() after that admission if it was batch-final.
            return [p for p, _ in results], [off for _, off in results]

    original_manage = modules["manage"].population_management

    def managed(pop, size):
        result = original_manage(pop, size)
        owner.population = result
        io.end_batch(result, snapshot())
        if not result:
            raise _InitializationFailed()
        return result

    freeze(search.root / "NATIVE_SOURCE_CONFIG.json", dict(source_manifest(arm),
           actual_horizon_generations=paras.ec_n_pop, original_horizon_generations=native_n_pop,
           main_positions=budget, prefixes=list(prefixes), actual_parameters=vars(paras),
           common_evaluator_search_seconds=500, native_evaluator_timeout_replaced=paras.eva_timeout,
           rng_seed_python=search.rng("native-python", 0), rng_seed_numpy=search.rng("native-numpy", 0)))
    status = "NATIVE_BUDGET_COMPLETE"
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(ev, "InterfaceLLM", Bridge))
            stack.enter_context(patch.object(ev.Evolution, "_get_alg", extract_once))
            stack.enter_context(patch.object(hm, "InterfaceEC", HookedEC))
            stack.enter_context(patch.object(modules["manage"], "population_management", managed))
            for module in modules.values():
                if hasattr(module, "random"):
                    stack.enter_context(patch.object(module, "random", rng))
                if hasattr(module, "np"):
                    stack.enter_context(patch.object(module, "np", proxy))
            engine_class = hm.EOH if arm == "E" else hm.HiFo
            problem = types.SimpleNamespace(prompts=modules["prompts"].GetPrompts())
            engine = engine_class(paras, problem, modules["select"], modules["manage"])
            rng.seed(search.rng("native-python", 0))  # Replace constructor's fixed seed.
            try:
                engine.run()
            except BudgetComplete:
                pass
    except _InitializationFailed:
        status = "NATIVE_INITIALIZATION_FAILED"
    except _InfrastructureStop as exc:
        raise exc.cause
    if status == "NATIVE_BUDGET_COMPLETE" and io.position != budget:
        raise RuntimeError("NATIVE_HORIZON_ENDED_BEFORE_FROZEN_BUDGET")
    return io.finish(owner.population, snapshot(), status=status)
