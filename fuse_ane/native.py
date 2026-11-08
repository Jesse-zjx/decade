"""Pinned full H/E, with one physical HiFo initialization and exact state handoff.

Native engine, operators, selection, management, insight/navigation are reused.
Only common I/O, faithful full-source extraction and paid budget boundaries differ.
"""
import copy,random
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from fuse_addon.evaluation import freeze,freeze_bytes,digest
from fuse_credit import baselines as legacy
from fuse_credit.baselines import _NumpyProxy
from fuse_patch.native import _CVRPPrompts
from fuse_patch.budget import DispatchStop,selection
from .budget import Boundary
from .data import TSP_NN,CVRP_NN

class NativeRandom:
    def __init__(self,g):self.generator=g
    def rand(self,*shape):return self.generator.random(shape or None)
    def __getattr__(self,k):return getattr(self.generator,k)

def _tuple(value):return tuple(_tuple(x) for x in value) if isinstance(value,(list,tuple)) else value

def reference(io):
    source=TSP_NN if io.task=='tsp' else CVRP_NN;q=io.evaluate(source,'public-nearest-neighbor')
    if not q['complete']:raise RuntimeError('PUBLIC_REFERENCE_FAILED')
    h=io.register_program(source,q,[],'public_nearest_neighbor',0)
    io.reference=dict(source=source,source_hash=h,mean=q['mean'],quality=q)
    io.state['scale']=q['mean']


def run_native(io,shared=None,initialization_only=False):
    arm='H' if initialization_only else io.arm
    if shared:io.attach_shared(shared)
    elif not io.reference:reference(io)
    boundary=Boundary(io);boundary.ensure_reference()
    if shared:boundary.inherit_prefixes(shared)
    modules=legacy._native_modules(arm,'_ane_native_'+arm+'_'+digest(str(io.root))[:12])
    hm,im,evolution=(modules[n] for n in ('engine','interface','evolution'))
    paras=modules['parameters'].Paras();paras.set_paras(ec_pop_size=8,ec_n_pop=8,exp_n_proc=1)
    paras.llm_api_endpoint='durable-authorized-provider';paras.llm_api_key='not-a-credential';paras.llm_model='DeepSeek-V4-Flash'
    paras.exp_output_path=str(io.root/'native');paras.eva_numba_decorator=False
    for f in ('results/pops','results/pops_best'):(io.root/'native'/f).mkdir(parents=True,exist_ok=True)
    rng=random.Random(io.rng('native-python',0));nrng=NativeRandom(np.random.Generator(np.random.PCG64(io.rng('native-numpy',0))))
    proxy=_NumpyProxy(np,nrng);owner=SimpleNamespace(current=None,interface=None,population=[],batches=0,generated=[])
    Base=im.InterfaceEC
    def snapshot():
        ec=owner.interface
        value=dict(python_rng=rng.getstate(),numpy_rng=copy.deepcopy(nrng.generator.bit_generator.state),completed_native_batches=owner.batches)
        if arm=='H' and ec is not None:
            value.update(best_fitness_history=ec.best_fitness_history,avg_fitness_history=ec.avg_fitness_history,diversity_history=ec.diversity_history,
                insight_tips=list(ec.insight_pool.tips),insight_stats=ec.insight_pool.tip_stats,insight_generation=ec.insight_pool.current_generation,navigator=vars(ec.navigator))
        return copy.deepcopy(value)
    class Bridge:
        def __init__(self,*args,**kwargs):pass
        def get_response(self,prompt):
            try:return boundary.auxiliary(prompt,'insight_extraction',owner.population)
            except Exception as exc:raise legacy._InfrastructureStop(exc) from exc
    def extract(evol,prompt):
        current=owner.current
        # Upstream initial + four finite parse retries, each a new paid request.
        for parse_attempt in range(5):
            individual=boundary.generate(prompt,current['operator'],current['parents'],dict(current['metadata'],native_parse_attempt=parse_attempt))
            if boundary.last_receipt['parser_status']=='EXTRACTED':return [individual['code'],individual['algorithm']]
        return [None,'']
    class HookedEC(Base):
        def __init__(ec,*args,**kwargs):
            super().__init__(*args,**kwargs);owner.interface=ec
            if shared and arm=='H':
                snap=shared['native_state']
                for key in ('best_fitness_history','avg_fitness_history','diversity_history'):setattr(ec,key,copy.deepcopy(snap[key]))
                ec.insight_pool.tips.clear();ec.insight_pool.tips.extend(snap['insight_tips']);ec.insight_pool.tip_stats=copy.deepcopy(snap['insight_stats']);ec.insight_pool.current_generation=snap['insight_generation']
                for k,v in snap['navigator'].items():setattr(ec.navigator,k,copy.deepcopy(v))
                rng.setstate(_tuple(snap['python_rng']));nrng.generator.bit_generator.state=copy.deepcopy(snap['numpy_rng'])
            for name in ('i1',*paras.ec_operators):
                original=getattr(ec.evol,name)
                def dispatch(*args,_name=name,_original=original,**kwargs):
                    owner.current['parents']=None if _name=='i1' else args[0]
                    if arm=='H':
                        offset=0 if _name=='i1' else 1
                        owner.current['metadata']={key:copy.deepcopy(args[offset+j] if len(args)>offset+j else kwargs.get(key)) for j,key in enumerate(('insights','design_directive','regime'))}
                    return _original(*args,**kwargs)
                setattr(ec.evol,name,dispatch)
        def get_offspring(ec,pop,operator):
            owner.current=dict(operator=operator,parents=[],metadata={})
            parents,off=ec._get_alg(pop,operator)
            if off['code'] is not None and ec.check_duplicate(pop,off['code']):
                boundary.before_dispatch(generation=True);parents,off=ec._get_alg(pop,operator)
            evaluated=boundary.last_individual
            off.update({k:evaluated[k] for k in ('code','algorithm','objective','other_inf')})
            if off['objective'] is None:
                # Upstream's exception branch discards metadata before its penalty test.
                off=dict(algorithm=None,code=None,objective=None,other_inf=None);parents=None
            elif arm=='H':ec.update_insight_feedback(off,pop)
            owner.generated.append(copy.deepcopy(off))
            return parents,off
        def get_algorithm(ec,pop,operator):
            boundary.before_dispatch();owner.population=pop
            if arm=='H':ec.insight_pool.update_generation(len(ec.best_fitness_history))
            results=[]
            for _ in range(ec.pop_size):
                boundary.before_dispatch(generation=True);results.append(ec.get_offspring(pop,operator))
            if arm=='H' and pop:
                ec.update_population_metrics(pop)
                if nrng.random()<.8:ec.extract_insights_from_population(pop)
            owner.batches+=1
            boundary.end_batch(pop,snapshot())
            return [p for p,_ in results],[o for _,o in results]
    original_manage=modules['manage'].population_management
    def managed(pop,size):
        result=original_manage(pop,size);owner.population=result;boundary.end_batch(result,snapshot())
        if not result and not initialization_only:raise DispatchStop('NATIVE_EMPTY_POPULATION')
        return result
    spec={k:v for k,v in legacy.source_manifest(arm).items() if k!='adaptations'}
    freeze(io.root/'NATIVE_SOURCE_CONFIG.json',dict(upstream=spec,task=io.task,population_size=8,horizon=8,n_proc=1,
        operators=paras.ec_operators,unified_hifo_initialization=True,faithful_parser='fuse_addon.response_parser.extract_response',
        native_parse_attempt_ceiling=5,native_duplicate_additional_attempt_ceiling=1,full_current_instance_API=True,
        changes=['shared trusted evaluator and bounded reconcile-first sender','unchanged whole module extraction with original description','no synthetic return/import/numba rewrite','all native retries/insight requests charged','native sleep removed','new independent RNG','same HiFo initial archive/state; E uses same valid archive'],
        future_insight_shared=False))
    status='NATIVE_HORIZON_EXHAUSTED';init_ok=False
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(evolution,'InterfaceLLM',Bridge));stack.enter_context(patch.object(evolution.Evolution,'_get_alg',extract))
            stack.enter_context(patch.object(hm,'InterfaceEC',HookedEC));stack.enter_context(patch.object(modules['manage'],'population_management',managed))
            for m in modules.values():
                if hasattr(m,'random'):stack.enter_context(patch.object(m,'random',rng))
                if hasattr(m,'np'):stack.enter_context(patch.object(m,'np',proxy))
            prompts=modules['prompts'].GetPrompts() if io.task=='tsp' else _CVRPPrompts()
            problem=SimpleNamespace(prompts=prompts)
            if initialization_only:
                ec=HookedEC(paras.ec_pop_size,paras.ec_m,paras.llm_api_endpoint,paras.llm_api_key,paras.llm_model,False,None,False,problem,
                    select=modules['select'],n_p=1,timeout=paras.eva_timeout,use_numba=False)
                pop=ec.population_generation();owner.population=managed(pop,8);init_ok=bool(owner.population);status='COMMON_HIFO_INITIALIZATION_COMPLETE'
            else:
                population=copy.deepcopy(shared['population']) if arm=='H' else original_manage([
                    dict(code=io.source(h),algorithm=r.get('algorithm',''),objective=float(np.round(r['mean'],5)),other_inf=None)
                    for h,r in shared['archive_state']['programs'].items() if r['origin']!='public_nearest_neighbor'],8)
                if not population:
                    r=io.reference;population=[dict(code=r['source'],algorithm='Public feasible nearest neighbor fallback after failed initialization',objective=float(np.round(r['mean'],5)),other_inf=None)]
                owner.population=population
                freeze(io.root/'native'/'INITIAL_POPULATION.json',population)
                paras.exp_use_continue=True;paras.exp_continue_path=str(io.root/'native'/'INITIAL_POPULATION.json');paras.exp_continue_id=0
                engine=(hm.HiFo if arm=='H' else hm.EOH)(paras,problem,modules['select'],modules['manage'])
                rng.seed(io.rng('native-python',0));engine.run()
    except DispatchStop as exc:status=exc.reason
    except legacy._InfrastructureStop as exc:raise exc.cause
    if initialization_only:
        # A budget stop retains every generated/paid valid initial program, never fills for free.
        if not owner.population:owner.population=original_manage(owner.generated,8)
        result=boundary.finish(owner.population,snapshot(),status=status)
        shared=dict(init_root=str(io.root),phase=io.phase,task=io.task,group=io.group,archive_state=copy.deepcopy(io.state),reference=copy.deepcopy(io.reference),
            request_set=result['request_set'],cost=io.own_cost_snapshot(),prefixes=result['prefixes'],selection=selection(io),population=copy.deepcopy(owner.population),native_state=snapshot(),
            initialization_complete=status=='COMMON_HIFO_INITIALIZATION_COMPLETE',initialization_failed=not bool(owner.population))
        freeze(io.root/'COMMON_INITIALIZATION.json',shared);return shared
    return boundary.finish(owner.population,snapshot(),status=status)
