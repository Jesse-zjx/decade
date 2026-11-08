"""V3 isolated runtime. Reuses the durable V2 sender and TSP evaluator.

New namespace, byte-exact source loading, no historical/common-result import,
logical replay cost frontier, and strict current-study stage barriers.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from fuse_addon import evaluation as ev, task_data, transport as tr
from fuse_paired import runtime as prior
from fuse_paired.runtime import PairedEvaluator
from fuse_paired.full_runtime import FullTransport, _number, _usage, _file_within

STUDY_ID = 'fuse-forward-continuation-diagnostic-20260928'
REMOTE_ROOT = Path('/opt/decade/patch')
CANDIDATE_CPUS = ('20', '22', '24', '26')
CONTROLLER_CPUS = (28, 30, 32, 34)
PHASE_GROUPS = {'dev': 8, 'confirm': 20, 'cvrp': 12}
ARMS = ('J','A','W','R','E','H')
TASK = 'TSP-Construct'
MODEL_SETTINGS = prior.MODEL_SETTINGS
NN_ANCHOR_SOURCE = prior.NN_ANCHOR_SOURCE
PANEL_SHAPES = {'Search32':(32,50),'Val128':(128,50),'Test50':(256,50),'Test100':(100,100),'Test200':(100,200)}

def seed_for(phase, task, group, role, stream):
    return int.from_bytes(hashlib.sha256(f'{STUDY_ID}|{phase}|{group}|{role}|{stream}'.encode()).digest()[:4], 'big')

def _require_remote(root):
    root = Path(root).resolve()
    if platform.system() != 'Linux' or not root.is_relative_to(REMOTE_ROOT):
        raise RuntimeError('SCIENTIFIC_IO_ONLY_IN_AUTHORIZED_V3_REMOTE_CHECKOUT')
    if not Path(ev.RUNTIME+'/bin/python').is_file():
        raise RuntimeError('AUTHORIZED_REMOTE_RUNTIME_MISSING')
    study = REMOTE_ROOT/'runs'/STUDY_ID
    if not (study/'authorization/EXECUTION_FREEZE.json').is_file():
        raise RuntimeError('V3_SCIENTIFIC_FREEZE_REQUIRED')
    if (study/'RESULT_SEALED.json').exists():
        raise RuntimeError('V3_RESULT_SEALED_FUTURE_DISPATCH_FALSE')

def make_panel(path, *, phase, group, role, sampler_path=None):
    if phase not in PHASE_GROUPS or group not in range(PHASE_GROUPS[phase]):
        raise ValueError('UNREGISTERED_GROUP')
    if phase == 'cvrp':
        return make_cvrp_panel(path,phase=phase,group=group,role=role)
    allowed = ('Search32','Val128') if phase == 'dev' else ('Search32','Test50','Test100','Test200')
    if role not in allowed:
        raise ValueError('UNREGISTERED_PHASE_PANEL')
    count,cities = PANEL_SHAPES[role]
    path = Path(path).resolve()
    identity = dict(study_id=STUDY_ID,phase=phase,stage=phase,group=str(group),role=role,count=count,cities=cities,task=TASK)
    limits = prior.panel_limits(role)
    if path.exists():
        panel=ev.read(path)
        if panel['identity'] != identity or panel['limits'] != limits:
            raise RuntimeError('PANEL_IDENTITY_CONFLICT')
        return task_data.panel_manifest(path,panel)
    seed = seed_for(phase,TASK,group,role,'data')
    data,witness = task_data.sample_tsp(count,cities,seed,sampler_path)
    items=[dict(instance_id=f'{phase}/{group}/{role}/{i:03d}',coordinates=c.tolist(),distance_matrix=d.tolist(),
        execution_seed=seed_for(phase,TASK,group,role,f'execution/{i:03d}')) for i,(c,d) in enumerate(data)]
    panel=dict(identity=identity,task=TASK,role=role,seed=seed,instances=items,limits=limits,sampler=witness,
        seed_derivation=f'{STUDY_ID}|{phase}|{group}|{role}|data',
        seed_conversion='SHA256 UTF8 fields joined by |; first4 bytes unsigned big endian',
        execution_seed_derivation='study_id|phase|group|role|execution/NNN',
        quality_access='SEARCH' if role=='Search32' else 'AFTER_EXACT_STAGE_OUTPUT_LOCKS',
        independent_of_old_quality=True,optimal_tour_reference=None,
        gap_status='MISSING_NO_FROZEN_OPTIMAL_REFERENCE_FOR_NEW_INSTANCES')
    ev.freeze(path,panel)
    return task_data.panel_manifest(path,panel)


def reference_source(phase):
    if phase=='cvrp':
        from .cvrp_contract import NN_ANCHOR_SOURCE as source
        return source
    return NN_ANCHOR_SOURCE


def make_cvrp_panel(path,*,phase,group,role):
    from . import cvrp_contract as contract
    if phase!='cvrp' or group not in range(12) or role not in contract.PANEL_SHAPES:
        raise ValueError('UNREGISTERED_CVRP_PANEL')
    count,customers=contract.PANEL_SHAPES[role];path=Path(path).resolve()
    identity=dict(study_id=STUDY_ID,phase=phase,stage=phase,group=str(group),role=role,count=count,
        customers=customers,cities=customers+1,task=contract.TASK)
    limits=prior.panel_limits(role)
    if path.exists():
        panel=ev.read(path)
        if panel['identity']!=identity or panel['limits']!=limits:raise RuntimeError('CVRP_PANEL_IDENTITY_CONFLICT')
        return task_data.panel_manifest(path,panel)
    seed=seed_for(phase,contract.TASK,group,role,'data')
    items,witness=contract.make_instances(count,customers,seed,lambda i:seed_for(phase,contract.TASK,group,role,f'execution/{i:03d}'))
    for i,item in enumerate(items):item.update(instance_id=f'{phase}/{group}/{role}/{i:03d}',task=contract.TASK)
    panel=dict(identity=identity,task=contract.TASK,role=role,seed=seed,instances=items,limits=limits,sampler=witness,
        seed_derivation=f'{STUDY_ID}|{phase}|{group}|{role}|data',
        seed_conversion='SHA256 UTF8 fields joined by |; first4 bytes unsigned big endian',
        quality_access='SEARCH' if role=='Search32' else 'AFTER_ALL_12_GROUP_SEARCH_LOCKS',
        optimal_tour_reference=None,gap_status='MISSING_NO_FROZEN_OPTIMAL_REFERENCE_FOR_NEW_INSTANCES')
    ev.freeze(path,panel);return task_data.panel_manifest(path,panel)


def aggregate_worker(events,panel,**kwargs):
    if panel['task']=='TSP-Construct':return ev.aggregate_worker(events,panel,**kwargs)
    if panel['task']!='CVRP-Construct':raise ValueError('UNREGISTERED_AGGREGATE_TASK')
    return aggregate_cvrp_worker(events,panel,**kwargs)


def aggregate_cvrp_worker(events, panel, *, source_hash, raw_path, container_state=None,
                     infrastructure_error=None, active_seconds=None):
    starts = [x for x in events if x.get('type') == 'INSTANCE_STARTED']
    observed = {x['instance_id']: x for x in events if x.get('type') == 'ROUTE_RESULT'}
    terminal = next((x for x in reversed(events) if x.get('type') == 'WORKER_RESULT'), None)
    rows = [observed.get(x['instance_id'], dict(instance_id=x['instance_id'], status='NOT_EXECUTED',
                                              length=None, route_seconds=None, load_seconds=None))
            for x in panel['instances']]
    integrity_errors = []
    for row, item in zip(rows, panel['instances']):
        if row.get('status') != 'EVALUATED':
            continue
        from .cvrp_contract import validate_route
        try:
            facts=validate_route(row,item)
        except ValueError as exc:
            integrity_errors.append(dict(instance_id=item['instance_id'],reason=str(exc)))
        else:
            row['host_independent_length']=facts['host_independent_length']
            row['host_vehicle_loads']=facts['vehicle_loads']
    if integrity_errors:
        infrastructure_error = 'WORKER_ROUTE_INTEGRITY_CONFLICT'
    complete = bool(terminal and terminal.get('complete') and not infrastructure_error and
                    all(x.get('status') == 'EVALUATED' and isinstance(x.get('length'), (int, float))
                        and math.isfinite(x['length']) for x in rows))
    import numpy as np
    mean = float(np.average([x['length'] for x in rows])) if complete else None
    candidate_resource_failure = infrastructure_error == 'CANDIDATE_CONTAINER_OOM'
    unknown = (terminal is None or infrastructure_error is not None) and not candidate_resource_failure
    # A worker-terminal candidate failure is a real observed failure, not host
    # uncertainty. A partial or absent raw terminal cannot be silently retried.
    return dict(complete=complete, mean=mean, L=mean, objective=mean, minimizing=True,
                status='EVALUATED' if complete else 'UNKNOWN' if unknown else 'ALGORITHM_FAILURE',
                rows=rows, observations=rows, source_hash=source_hash, raw_result_path=raw_path,
                denominator=len(rows), physical_started=len(starts), known_physical_started=len(starts),
                physical_started_uncertain=unknown, completed=sum(x.get('status')=='EVALUATED' for x in rows),
                hard_stop=unknown, infrastructure_error=infrastructure_error,
                active_seconds=active_seconds,
                load_seconds_known=sum(x.get('load_seconds') or 0 for x in rows),
                route_seconds_known=sum(x.get('route_seconds') or 0 for x in rows),
                route_seconds_missing=sum(x.get('route_seconds') is None for x in rows),
                child_cpu_user_seconds_known=sum(x.get('child_cpu_user_seconds') or 0 for x in rows),
                child_cpu_system_seconds_known=sum(x.get('child_cpu_system_seconds') or 0 for x in rows),
                child_cpu_missing_rows=sum(x.get('child_cpu_user_seconds') is None for x in rows),
                optimal_reference=None, gap=None,
                gap_status='MISSING_NO_FROZEN_OPTIMAL_REFERENCE_FOR_NEW_INSTANCES',
                worker_terminal=terminal, container_state=container_state, integrity_errors=integrity_errors)

class PatchEvaluator(PairedEvaluator):
    def __init__(self,run_dir,cpu,config=None):
        self.root=Path(run_dir).resolve(); self.cpu=str(cpu); self.config=dict(config or {})
        if self.cpu not in CANDIDATE_CPUS:
            raise ValueError('CPU_OUTSIDE_V3_RESERVED_SLOTS')
        if any(k.startswith('imported') for k in self.config):
            raise ValueError('HISTORICAL_IMPORT_FORBIDDEN')
        self.study=Path(self.config['study_root']).resolve()
        if not self.root.is_relative_to(self.study):
            raise ValueError('EVALUATOR_OUTSIDE_NEW_STUDY')
        for cache in self.config.get('common_cache_roots',[]):
            cache=Path(cache).resolve()
            if not cache.is_relative_to(self.study): raise ValueError('CACHE_OUTSIDE_NEW_STUDY')
            relative=cache.relative_to(self.study).parts
            if not (relative[0]=='references' or (relative[0]=='searches' and relative[-2:] in (('INIT','evaluator'),('PROBE','evaluator')))):
                raise ValueError('POST_INIT_CROSS_ARM_CACHE_FORBIDDEN')
        self.worker=Path(__file__).with_name('rpc_worker.py')
        self.guard=threading.RLock();self.root.mkdir(parents=True,exist_ok=True)
        self.execution_binding=dict(version='EXECUTABLE-PATCH-EVALUATION-V3.1-RPC',study_id=STUDY_ID,
            cpu_count=1,memory='8g',threads=1,image=ev.IMAGE,runtime=ev.RUNTIME,
            worker_sha256=ev.digest(self.worker.read_bytes()),
            trusted_rpc_host_sha256=ev.digest(Path(__file__).with_name('rpc_host.py').read_bytes()),
            trusted_tsp_math_sha256=ev.digest(Path(__file__).with_name('sandbox_worker.py').read_bytes()),
            trusted_cvrp_math_sha256=ev.digest(Path(__file__).with_name('cvrp_contract.py').read_bytes()),native_evaluator_sha256=ev.digest(Path(ev.__file__).read_bytes()),
            inherited_paired_runtime_sha256=ev.digest(Path(prior.__file__).read_bytes()),
            runtime_sha256=ev.digest(Path(__file__).read_bytes()),load_seconds=10,search_panel_seconds=500,
            validation_test_route_seconds=300,isolation='external-trusted-host/per-instance-container/stepwise-node-rpc',
            ordinary_candidate_clocks='allowed',node_frame_max_bytes=8192,diagnostic_retained_bytes=8192,
            readiness_semantics='protocol READY by t0+10; not proof of internal module loading',
            hidden_deadline='min(t0+310,t_route+300,existing_panel_deadline)',
            cache_semantics='reuse same paid observation; not independent repetition',
            search_timer='entire Search32 panel including all instance loads',
            runtime_identity=self.config.get('runtime_identity',{}))
        self.binding=dict(self.execution_binding,cache_scope=str(self.root),cpu=self.cpu)
        ev.freeze(self.root/'CREDIT_EVALUATOR_BINDING.json',self.binding)

    def _historical_result(self,source,panel_hash): return None

    def _common_result(self,key):
        for cache in self.config.get('common_cache_roots',[]):
            base=Path(cache).resolve(); path=base/'physical_evaluations'/key/'EVALUATION.json'
            if not path.exists():
                if path.parent.exists(): raise RuntimeError('INHERITED_ENDPOINT_TERMINAL_MISSING')
                continue
            binding=ev.read(base/'CREDIT_EVALUATOR_BINDING.json')
            if {k:v for k,v in binding.items() if k not in ('cache_scope','cpu')} != self.execution_binding:
                raise RuntimeError('SHARED_CACHE_EXECUTION_BINDING_CONFLICT')
            return path,ev.read(path)
        return None

    def _validate_panel_access(self,panel,panel_path):
        identity=panel['identity'];phase=identity.get('phase');group=int(identity.get('group',-1));role=panel['role']
        if identity.get('study_id') != STUDY_ID or phase not in PHASE_GROUPS or group not in range(PHASE_GROUPS[phase]):
            raise RuntimeError('V3_PANEL_IDENTITY_CONFLICT')
        from . import cvrp_contract
        shapes=cvrp_contract.PANEL_SHAPES if phase=='cvrp' else PANEL_SHAPES
        task=cvrp_contract.TASK if phase=='cvrp' else TASK
        if panel['task'] != task or role not in shapes: raise RuntimeError('UNREGISTERED_TASK_ROLE')
        count,n=shapes[role]
        if phase=='cvrp':n+=1
        if len(panel['instances'])!=count or any(len(x['coordinates'])!=n for x in panel['instances']):
            raise RuntimeError('PANEL_SHAPE_CONFLICT')
        if panel['limits']!=prior.panel_limits(role):raise RuntimeError('FROZEN_LIMIT_CONFLICT')
        if role=='Search32': return
        lock=Path(self.config.get('stage_lock_path',''))
        if not lock.is_file() or not lock.resolve().is_relative_to(self.study/'stages'):
            raise RuntimeError('HIDDEN_PANEL_STAGE_LOCK_MISSING')
        receipt=ev.read(lock)
        stage=receipt.get('stage')
        expected={'dev_core':('dev',8,('J','A','W','R')),'dev_calibration':('dev',8,('E','H')),
            'confirm':('confirm',20,ARMS),'cvrp':('cvrp',12,ARMS)}
        if stage not in expected or receipt.get('all_search_and_budget_locked') is not True:
            raise RuntimeError('HIDDEN_ACCESS_BEFORE_COMPLETE_STAGE_SEARCH')
        expectedphase,groups,arms=expected[stage]
        if phase!=expectedphase or (role=='Val128') != (phase=='dev'):
            raise RuntimeError('HIDDEN_PANEL_STAGE_CONFLICT')
        outputs=receipt.get('outputs',{})
        if set(outputs)!={f'g{i:02d}' for i in range(groups)}:
            raise RuntimeError('STAGE_GROUP_LOCKS_INCOMPLETE')
        if ev.digest(Path(panel_path).read_bytes()) not in receipt.get('panel_hashes',[]):
            raise RuntimeError('HIDDEN_PANEL_NOT_BOUND_TO_STAGE_LOCK')
        for group_id,group_outputs in outputs.items():
            if set(group_outputs)!=set(arms):raise RuntimeError('STAGE_ARM_LOCKS_INCOMPLETE')
            for arm,records in group_outputs.items():
                labels=set()
                for record in records:
                    path=_file_within(self.study,record['path'])
                    if ev.digest(path.read_bytes())!=record['sha256']:raise RuntimeError('STAGE_OUTPUT_HASH_CONFLICT')
                    output=ev.read(path)
                    if output.get('phase')!=phase or output.get('group')!=int(group_id[1:]) or output.get('arm')!=arm:
                        raise RuntimeError('STAGE_OUTPUT_IDENTITY_CONFLICT')
                    labels.add(output.get('label'))
                    if not output.get('search_locked'):raise RuntimeError('STAGE_OUTPUT_NOT_SEARCH_LOCKED')
                    h=output.get('search_best_hash')
                    if h:
                        source=self.study/'programs'/f'{h}.py'
                        if ev.digest(source.read_bytes())!=h:raise RuntimeError('STAGE_SELECTED_SOURCE_CHANGED')
                thresholds=(100000,200000,300000) if arm in ('J','A','W') else (100000,200000,300000,400000)
                if labels!={f'{arm}_terminal',*(f'{arm}_p{n}' for n in thresholds)} or len(records)!=len(labels):
                    raise RuntimeError('STAGE_PREDECLARED_ENDPOINTS_INCOMPLETE')
        initial=receipt.get('initialization_outputs',[])
        if 'J' in arms:
            if len(initial)!=groups:raise RuntimeError('I30_OUTPUT_LOCKS_INCOMPLETE')
            seen=set()
            for record in initial:
                path=_file_within(self.study,record['path'])
                if ev.digest(path.read_bytes())!=record['sha256']:raise RuntimeError('I30_OUTPUT_HASH_CONFLICT')
                output=ev.read(path);g=output.get('group');seen.add(g)
                if output.get('phase')!=phase or output.get('arm')!='I30' or output.get('label')!='I30_terminal' or not output.get('search_locked'):
                    raise RuntimeError('I30_OUTPUT_IDENTITY_CONFLICT')
                shared=ev.read(self.study/'searches'/phase/f'g{g:02d}'/'INIT/SHARED_INITIALIZATION.json')
                if output['selection']!=shared['selection'] or output['cost']!=shared['cost']:
                    raise RuntimeError('I30_SHARED_INITIALIZATION_BINDING_CONFLICT')
                h=output['search_best_hash']
                if ev.digest((self.study/'programs'/f'{h}.py').read_bytes())!=h:raise RuntimeError('I30_SELECTED_SOURCE_CHANGED')
            if seen!=set(range(groups)):raise RuntimeError('I30_GROUP_LOCKS_INCOMPLETE')
        elif initial:raise RuntimeError('UNREGISTERED_I30_CALIBRATION_OUTPUT')


    def _execute_locked(self, folder, source_path, panel_path, panel, cpu_slot_wait_seconds=0):
        # The panel and scorer never enter the candidate container. Each callback
        # exchanges one legal next-node decision with one current-instance container.
        from .rpc_host import run_instance
        _require_remote(self.root)
        if not self.config.get('cpu_slot_lock_dir'):
            raise RuntimeError('V3_SHARED_CPU_SLOT_LOCK_REQUIRED')
        terminal=folder/'RAW_RESULT.json'
        if terminal.exists():return ev.read(terminal)
        intent_path=folder/'INTENT.json';resumed=intent_path.exists()
        identity=dict(source_sha256=ev.digest(source_path.read_bytes()),
            panel_sha256=ev.digest(panel_path.read_bytes()),frozen_binding=self.binding)
        if resumed:
            intent=ev.read(intent_path)
            if any(intent.get(k)!=v for k,v in identity.items()):
                raise RuntimeError('EVALUATION_INTENT_IDENTITY_CONFLICT')
        else:
            intent=ev.freeze(intent_path,dict(identity,time_unix=time.time(),
                execution='trusted-panel-host/fresh-instance-container/stepwise-node-rpc'))
        import resource
        usages=lambda:[resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)]
        cpu_before=usages()
        began=time.monotonic();limit=panel['limits']['panel_seconds']
        deadline=began+limit if limit is not None else None
        events=[];error='HOST_INTERRUPTED_PANEL_NO_REDISPATCH' if resumed else None
        for index,item in enumerate(panel['instances']):
            instance_dir=folder/'instances'/f'{index:04d}'
            if resumed and not (instance_dir/'INTENT.json').exists():break
            if not resumed and deadline is not None and time.monotonic()>=deadline:
                break
            events.append(dict(type='INSTANCE_ATTEMPTED',instance_id=item['instance_id']))
            try:
                row=run_instance(source_path,item,task=panel['task'],cpu=self.cpu,
                    output_dir=instance_dir,panel_deadline=deadline,limits=panel['limits'])
            except Exception as exc:
                # No silent rerun: raw intent remains the proof of attempted work.
                row=dict(instance_id=item['instance_id'],status='UNKNOWN',length=None,
                    hard_stop=True,infrastructure_error='TRUSTED_HOST_EXCEPTION',
                    error_type=type(exc).__name__,error=str(exc)[:8192])
                ev.freeze(instance_dir/'HOST_EXCEPTION.json',row)
            if row.get('container_started') is True:
                events.append(dict(type='INSTANCE_STARTED',instance_id=item['instance_id']))
            events.append(dict(row,type='ROUTE_RESULT'))
            if row.get('hard_stop') or row.get('status')=='UNKNOWN':
                error=row.get('infrastructure_error') or 'INSTANCE_INFRASTRUCTURE_UNKNOWN'
                break
            if row.get('status')!='EVALUATED':break
        observed=[x for x in events if x.get('type')=='ROUTE_RESULT']
        events.append(dict(type='WORKER_RESULT',complete=not error and
            len(observed)==len(panel['instances']) and all(x.get('status')=='EVALUATED' for x in observed),
            trusted_origin='external-route-host',panel_deadline_exhausted=bool(deadline and time.monotonic()>=deadline)))
        result=aggregate_worker(events,panel,source_hash=intent['source_sha256'],
            raw_path=str(terminal),infrastructure_error=error,
            active_seconds=None if resumed else time.monotonic()-began)
        result.update(events=events,resumed_existing_intent=resumed,intent_path=str(intent_path),
            terminal_unix=time.time(),cpu_slot_wait_seconds=cpu_slot_wait_seconds,
            panel_sha256=intent['panel_sha256'],execution_binding=self.execution_binding)
        result.update(rpc_cost_fields(observed))
        result['physical_started_uncertain']=any(r.get('container_started') is None for r in observed)
        result['physical_attempted_instances']=len(observed)
        cpu_after=usages()
        host_cpu=sum((a.ru_utime+a.ru_stime)-(b.ru_utime+b.ru_stime) for a,b in zip(cpu_after,cpu_before))
        # Whole-panel accounting includes route construction, IPC, CLI cleanup,
        # integrity aggregation and gaps between instances. Container CPU is
        # supplied only by the external cgroup witness, never candidate events.
        if not resumed:
            result.update(trusted_host_cpu_seconds_known=host_cpu,trusted_host_cpu_missing_rows=0,
                total_evaluator_cpu_seconds_known=result['container_cpu_seconds_known']+host_cpu,
                total_evaluator_cpu_missing_rows=result['container_cpu_missing_rows'])
        else:
            # Recovery only knows retained instance subtotals and current
            # cleanup. The interrupted panel host span cannot be reconstructed.
            result['trusted_host_cpu_missing_rows']+=1
            result['total_evaluator_cpu_missing_rows']+=1
            result['interrupted_panel_host_cpu_missing']=True
        result['active_seconds']=None if resumed else time.monotonic()-began
        ev.freeze(terminal,result)
        return result


def rpc_cost_fields(rows):
    """Paid instance costs, including the trusted host; missing never becomes zero."""
    numeric=lambda value:type(value) in (int,float) and math.isfinite(value)
    add=lambda key:sum(r[key] for r in rows if numeric(r.get(key)))
    container=add('container_cpu_seconds_known')
    missing=sum(r.get('container_cpu_seconds_missing',1) for r in rows)
    host=add('host_cpu_user_seconds')+add('host_cpu_system_seconds')
    host_missing=sum(not numeric(r.get('host_cpu_user_seconds')) or not numeric(r.get('host_cpu_system_seconds')) for r in rows)
    return dict(container_cpu_seconds_known=container,container_cpu_missing_rows=missing,
        trusted_host_cpu_seconds_known=host,trusted_host_cpu_missing_rows=host_missing,
        total_evaluator_cpu_seconds_known=container+host,total_evaluator_cpu_missing_rows=missing+host_missing,
        cpu_accounting='container cgroup CPU plus external host and CLI CPU; incomplete components remain missing')


class PatchTransport(FullTransport):
    def __init__(self,run_dir,service_root,credential_file,run_id=None,config=None):
        self.root=Path(run_dir).resolve();self.service=tr.Circuit(service_root)
        self.credentials=str(credential_file);tr.credentials(self.credentials)
        self.run_id=run_id or self.root.name;self.config=dict(config or {})
        self.counter=0;self.records=[];self.logical_records=[];self.guard=threading.RLock();self._message_override=None
        self.binding=dict(study_id=STUDY_ID,model_settings=MODEL_SETTINGS,
            max_physical_attempts=3,unknown_replay=False,
            retry_policy='complete retryable HTTP rejection only; first actual model response ends task',
            service_namespace=str(Path(service_root).resolve()),credential_file=self.credentials,
            native_transport_sha256=ev.digest(Path(tr.__file__).read_bytes()),runtime_sha256=ev.digest(Path(__file__).read_bytes()),
            native_messages='exact immutable role/content list')
        tr.write(self.root/'CREDIT_TRANSPORT_BINDING.json',self.binding)

    def request_payload(self,prompt,metadata):
        body=prior.payload(prompt,self.config)
        if metadata.get('selector') is True:
            if metadata.get('max_tokens')!=256:raise ValueError('SELECTOR_LIMIT_CONFLICT')
            body['max_tokens']=256
        return body

    def __call__(self,prompt,metadata):
        system=metadata.get('system_prompt')
        if system is None:return super().__call__(prompt,metadata)
        from fuse_paired.full_runtime import flatten_native_messages
        messages=[dict(role='system',content=system),dict(role='user',content=prompt)]
        with self.native_messages(messages):
            return super().__call__(flatten_native_messages(messages),metadata)

    def _attempt(self, folder, body, frozen, identity, logical, metadata, attempt):
        folder.mkdir(parents=True, exist_ok=True)
        logical_attempt = logical+f'/attempt-{attempt:02d}'
        tr.write(folder/'FROZEN_INPUT.json', frozen)
        tr.write_bytes(folder/'payload.json', body)
        if (folder/'intent.json').exists() and tr.read(folder/'intent.json')['identity'] != identity:
            raise RuntimeError('REQUEST_FROZEN_IDENTITY_CONFLICT')
        while True:
            action, until = self.service.before(logical_attempt, folder, identity)
            if action != 'WAIT':
                break
            time.sleep(min(1, max(.05, (until or time.time()+1)-time.time())))
        if not (folder/'terminal.json').exists():
            if action == 'SEND':
                _require_remote(self.root)
                tr.write(folder/'intent.json', dict(identity=identity, submitted_unix=time.time(),
                    logical_id=logical_attempt, generation_id=logical, submission=1,
                    transport_attempt=attempt, timeout_seconds=150, retries=attempt-1, metadata=metadata))
                self._send_process(folder, identity)
            else:
                tr.settle_existing(folder)
            if (folder/'SENDER_HOST_FAILURE.json').exists():
                raise tr.HostFailure('PROVEN_PRE_HTTP_HOST_FAILURE: '+str(folder))
            result = tr.classify(folder)
            intent = tr.read(folder/'intent.json')
            result.update(identity=identity, logical_id=logical_attempt, generation_id=logical,
                          transport_attempt=attempt, request_path=str(folder),
                          submitted_unix=intent['submitted_unix'], metadata=metadata,
                          network_boundary_entered=(folder/'NETWORK_BOUNDARY_INTENT.json').exists(),
                          known_usage=result.get('usage') is not None)
            tr.write(folder/'terminal.json', result)
        result = tr.read(folder/'terminal.json')
        if result.get('identity') != identity or result.get('logical_id') != logical_attempt:
            raise RuntimeError('TERMINAL_REQUEST_IDENTITY_CONFLICT')
        self.service.observe(logical_attempt, result, folder)
        return result

    def _send_process(self, folder, identity):
        began = time.monotonic()
        try:
            done = subprocess.run([sys.executable, '-m', 'fuse_patch.runtime', '--submit',
                                   str(folder), '--credentials', self.credentials],
                                  capture_output=True, timeout=150)
            process = dict(returncode=done.returncode, stderr=done.stderr.decode(errors='replace'),
                           wall_seconds=time.monotonic()-began)
        except subprocess.TimeoutExpired:
            process = dict(returncode=None, status='REQUEST_PROCESS_DEADLINE', wall_seconds=time.monotonic()-began)
        except OSError as exc:
            process = dict(returncode=None, status='SENDER_PROCESS_START_FAILED',
                           exception_type=type(exc).__name__, message=str(exc), wall_seconds=time.monotonic()-began)
            tr.write(folder/'SENDER_HOST_FAILURE.json', dict(**process, definitely_pre_http=True,
                     payload_sha256=identity['payload_sha256']))
        tr.write(folder/'process.json', process)

def physical_cost_frontier(root,request_counter,roles,*,started=None):
    """Only logically consumed requests/roles, never future receipts on replay."""
    root=Path(root).resolve();requests=[]
    for index in range(request_counter):
        folder=root/'transport/requests'/f'{index:06d}'/'attempts'
        for path in sorted(folder.glob('*')):
            intent=(path/'intent.json').is_file()
            if not intent and not (path/'terminal.json').is_file():continue
            row=ev.read(path/'terminal.json') if (path/'terminal.json').is_file() else {}
            process=ev.read(path/'process.json') if (path/'process.json').is_file() else {}
            requests.append((intent,row,process,path))
    paths={Path(r['physical_record']).resolve() for r in roles if r.get('physical_record') and
           Path(r['physical_record']).resolve().is_relative_to(root/'evaluator')}
    physical=[ev.read(p) for p in sorted(paths)]
    add=lambda key:sum(r[key] for r in physical if _number(r.get(key)))
    totals=[_usage(r).get('total_tokens') for _,r,_,_ in requests]
    missing=sum(not intent or not _number(v) for (intent,_,_,_),v in zip(requests,totals))
    known=sum(v for v in totals if _number(v))
    cpu=sum(r.get('total_evaluator_cpu_seconds_known',r.get('child_cpu_user_seconds_known',0)+r.get('child_cpu_system_seconds_known',0)) for r in physical)
    cpumissing=sum(r.get('total_evaluator_cpu_missing_rows',r.get('child_cpu_missing_rows',0)) for r in physical)
    cpumissing+=sum(not _number(r.get('total_evaluator_cpu_seconds_known',r.get('child_cpu_user_seconds_known'))) for r in physical)
    active_missing=sum(not _number(r.get('active_seconds')) for r in physical)
    ends=[r.get('terminal_unix',0) for r in roles]
    ends += [r.get('submitted_unix',0)+(p.get('wall_seconds') or 0) for _,r,p,_ in requests]
    wall=max(0,max(ends,default=started or 0)-(started or 0)) if started is not None else None
    return dict(http=sum(a for a,_,_,_ in requests),response_received=sum(r.get('service_success') is True for _,r,_,_ in requests),
        unknown=sum(r.get('status')=='UNKNOWN' for _,r,_,_ in requests),pending_attempts=sum(not r for _,r,_,_ in requests),
        tokens_known=known,tokens_missing=missing,tokens=known if not missing else None,
        input_tokens_known=sum(v for _,r,_,_ in requests if _number(v:=_usage(r).get('prompt_tokens'))),
        input_tokens_missing=sum(not _number(_usage(r).get('prompt_tokens')) for _,r,_,_ in requests),
        output_tokens_known=sum(v for _,r,_,_ in requests if _number(v:=_usage(r).get('completion_tokens'))),
        output_tokens_missing=sum(not _number(_usage(r).get('completion_tokens')) for _,r,_,_ in requests),
        llm_wall_seconds_known=sum(p['wall_seconds'] for _,_,p,_ in requests if _number(p.get('wall_seconds'))),
        llm_wall_seconds_missing=sum(not _number(p.get('wall_seconds')) for _,_,p,_ in requests),
        network_boundary_intents=sum((p/'NETWORK_BOUNDARY_INTENT.json').is_file() for _,_,_,p in requests),
        evaluation_roles=len(roles),physical_evaluations=len(physical),physical_instances=add('physical_started'),
        physical_unknown=sum(r.get('physical_started_uncertain') is not False for r in physical),
        evaluator_active_seconds=add('active_seconds') if not active_missing else None,
        evaluator_active_seconds_known=add('active_seconds'),evaluator_active_seconds_missing=active_missing,
        evaluator_queue_seconds=add('cpu_slot_wait_seconds'),evaluator_cpu_seconds=cpu if not cpumissing else None,
        evaluator_cpu_seconds_known=cpu,cpu_seconds=cpu if not cpumissing else None,cpu_seconds_known=cpu,
        evaluator_cpu_missing_rows=cpumissing,container_cpu_seconds_known=add('container_cpu_seconds_known'),
        trusted_host_cpu_seconds_known=add('trusted_host_cpu_seconds_known'),evaluation_failures=sum(r.get('complete') is False for r in physical),
        evaluation_pending=sum(not r for r in physical),wall_seconds=wall)


def physical_cost_snapshot(root,*,started=None):
    """All physically attempted work for delivery, including interrupted intents."""
    from fuse_paired.full_runtime import physical_cost_snapshot as inherited_snapshot
    root=Path(root);result=inherited_snapshot(root,started=started)
    paths={p.parent for pattern in ('*/INTENT.json','*/EVALUATION.json')
           for p in (root/'evaluator/physical_evaluations').glob(pattern)}
    physical=[ev.read(p/'EVALUATION.json') if (p/'EVALUATION.json').exists() else {} for p in sorted(paths)]
    total=sum(r.get('total_evaluator_cpu_seconds_known',0) for r in physical)
    missing=sum(r.get('total_evaluator_cpu_missing_rows',1) for r in physical)
    result.update(cpu_seconds_known=total,evaluator_cpu_seconds_known=total,
        cpu_seconds=total if not missing else None,evaluator_cpu_seconds=total if not missing else None,
        evaluator_cpu_missing_rows=missing,
        container_cpu_seconds_known=sum(r.get('container_cpu_seconds_known',0) for r in physical),
        trusted_host_cpu_seconds_known=sum(r.get('trusted_host_cpu_seconds_known',0) for r in physical))
    return result


def combine_cost(*costs):
    keys=set().union(*(set(c) for c in costs));result={}
    for key in keys:
        values=[c.get(key,0) for c in costs]
        if any(v is None for v in values):result[key]=None
        elif all(_number(v) for v in values):result[key]=sum(values)
    return result


def _fresh_state(context,reference):
    return dict(schema='executable-patch-state-v3',context=context,programs={},qualities={},occurrences={},
        occurrence_counter=0,history=[],patch_versions={},failures=[],native_auxiliary=[],
        init_completed=0,noninit_completed=0,archive_counter=0,scale=reference['mean'],reference_hash=reference['source_hash'])


class SearchIO:
    """No model/candidate work at construction. All such work is remote guarded."""
    def __init__(self,study,phase,group,arm,cpu,credential_file):
        self.study=Path(study).resolve();self.phase=phase;self.group=group;self.arm=arm
        if phase not in PHASE_GROUPS or type(group) is not int or group not in range(PHASE_GROUPS[phase]) or arm not in (*ARMS,'INIT','PROBE'):
            raise ValueError('UNREGISTERED_V3_SEARCH')
        self.root=self.study/'searches'/phase/f'g{group:02d}'/arm
        self.root.mkdir(parents=True,exist_ok=True)
        self.panel=self.study/'panels'/phase/f'g{group:02d}'/'Search32.json'
        panel=ev.read(self.panel);self.task=panel['task'];self.panel_hash=ev.digest(self.panel.read_bytes())
        identity=panel['identity']
        if identity.get('study_id')!=STUDY_ID or identity.get('phase')!=phase or str(identity.get('group'))!=str(group) or panel['role']!='Search32':
            raise RuntimeError('SEARCH_PANEL_IDENTITY_CONFLICT')
        refroot=self.study/'references'/phase/f'g{group:02d}'
        reference=ev.read(refroot/'REFERENCE.json')
        expected=reference_source(phase)
        if reference['source']!=expected or reference['source_hash']!=ev.digest(expected):
            raise RuntimeError('PUBLIC_REFERENCE_SOURCE_CONFLICT')
        cache=[str(refroot/'evaluator')]
        if arm in ('J','A','W','R','PROBE'):cache.append(str(self.study/'searches'/phase/f'g{group:02d}'/'INIT/evaluator'))
        if arm in ('J','A','W'):cache.append(str(self.study/'searches'/phase/f'g{group:02d}'/'PROBE/evaluator'))
        config=dict(study_root=str(self.study),runtime_identity=ev.read(self.study/'engineering/RUNTIME_IDENTITY.json'),
            cpu_slot_lock_dir=str(self.study/'cpu-slots'),common_cache_roots=cache)
        self.evaluator=PatchEvaluator(self.root/'evaluator',cpu,config)
        quality=ev.read(_file_within(self.study,reference['evaluation_path']))
        if quality.get('complete') is not True or quality['source_hash']!=reference['source_hash'] or quality['panel_sha256']!=self.panel_hash or quality['execution_binding']!=self.evaluator.execution_binding or quality['mean']!=reference['mean']:
            raise RuntimeError('PUBLIC_REFERENCE_BINDING_CONFLICT')
        self.reference=dict(reference,quality=quality)
        self.environment_hash=ev.digest(ev.encoded(self.evaluator.execution_binding))
        self.context=dict(study_id=STUDY_ID,phase=phase,group=group,arm=arm,task='tsp' if self.task=='TSP-Construct' else 'cvrp',task_id=self.task,
            panel_sha256=self.panel_hash,environment_hash=self.environment_hash,execution_binding=self.evaluator.execution_binding)
        from fuse_patch.native import TSP_DESCRIPTION
        from fuse_patch.cvrp_contract import CVRP_DESCRIPTION,CVRP_FUNC_DESCRIPTION
        self.task_prompt=TSP_DESCRIPTION+' API: select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix) -> int. Node IDs are integers; unvisited_nodes is a NumPy array of currently allowed city IDs, distance_matrix is the full Euclidean matrix. Return one allowed node, never bool. The host closes the complete tour and measures its length.'
        if phase=='cvrp':self.task_prompt=CVRP_DESCRIPTION+' API: '+CVRP_FUNC_DESCRIPTION
        from .budget import EXECUTION_CAPABILITIES
        self.task_prompt += EXECUTION_CAPABILITIES
        self.transport=PatchTransport(self.root/'transport',self.study/'service',credential_file,
            run_id=f'{STUDY_ID}/{phase}/g{group:02d}/{arm}')
        started=self.root/'STARTED.json'
        if not started.exists():ev.freeze(started,dict(unix=time.time(),phase=phase,group=group,arm=arm,cpu=str(cpu)))
        self.started=ev.read(started)['unix'];self.controllers={};self._eval_roles={};self._shared=None;self._shared_probes=None
        self.state=_fresh_state(self.context,self.reference)

    def replay_native_from_start(self):
        self.state=_fresh_state(self.context,self.reference);self.transport.counter=0
        self.transport.records.clear();self.transport.logical_records.clear();self._eval_roles={};self._shared=None;self._shared_probes=None

    def attach_shared(self,shared):
        if self.arm not in ('J','A','W','R','PROBE') or shared['phase']!=self.phase or shared['group']!=self.group:
            raise RuntimeError('SHARED_INITIALIZATION_IDENTITY_CONFLICT')
        expected=self.study/'searches'/self.phase/f'g{self.group:02d}'/'INIT'
        if Path(shared['init_root']).resolve()!=expected or shared['reference']['source_hash']!=self.reference['source_hash']:
            raise RuntimeError('SHARED_INITIALIZATION_OWNER_CONFLICT')
        original=shared['archive_state']['context']
        if original['panel_sha256']!=self.panel_hash or original['execution_binding']!=self.evaluator.execution_binding:
            raise RuntimeError('SHARED_INITIALIZATION_CONTEXT_CONFLICT')
        self.state=copy.deepcopy(shared['archive_state']);self.state['context']=self.context
        self.state.pop('_io_transport_counter',None);self.state.pop('_io_role_frontier',None)
        self._shared=copy.deepcopy(shared)
        ev.freeze(self.root/'SHARED_INITIALIZATION_BINDING.json',dict(shared_initialization_sha256=ev.digest(ev.encoded(shared)),
            physical_owner=shared['physical_owner'],full_cost_charged=True,physical_reexecution=False))

    def inherit_probe_shared(self,shared_probes):
        if self.arm not in ('J','A','W') or self._shared is None:
            raise RuntimeError('PROBE_SHARING_ONLY_INITIALIZED_JAW')
        original=shared_probes['archive_state']['context']
        expected=self.study/'searches'/self.phase/f'g{self.group:02d}'/'PROBE'
        owner=shared_probes.get('probe_root',shared_probes.get('physical_owner'))
        if str(Path(owner).resolve())!=str(expected) or original['phase']!=self.phase or original['group']!=self.group:
            raise RuntimeError('SHARED_PROBE_OWNER_CONFLICT')
        if original['panel_sha256']!=self.panel_hash or original['execution_binding']!=self.evaluator.execution_binding:
            raise RuntimeError('SHARED_PROBE_CONTEXT_CONFLICT')
        self.state=copy.deepcopy(shared_probes['archive_state']);self.state['context']=self.context
        self.state.pop('_io_transport_counter',None);self.state.pop('_io_role_frontier',None)
        self._shared_probes=copy.deepcopy(shared_probes)
        ev.freeze(self.root/'SHARED_PROBE_BINDING.json',dict(shared_probes_sha256=ev.digest(ev.encoded(shared_probes)),
            physical_owner=str(expected),full_cost_charged=True,physical_reexecution=False))

    def probe_cost_snapshot(self):
        return self.own_cost_snapshot()

    def save_program(self,source):
        h=ev.digest(source);ev.freeze_bytes(self.study/'programs'/f'{h}.py',source.encode('utf-8'));return h

    def evaluate(self,source,role):
        result=self.evaluator.evaluate(source,self.panel,role)
        self._eval_roles[role]=result
        if result.get('hard_stop'):
            raise ev.EvaluationInfrastructureError('V3_EVALUATOR_BLOCKED:'+str(result.get('infrastructure_error')))
        if result.get('complete'):self.state['qualities'][ev.digest(source)]=result
        return result

    def register_program(self,source,quality,parents,origin,position,algorithm=''):
        h=self.save_program(source)
        if quality.get('complete') is not True or quality.get('source_hash')!=h or not _number(quality.get('mean')):
            raise ValueError('ADMISSION_REQUIRES_COMPLETE_BOUND_QUALITY')
        self.state['qualities'][h]=quality
        self.state['occurrence_counter']+=1;index=self.state['occurrence_counter']
        oid=f'{self.arm}/{index:06d}/{h[:12]}'
        parentids=[self.state['programs'][p]['latest_occurrence_id'] for p in parents if p in self.state['programs']]
        occurrence=dict(id=oid,occurrence_id=oid,index=index,program_hash=h,source_hash=h,source=source,
            parents=list(parents),parent_occurrence_ids=parentids,origin=origin,position=position,
            mean=quality['mean'],quality_available=True,algorithm=algorithm)
        self.state['occurrences'][oid]=occurrence
        if h not in self.state['programs']:
            self.state['archive_counter']+=1
            self.state['programs'][h]=dict(hash=h,source=source,mean=quality['mean'],parents=list(parents),origin=origin,
                first_position=position,first_index=index,occurrence_id=oid,algorithm=algorithm,quality_available=True)
        self.state['programs'][h]['latest_occurrence_id']=oid
        return h

    def checkpoint(self,state,tag=None):
        if tag is None:tag,state=state,self.state
        if not isinstance(tag,str) or Path(tag).name!=tag or tag in ('.','..','LATEST'):raise ValueError('UNSAFE_CHECKPOINT_TAG')
        state['_io_transport_counter']=self.transport.counter
        state['_io_role_frontier']=list(self._eval_roles)
        path=self.root/'states'/f'{tag}.json';ev.freeze(path,state)
        pointer=dict(path=str(path.relative_to(self.root)),sha256=ev.digest(path.read_bytes()),tag=tag)
        target=self.root/'states/LATEST.json';fd,tmp=tempfile.mkstemp(prefix='.latest-',dir=target.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(ev.encoded(pointer)+b'\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,target)
        finally:
            if Path(tmp).exists():Path(tmp).unlink()
        self.state=state;return state

    def rng(self,stream,position):return seed_for(self.phase,self.task,self.group,self.arm,f'{position}/{stream}')
    def native_messages(self,messages):return self.transport.native_messages(messages)
    def own_cost_snapshot(self):
        return physical_cost_frontier(self.root,self.transport.counter,list(self._eval_roles.values()),started=self.started)
    def cost_snapshot(self):
        own=self.own_cost_snapshot();reference=self.reference['cost']
        initialization=self._shared['own_physical_cost'] if self._shared is not None else {}
        probes=self._shared_probes['own_physical_cost'] if self._shared_probes is not None else {}
        combined=combine_cost(reference,initialization,probes,own)
        return dict(combined,reference_cost=reference,shared_initialization_cost=initialization,shared_probe_cost=probes,own_physical_cost=own,
            reference_charged_in_full=True,initialization_charged_in_full=self._shared is not None,
            probes_charged_in_full=self._shared_probes is not None,
            cost_scope='ONE_NN_REFERENCE_PLUS_INIT_PLUS_JAW_SHARED_PROBES_WHEN_INHERITED_PLUS_OWN_WORK',
            wall_interpretation='sum of physical-owner wall spans; includes scheduling pauses; not speedup evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--submit',required=True);parser.add_argument('--credentials',required=True)
    args=parser.parse_args();_require_remote(args.submit);tr.sender(args.submit,args.credentials)
