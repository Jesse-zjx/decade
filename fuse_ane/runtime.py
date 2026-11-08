"""ANE-r1 I/O seams around the unchanged trusted candidate executor.

The amendment changes only the execution namespace, output decoder, bounded
transport/replay and accounting. Trusted candidate source and limits are pinned.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np
from fuse_addon import evaluation as ev, transport as tr
from fuse_patch import runtime as inherited, rpc_host
from fuse_paired.runtime import panel_limits, payload
from fuse_paired.full_runtime import flatten_native_messages

STUDY_ID = 'fuse-shared-edit-siblings-20261007'
EXECUTION_ID = STUDY_ID
POLICY_REVISION = 'FUSE_SIBLING1'
ROOT = Path('/opt/decade/edit-search')
STUDY = ROOT/'runs'/EXECUTION_ID
MODEL_SETTINGS = copy.deepcopy(inherited.MODEL_SETTINGS)
_ORIGINAL_RUN_INSTANCE = rpc_host.run_instance
_ORIGINAL_GUARD = rpc_host._guard


def stop_dispatch(reason, study=None):
    study = STUDY if study is None else study
    target = Path(study)/'STOP_DISPATCH.json'
    if not target.exists():
        try:
            ev.freeze(target, dict(reason=reason, time_unix=time.time(), future_dispatch=False,state='DRAINING',inflight_evaluation_may_finish=True))
        except RuntimeError:
            if not target.exists():
                raise


def require_open(path, *, evaluation=False):
    path = Path(path).resolve()
    if platform.system() != 'Linux' or not path.is_relative_to(STUDY):
        raise RuntimeError('IG_SCIENCE_REQUIRES_AUTHORIZED_REMOTE_NAMESPACE')
    if not Path(ev.RUNTIME+'/bin/python').is_file():
        raise RuntimeError('AUTHORIZED_REMOTE_RUNTIME_MISSING')
    for name in ('EXECUTION_FREEZE.json', 'RUN_ALLOWED.json'):
        if not (STUDY/name).is_file():
            raise RuntimeError('IG_DISPATCH_LOCK_MISSING:'+name)
        lock = ev.read(STUDY/name)
        if (lock.get('study_id') != STUDY_ID or lock.get('execution_id') != EXECUTION_ID
                or lock.get('user_authorized') is not True):
            raise RuntimeError('IG_DISPATCH_LOCK_IDENTITY_CONFLICT:'+name)
    for name in (('RESULT_SEALED.json', 'TERMINAL.json', 'STOP_UNKNOWN.json') if evaluation else ('RESULT_SEALED.json', 'TERMINAL.json', 'STOP_DISPATCH.json', 'STOP_UNKNOWN.json')):
        if (STUDY/name).exists():
            raise RuntimeError('IG_FUTURE_DISPATCH_FALSE:'+name)


def require_evaluation_open(path):
    return require_open(path, evaluation=True)


def guarded_instance(source, item, **kwargs):
    output = Path(kwargs['output_dir'])
    if kwargs.get('public_acceptance'):
        if not output.resolve().is_relative_to(ROOT/'engineering/ane-public'):
            raise RuntimeError('PUBLIC_CHECK_SCOPE')
        return _ORIGINAL_RUN_INSTANCE(source, item, **kwargs)
    require_evaluation_open(output)
    result = _ORIGINAL_RUN_INSTANCE(source, item, **kwargs)
    if result.get('status') == 'UNKNOWN' or result.get('hard_stop'):
        stop_dispatch(result.get('infrastructure_error') or 'UNKNOWN_INSTANCE_OUTCOME')
    return result


def scoped_guard(source,output,public):
    if not public:return _ORIGINAL_GUARD(source,output,public)
    if platform.system()!='Linux' or not source.is_relative_to(ROOT/'engineering/ane-public') or not output.is_relative_to(ROOT/'engineering/ane-public'):
        raise RuntimeError('ANE_PUBLIC_REMOTE_SCOPE')
    authorization=ev.read(ROOT/'engineering/ane-public/AUTHORIZED.json')
    if authorization.get('study_id')!=STUDY_ID or authorization.get('instances')!=4 or authorization.get('user_authorized') is not True:
        raise RuntimeError('ANE_PUBLIC_AUTHORITY')
    if (ROOT/'engineering/ANE_PUBLIC_CHECKS.json').exists():raise RuntimeError('ANE_PUBLIC_ALREADY_FINISHED')


def bind_namespace():
    """Process-local rebinding only; inherited source files stay unchanged."""
    from .bounded_io import install
    install(STUDY)
    from fuse_ig.recovery_credentials import credentials
    tr.credentials = credentials
    inherited.STUDY_ID = STUDY_ID
    inherited.REMOTE_ROOT = ROOT
    inherited._require_remote = require_evaluation_open
    rpc_host.REMOTE_ROOT = ROOT
    rpc_host.run_instance = guarded_instance
    rpc_host._guard = scoped_guard


class Transport(inherited.PatchTransport):
    """Immutable logical positions, reconciliation and at most two physical attempts."""
    def __init__(self, run_dir, service_root, credential_file, run_id=None, config=None):
        bind_namespace()
        self.root = Path(run_dir).resolve()
        self.service = tr.Circuit(service_root)
        self.credentials = str(credential_file)
        tr.credentials(self.credentials)
        self.run_id = run_id or self.root.name
        self.config = dict(config or {})
        self.counter = 0
        self.records, self.logical_records = [], []
        self.guard = threading.RLock()
        self._message_override = None
        self.binding = dict(study_id=STUDY_ID, execution_id=EXECUTION_ID, model_settings=MODEL_SETTINGS,
            max_physical_attempts=2, unknown_replay='no historical coordinates', sender_seconds=300, supervisor_seconds=360,
            retry_policy='RNE1_RECONCILE_ONCE_TRANSIENT_NO_HTTP400',
            operation_output_caps=dict(single=8192, sibling_batch=16384, repair=8192, native=8192),
            service_namespace=str(Path(service_root).resolve()), credential_file=self.credentials,
            native_transport_sha256=ev.digest(Path(tr.__file__).read_bytes()),
            inherited_runtime_sha256=ev.digest(Path(inherited.__file__).read_bytes()),
            runtime_sha256=ev.digest(Path(__file__).read_bytes()),
            model_sampling_seed_supported=False, native_messages='exact role/content list')
        tr.write(self.root/'CREDIT_TRANSPORT_BINDING.json', self.binding)

    def request_payload(self, prompt, metadata):
        body = payload(prompt, self.config)
        cap = 16384 if metadata.get('kind') == 'sibling_batch' else 8192
        if metadata.get('max_tokens', cap) != cap:
            raise ValueError('FROZEN_OUTPUT_CAP_CONFLICT')
        if cap == 16384 and metadata.get('arm') not in ('F2', 'N2'):
            raise ValueError('BATCH_ARM_CONFLICT')
        body['max_tokens'] = cap
        return body

    def __call__(self, prompt, metadata):
        from .r1_transport import call
        return call(self, prompt, metadata)

    def _attempt(self, folder, body, frozen, identity, logical, metadata, attempt):
        from .r1_transport import attempt_request
        return attempt_request(self, folder, body, frozen, identity, logical, metadata, attempt)

    def _send_process(self, folder, identity):
        require_evaluation_open(folder)
        began = time.monotonic()
        try:
            done = subprocess.run([sys.executable, '-m', 'fuse_ane.runtime', '--submit',
                str(folder), '--credentials', self.credentials], capture_output=True, timeout=360)
            process = dict(returncode=done.returncode, stderr=done.stderr.decode(errors='replace'), wall_seconds=time.monotonic()-began)
        except subprocess.TimeoutExpired:
            process = dict(returncode=None, status='REQUEST_PROCESS_DEADLINE', wall_seconds=time.monotonic()-began)
        except OSError as exc:
            process = dict(returncode=None, status='SENDER_PROCESS_START_FAILED',
                exception_type=type(exc).__name__, message=str(exc), wall_seconds=time.monotonic()-began)
            tr.write(folder/'SENDER_HOST_FAILURE.json', dict(**process, definitely_pre_http=True, payload_sha256=identity['payload_sha256']))
        tr.write(folder/'process.json', process)



def save_program(study,source):
    h=ev.digest(source);ev.freeze_bytes(Path(study)/'programs'/f'{h}.py',source.encode());return h

def split_entries(study,phase,task,group,split):
    return ev.read(Path(study)/'phases'/phase/f'PANEL_MANIFEST_{split}.json')['groups'][f'{task}/{group}']

class Evaluator(inherited.PatchEvaluator):
    def __init__(self,run_dir,cpu,config=None):
        bind_namespace();super().__init__(run_dir,cpu,config)

    def _validate_panel_access(self,panel,panel_path):
        require_evaluation_open(self.root)
        from .data import groups,COUNTS,TASKS
        i=panel['identity'];phase,task,g,split=(i[k] for k in ('phase','task','group','split'))
        reused=i['study_id']=='fuse-reactive-neighborhood-evolution-20261006' and phase=='pilot'
        if reused:phase='dev_reuse'
        if (i['study_id']!=STUDY_ID and not reused) or (task,g) not in groups(phase) or split not in COUNTS:raise RuntimeError('PANEL_IDENTITY')
        if panel['task']!=TASKS[task] or panel['limits']!=panel_limits(panel['role']):raise RuntimeError('PANEL_TASK_RESOURCE')
        entries=split_entries(self.study,phase,task,g,split)
        matches=[r for r in entries if r['sha256']==ev.digest(Path(panel_path).read_bytes())]
        if len(matches)!=1 or matches[0]['count']!=len(panel['instances']):raise RuntimeError('PANEL_BYTES')
        base=self.study/'phases'/phase
        if split in ('screen','test'):
            short=ev.read(base/'SHORTLISTS.json')
            if not short['all_searches_stopped'] or set(short['groups'])!={f'{t}/{g}' for t,g in groups(phase)}:raise RuntimeError('ALL_SEARCH_LOCK')
        if split=='test':
            lock=ev.read(base/'SELECTION_LOCK.json')
            if not lock['locked_before_test'] or lock['shortlists_sha256']!=ev.digest((base/'SHORTLISTS.json').read_bytes()):raise RuntimeError('TEST_SELECTION_LOCK')

    def _evaluate_locked(self,source,panel_path,role,wait_seconds):
        p=ev.read(panel_path);self._validate_panel_access(p,panel_path);i=p['identity'];s=i['split'];base=self.study/'phases'/('dev_reuse' if i['study_id']=='fuse-reactive-neighborhood-evolution-20261006' else i['phase']);key=f"{i['task']}/{i['group']}"
        if s=='screen':allowed={h for r in ev.read(base/'SHORTLISTS.json')['groups'][key].values() for h in r['shortlist']}
        elif s=='test':allowed={r['selected_source'] for r in ev.read(base/'SELECTION_LOCK.json')['groups'][key].values() if r.get('selected_source')}
        else:allowed=None
        if allowed is not None and ev.digest(source) not in allowed:raise RuntimeError('SOURCE_NOT_FROZEN_FOR_SPLIT')
        return super()._evaluate_locked(source,panel_path,role,wait_seconds)

    def _execute_locked(self,*args,**kwargs):
        folder = args[0] if args else kwargs['folder']
        if (folder/'RESUME_ORIGIN.json').exists():raise RuntimeError('NO_HISTORICAL_EVALUATION_IMPORT')
        r = super()._execute_locked(*args,**kwargs)
        if r.get('hard_stop') or r.get('status')=='UNKNOWN':stop_dispatch('CANDIDATE_EVALUATION_UNKNOWN')
        return r

def evaluate_program(study,phase,task,group,source,split,role,evaluator):
    from .data import COUNTS
    h=save_program(study,source)
    if phase=='dev_reuse':
        from .sibling_reuse import cached_historical
        cached=cached_historical(study,task,group,h,split)
        if cached is not None:
            # Same source bytes, original panel bytes and trusted execution contract.
            if cached['execution_binding']['study_id']!='fuse-reactive-neighborhood-evolution-20261006':raise RuntimeError('REUSE_SOURCE_STUDY')
            current={k:v for k,v in evaluator.execution_binding.items() if k not in ('runtime_identity','study_id')}
            original={k:v for k,v in cached['execution_binding'].items() if k not in ('runtime_identity','study_id')}
            if current!=original:raise RuntimeError('INIT_EXECUTION_CONTRACT_CHANGED')
            if cached['panel_hashes']!=[e['sha256'] for e in split_entries(study,phase,task,group,split)]:raise RuntimeError('INIT_PANEL_MISMATCH')
            return dict(cached,reused_paid_initialization=True,reuse_source_study='fuse-reactive-neighborhood-evolution-20261006')
    rows=[];fragments=[];failed=False;unknown=False
    for index,e in enumerate(split_entries(study,phase,task,group,split)):
        path=Path(study)/e['path'];panel=ev.read(path)
        if ev.digest(path.read_bytes())!=e['sha256']:raise RuntimeError('PANEL_BYTES_CHANGED')
        if failed:
            rows.extend(dict(instance_id=x['instance_id'],status='NOT_EXECUTED',length=None,reason='FIRST_SHARD_FAILED') for x in panel['instances']);continue
        result=evaluator.evaluate(source,path,f'{role}/{index}')
        receipt=evaluator.root/'evaluations'/role/str(index)/'EVALUATION.json'
        fragments.append(dict(path=str(receipt),sha256=ev.digest(receipt.read_bytes()),physical_record=result['physical_record'],physical_record_sha256=result['physical_record_sha256'],cache_hit=result.get('cache_hit'),panel_sha256=e['sha256']))
        rows.extend(result['rows']);failed=not result['complete'];unknown=result.get('status')=='UNKNOWN' or bool(result.get('hard_stop'))
    complete=not failed and len(rows)==COUNTS[split] and all(x['status']=='EVALUATED' for x in rows)
    return dict(source_hash=h,mean=float(np.mean([x['length'] for x in rows])) if complete else None,complete=complete,
        status='EVALUATED' if complete else 'UNKNOWN' if unknown else 'ALGORITHM_FAILURE',hard_stop=unknown,
        phase=phase,task=task,group=group,split=split,denominator=COUNTS[split],completed=sum(x['status']=='EVALUATED' for x in rows),
        rows=rows,fragments=fragments,panel_hashes=[e['sha256'] for e in split_entries(study,phase,task,group,split)],execution_binding=evaluator.execution_binding)

class SearchIO:
    def __init__(self,study,phase,task,group,arm,cpu,credential_file):
        from .core import stable_seed
        from fuse_patch.native import TSP_DESCRIPTION,CVRP_DESCRIPTION,CVRP_FUNC_DESCRIPTION
        from fuse_patch.budget import EXECUTION_CAPABILITIES
        self.study=Path(study);self.phase=phase;self.task=task;self.group=group;self.arm=arm
        self.root=self.study/'searches'/phase/task/group/arm;self.root.mkdir(parents=True,exist_ok=True)
        cfg=dict(study_root=str(self.study),runtime_identity=ev.read(self.study/'engineering/RUNTIME_IDENTITY.json'),cpu_slot_lock_dir=str(self.study/'cpu-slots'),common_cache_roots=[] if arm=='INIT' else [str(self.root.parent/'INIT/evaluator')])
        self.evaluator=Evaluator(self.root/'evaluator',cpu,cfg)
        self.transport=Transport(self.root/'transport',self.study/'service',credential_file,run_id=f'{STUDY_ID}/{phase}/{task}/{group}/{arm}')
        self.context=dict(study_id=STUDY_ID,phase=phase,task=task,group=group,arm=arm,execution_binding=self.evaluator.execution_binding,
            feedback_panels=copy.deepcopy(split_entries(self.study,phase,task,group,'feedback')))
        self.task_description=(TSP_DESCRIPTION+' API: select_next_node(current_node,destination_node,unvisited_nodes,distance_matrix) -> int. Fixed destination is node0; NumPy unvisited array and full matrix; return one allowed integer node, never bool. ' if task=='tsp' else CVRP_DESCRIPTION+' '+CVRP_FUNC_DESCRIPTION)+EXECUTION_CAPABILITIES
        self.task_prompt=self.task_description;self.reference=None;self._shared=None;self._eval_roles={};self.started=time.time()
        ev.freeze(self.root/'STARTED.json',dict(unix=self.started,cpu=cpu,phase=phase,task=task,group=group,arm=arm))
        self.state=dict(context=self.context,programs={},qualities={},occurrences={},occurrence_counter=0,archive_counter=0,failures=[],native_auxiliary=[],init_completed=0,noninit_completed=0,attempted_source_hashes=[],feedback_attempts=0,scale=None)
    def source(self,h):
        b=(self.study/'programs'/f'{h}.py').read_bytes()
        if ev.digest(b)!=h:raise RuntimeError('SOURCE_BYTES_CHANGED')
        return b.decode()
    def save_program(self,source):return save_program(self.study,source)
    def evaluate(self,source,role):
        from fuse_patch.budget import DispatchStop
        cap=384 if self.arm in ('F2','F1','N2') else 256 if self.arm in ('H','E') else 160
        if self.state['feedback_attempts']>=cap:raise DispatchStop('FEEDBACK_ATTEMPT_CAP')
        self.state['feedback_attempts']+=1
        h=self.save_program(source)
        if h not in self.state['attempted_source_hashes']:self.state['attempted_source_hashes'].append(h)
        q=evaluate_program(self.study,self.phase,self.task,self.group,source,'feedback',role,self.evaluator)
        for i,f in enumerate([] if q.get('reused_paid_initialization') else q['fragments']):self._eval_roles[f'{role}/{i}']=ev.read(f['path'])
        self.state['qualities'][h]=q;ev.freeze(self.root/'qualities'/f'{role}.json',q)
        if q['hard_stop']:raise ev.EvaluationInfrastructureError('UNKNOWN_NO_REDISPATCH')
        return q
    register_program=inherited.SearchIO.register_program
    def attach_shared(self,shared):
        self._shared=copy.deepcopy(shared);self.state=copy.deepcopy(shared['archive_state']);self.state['context']=self.context
        self.reference=copy.deepcopy(shared['reference'])
        ev.freeze(self.root/'SHARED_INITIALIZATION_BINDING.json',dict(owner=shared['init_root'],sha256=ev.digest(ev.encoded(shared)),full_cost_charged=True,physical_reexecution=False))
    def checkpoint(self,tag):return ev.freeze(self.root/'states'/f'{tag}.json',self.state)
    def rng(self,purpose,index):
        from .core import stable_seed
        return stable_seed(STUDY_ID,self.phase,self.task,self.group,self.arm,purpose,index)
    def own_cost_snapshot(self):
        from .r1_transport import cost_frontier
        return cost_frontier(self)
    def cost_snapshot(self):
        own=self.own_cost_snapshot();shared=self._shared['cost'] if self._shared else {}
        cost=inherited.combine_cost(shared,own)
        return dict(cost,own_physical_cost=own,shared_initialization_cost=shared,initialization_charged_in_full=bool(self._shared),budget_semantics='known_usage_budget_with_unresolved_extra_cost' if cost.get('tokens_missing') else 'complete_usage_budget')
    def native_messages(self,messages):return self.transport.native_messages(messages)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',required=True);p.add_argument('--credentials',required=True);a=p.parse_args()
    bind_namespace();require_evaluation_open(a.submit);tr.sender(a.submit,a.credentials)
