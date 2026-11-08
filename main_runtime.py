"""DECADE method-preserving adapters to the frozen B1/B2 candidate ledger."""
from __future__ import annotations
import copy,contextlib,fcntl,hashlib,json,os,platform,subprocess,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import common
from common import ROOT,read,write,BudgetEnd,ServiceStop
from eval_budget import EvaluationBudget,RunSafetyLimit
from fuse_addon import evaluation as ev,transport as tr
from fuse_ane import runtime as rt,r1_transport as r1,native
from fuse_ane import sibling_policy as sp,sibling_view as view
from fuse_patch.budget import TokenBoundary,selection
from fuse_patch.runtime import SearchIO as OriginalIO
from fuse_ig.recovery_credentials import CREDENTIAL_FILE, credentials

ID='decade-f1-eval200-supplement-20261010'
EXECUTION_ID=ID+'-recovery-r2-alias'
REMOTE=Path(os.environ.get('DECADE_EXECUTION_ROOT', '/opt/decade/main')).resolve()
CREDENTIAL=str(CREDENTIAL_FILE)
_original_retryable=r1.retryable

def require_open(path,**kw):
 if platform.system()!='Linux' or ROOT!=REMOTE or not Path(path).resolve().is_relative_to(ROOT):raise RuntimeError('AUTHORIZED_MAIN_REMOTE_ONLY')
 for f in ['AUTHORITY.json','SOURCE_FREEZE.json']:
  if not (ROOT/'engineering'/f).is_file():raise RuntimeError('MISSING_EXECUTION_AUTHORITY')
 for f in ['STOP_DISPATCH.json','SEALED.json']:
  if (ROOT/f).exists():raise ServiceStop('DISPATCH_CLOSED:'+f)
 if (ROOT/'SEARCH_LOCK.json').exists():raise ServiceStop('SEARCH_ALREADY_LOCKED')

def stop(reason):
 try:write(ROOT/'STOP_DISPATCH.json',dict(reason=reason,unix=time.time(),inflight_settlement_preserved=True))
 except FileExistsError:pass

def retryable(folder,result):
 if result.get('http_status')!=400:return _original_retryable(folder,result)
 if result.get('fatal_reason') or result.get('service_success') or not result.get('body_complete'):return False
 try:
  raw=(Path(folder)/'response.body').read_text()
  try:
   obj=json.loads(raw);error=obj.get('error',obj);message=error.get('message') if isinstance(error,dict) else None
  except json.JSONDecodeError:message=raw
 except (OSError,UnicodeError,AttributeError):return False
 return message=='The model is overloaded. Please try again later.'

from fuse_ane.completion_rules import retry_wait
r1.retryable=retryable;r1.retry_delay=retry_wait
rt.STUDY=ROOT;rt.STUDY_ID=ID;rt.EXECUTION_ID=EXECUTION_ID;rt.require_open=require_open;rt.stop_dispatch=stop
tr.credentials=credentials

@contextlib.contextmanager
def http_lease():
 (ROOT/'service').mkdir(exist_ok=True)
 while True:
  require_open(ROOT)
  for i in range(2):
   f=(ROOT/'service'/f'http-{i}.lock').open('a+')
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:f.close();continue
   try:yield i
   finally:fcntl.flock(f,fcntl.LOCK_UN);f.close()
   return
  time.sleep(.25)

class Transport(rt.Transport):
 def __init__(self,run):
  self.run=run;self.root=run/'transport';self.credentials=CREDENTIAL;self.run_id=ID+'/'+run.name
  self.counter=0;self.records=[];self.logical_records=[];self.requests=[];self.config={}
  self.guard=threading.RLock();self._message_override=None;self.evaluation_budget=None
  ev.freeze(self.root/'BINDING.json',dict(run_id=self.run_id,sender_seconds=600,outer_seconds=660,connect_seconds=15,http_concurrency=2,
      max_physical_attempts=2,overload_400_exact_message='The model is overloaded. Please try again later.',model=rt.MODEL_SETTINGS,
      independent_cold_start=True,transport_diff_from_baseline='R2 bounded transport retained; baseline one-shot 300 seconds'))
 def request_payload(self,prompt,metadata):
  body=super().request_payload(prompt,metadata)
  if getattr(self,'_use_new_alias',False):body['model']='DeepSeek-V4.1-Flash'
  return body
 def _verify_model(self,terminal,metadata,folder):
  if not getattr(self,'_use_new_alias',False):return super()._verify_model(terminal,metadata,folder)
  model=terminal.get('response_model')
  if not terminal.get('service_success'):return
  if str(model).lower() not in ('deepseek-v4-flash','deepseek-v4.1-flash'):raise RuntimeError('UNAUTHORIZED_RESPONSE_MODEL_ALIAS')
  details=(terminal.get('usage') or {}).get('completion_tokens_details') or {}
  reasoning=details.get('reasoning_tokens')
  if reasoning is not None and reasoning!=0:raise RuntimeError('ACTUAL_THINKING_CAPABILITY_CONFLICT')
  ev.freeze(folder/'MODEL_CHECK.json',dict(response_model=model,canonical_research_model='DeepSeek-V4-Flash',
   requested_settings={k:v for k,v in self.request_payload('',metadata).items() if k!='messages'},
   usage=terminal.get('usage'),reasoning_tokens=reasoning,logical_id=terminal['logical_id'],metadata=metadata,
   alias_authority='MODEL_ALIAS_AUTHORIZATION.json',identity_equivalence='user-confirmed provider rename; not independently reprobed'))
 def _attempt(self,folder,body,frozen,identity,logical,metadata,attempt):
  # No re-entry or coordinate reuse is authorized for this newly launched study.
  self.evaluation_budget.guard_request()
  with http_lease():
   require_open(folder);self.evaluation_budget.guard_request()
   if (folder/'intent.json').exists():raise RuntimeError('EXISTING_INTENT_NO_AUTOMATIC_RESTART')
   tr.write(folder/'FROZEN_INPUT.json',frozen);tr.write_bytes(folder/'payload.json',body)
   idx=len(self.requests);physical=self.run/'requests'/f'{idx:06}'
   intent=dict(identity=identity,submitted_unix=time.time(),logical_id=logical+f'/attempt-{attempt:02d}',generation_id=logical,
      submission=1,transport_attempt=attempt,timeout_seconds=600,retries=attempt-1,metadata=metadata)
   tr.write(folder/'intent.json',intent)
   write(physical/'INTENT.json',dict(intent,request_id=idx,physical_path=str(folder),logical_request_id=logical,attempt=attempt))
   self.requests.append(dict(request_id=idx,total_tokens=None,success=False,pending=True))
   began=time.monotonic()
   try:
    done=subprocess.run([sys.executable,str(ROOT/'main_runtime.py'),'--send',str(folder)],capture_output=True,timeout=660)
    process=dict(returncode=done.returncode,wall_seconds=time.monotonic()-began,stderr=done.stderr.decode(errors='replace'))
   except subprocess.TimeoutExpired:process=dict(returncode=None,status='REQUEST_PROCESS_DEADLINE',wall_seconds=time.monotonic()-began)
   except OSError as exc:
    process=dict(returncode=None,status='SENDER_PROCESS_START_FAILED',exception_type=type(exc).__name__,wall_seconds=time.monotonic()-began)
    tr.write(folder/'SENDER_HOST_FAILURE.json',dict(process,definitely_pre_http=True,payload_sha256=identity['payload_sha256']))
   tr.write(folder/'process.json',process);tr.settle_existing(folder)
   result=tr.classify(folder)
   result.update(identity=identity,logical_id=intent['logical_id'],generation_id=logical,transport_attempt=attempt,request_path=str(folder),
       submitted_unix=intent['submitted_unix'],metadata=metadata,network_boundary_entered=(folder/'NETWORK_BOUNDARY_INTENT.json').exists())
   tr.write(folder/'terminal.json',result)
   total=(result.get('usage') or {}).get('total_tokens')
   if not isinstance(total,(int,float)) or isinstance(total,bool) or total<0:total=None
   rec=dict(request_id=idx,logical_request_id=logical,physical_path=str(folder),attempt=attempt,http_status=result.get('http_status'),
       success=bool(result.get('service_success')),usage=result.get('usage'),total_tokens=total,error=result.get('error'),status=result['status'],
       metadata=metadata,elapsed_seconds=process['wall_seconds'],currency=None,payload_sha256=identity['payload_sha256'],finish_reason=result.get('finish_reason'))
   self.requests[idx]=rec;write(physical/'RESULT.json',rec)
   return result

class InitBoundary(TokenBoundary):
 def __init__(self,io):
  super().__init__(io);self.prefixes=();io.transport._budget_boundary=self
 def before_dispatch(self,**kw):self.search.budget.guard_request();require_open(self.search.root)
 def settled_failed_attempt(self,result):pass
 def retry_budget_available(self):self.before_dispatch();return True
 def observe(self,kind,**detail):
  self.event_counter+=1
  return ev.freeze(self.search.root/'init_boundaries'/f'{self.event_counter:06}.json',dict(kind=kind,position=self.position,
      pending_position=self._pending_position,evaluations=self.search.budget.started,physical_requests=len(self.search.transport.requests),**detail))
 def finish(self,population,native_state=None,*,status,**kw):
  self.end_batch(population,native_state);self.search.checkpoint('INIT_COMPLETE')
  return dict(request_set=list(range(len(self.search.transport.requests))),prefixes={},status=status,selection=selection(self.search))
native.Boundary=InitBoundary

class SearchIO:
 def __init__(self,spec,cpu):
  from fuse_patch.native import TSP_DESCRIPTION,CVRP_DESCRIPTION,CVRP_FUNC_DESCRIPTION
  from fuse_patch.budget import EXECUTION_CAPABILITIES
  self.root=ROOT/'runs'/spec['id'];self.root.mkdir(parents=True,exist_ok=True)
  self.study=ROOT;self.phase='seen_eval200';self.task=spec['task'];self.group=spec['group'];self.arm=spec['method'];self.seed=spec['search_seed']
  self.started=time.time();self.candidate_id=0;self.stage='init';self.reference=None;self.current_pre=None
  self.transport=Transport(self.root);self.evaluator=common.Evaluator(self.root,self.task,self.group,cpu)
  self.budget=EvaluationBudget(self.evaluator,self.transport,self.root,200,BudgetEnd);self.transport.evaluation_budget=self.budget
  panels=sorted((ROOT/'panels'/self.task/self.group).glob('feedback-*.json'))
  self.panel_files=panels;self.panel_hashes=[ev.digest(x.read_bytes()) for x in panels]
  self.instances=[x for p in panels for x in read(p)['instances']]
  self.binding=dict(study_id=ID,task=self.task,panel_hashes=self.panel_hashes,runtime_identity=read(ROOT/'engineering/RUNTIME.json'),
    evaluator_source_sha256=ev.digest((ROOT/'common.py').read_bytes()),rpc_source_sha256=ev.digest((ROOT/'fuse_patch/rpc_host.py').read_bytes()),
    candidate_cpus=1,memory_gib=8,load_seconds=10,feedback_fragment_seconds=500)
  self.context=dict(study_id=ID,phase=self.phase,task=self.task,group=self.group,arm=self.arm,execution_binding=self.binding,
    feedback_panels=[dict(path=str(p.relative_to(ROOT)),sha256=h,count=len(read(p)['instances'])) for p,h in zip(panels,self.panel_hashes)])
  self.task_description=(TSP_DESCRIPTION+' API: select_next_node(current_node,destination_node,unvisited_nodes,distance_matrix) -> int. Fixed destination is node0; NumPy unvisited array and full matrix; return one allowed integer node, never bool. ' if self.task=='tsp' else CVRP_DESCRIPTION+' '+CVRP_FUNC_DESCRIPTION)+EXECUTION_CAPABILITIES
  self.task_prompt=self.task_description
  self.state=dict(context=self.context,programs={},qualities={},occurrences={},occurrence_counter=0,archive_counter=0,failures=[],native_auxiliary=[],init_completed=0,noninit_completed=0,attempted_source_hashes=[],feedback_attempts=0,scale=None)
  write(self.root/'START.json',dict(spec,unix=self.started,cpu=cpu,online_edit_memory=[],cold_start=False,original_cold_start=True,paid_initialization_prefix_replayed=True,initialization_policy='INITIALIZATION_POLICY.json'))
 def save_program(self,source):
  h=ev.digest(source);ev.freeze_bytes(self.root/'programs'/f'{h}.py',source.encode());return h
 def source(self,h):
  source=(self.root/'programs'/f'{h}.py').read_text();assert ev.digest(source)==h;return source
 def evaluation_exhausted(self):return self.budget.started>=self.budget.limit
 def _candidate(self,source,role):
  self.candidate_id+=1;folder=self.root/'candidates'/f'{self.candidate_id:06}'
  h=self.save_program(source);ev.freeze_bytes(folder/'assembled.py',source.encode())
  write(folder/'BINDING.json',dict(candidate_id=self.candidate_id,source_sha256=h,phase=self.stage,role=role,
     owner_request_id=len(self.transport.requests)-1 if self.transport.requests else None,
     logical_request_id=self.transport.logical_records[-1]['logical_id'] if self.transport.logical_records else None,
     actual_parent_sha=(self.current_pre or {}).get('parent_sha'),normal_slots=(self.current_pre or {}).get('normal_slot_numbers'),
     trajectory=self.root.name,execution_contract_sha256=ev.digest(ev.encoded(self.binding))))
  return folder,h
 def retain_unevaluated(self,source,role,slot=None):
  folder,h=self._candidate(source,role)
  write(folder/'RESULT.json',dict(status='GENERATED_NOT_EVALUATED_BUDGET_LIMIT',valid=False,source_sha256=h,quality=None,target_credit=None,counts_against_budget=False,slot_id=(slot or {}).get('slot_id')))
 def evaluate(self,source,role):
  folder,h=self._candidate(source,role)
  result=self.budget.evaluate(source,'feedback',self.candidate_id)
  write(folder/'RESULT.json',result)
  valid=result['valid'];rows=[]
  raw=self.root/'evaluations/feedback'/h
  for panel in self.panel_files:
   for index,item in enumerate(read(panel)['instances']):
    record=raw/panel.stem/f'{index:03}'/'RESULT.json'
    if record.exists():
     row=read(record)
     if row['instance_id']!=item['instance_id']:raise RuntimeError('ACTUAL_INSTANCE_ID_MISMATCH')
     rows.append(row)
    else:rows.append(dict(instance_id=item['instance_id'],status='NOT_EXECUTED',length=None))
  if valid and (len(rows)!=64 or any(x['status']!='EVALUATED' for x in rows)):raise RuntimeError('PAIRED_ROWS_INCOMPLETE')
  q=dict(source_hash=h,mean=result['loss'],complete=valid,status='EVALUATED' if valid else 'INFRASTRUCTURE_TIMEOUT' if result.get('error')=='CONTAINER_CREATE_TIMEOUT_NO_CANDIDATE_START' else 'ALGORITHM_FAILURE',hard_stop=False,
      phase=self.phase,task=self.task,group=self.group,split='feedback',denominator=64,completed=64 if valid else None,rows=rows,
      actual_instances_attempted=result['evaluated_instances'],panel_hashes=self.panel_hashes,execution_binding=self.binding,
      evaluation_index=result['evaluation_index'],counts_against_budget=result['counts_against_budget'],physical_summary=str(self.root/'evaluations/feedback'/h/'SUMMARY.json'))
  self.state['qualities'][h]=q;self.state['feedback_attempts']=self.budget.started
  if h not in self.state['attempted_source_hashes']:self.state['attempted_source_hashes'].append(h)
  ev.freeze(folder/'PAIRED_QUALITY.json',q)
  if result['counts_against_budget'] and result['evaluation_index'] in (50,100,150,200):
   ev.freeze(self.root/'checkpoints'/str(result['evaluation_index'])/'FRONTIER.json',dict(evaluation_index=result['evaluation_index'],
      current_batch_id=(self.current_pre or {}).get('batch_id'),current_role=role,online_archive_before_batch_update=sorted(self.state['programs']),
      pending_repair_possible=self.current_pre is not None and not self.evaluation_exhausted(),eligibility='only evaluation ledger indices <= checkpoint',
      actual_completed_prefix=list(range(1,result['evaluation_index']+1)),cost_request_ids=list(range(len(self.transport.requests)))))
  return q
 register_program=OriginalIO.register_program
 def checkpoint(self,tag):return ev.freeze(self.root/'states'/f'{tag}.json',self.state)
 def rng(self,purpose,index):
  if purpose in ('native-python','native-numpy') and index==0:return self.seed
  return int(hashlib.sha256(f'{self.seed}/{purpose}/{index}'.encode()).hexdigest()[:16],16)
 def own_cost_snapshot(self):
  req=self.transport.requests;known=sum(r['total_tokens'] for r in req if r.get('total_tokens') is not None);missing=sum(r.get('total_tokens') is None for r in req)
  return dict(http=len(req),tokens=known if not missing else None,tokens_known=known,tokens_missing=missing,currency=None,cpu_seconds=None)
 cost_snapshot=own_cost_snapshot
 def native_messages(self,messages):return self.transport.native_messages(messages)

def search(spec,cpu):
 if spec['method']!='F1':raise RuntimeError('F1_ONLY_SCOPE')
 io=None;status='UNSTARTED';error=None
 try:
  require_open(ROOT)
  io=SearchIO(spec,cpu)
  native.run_native(io,initialization_only=True)
  io.transport._budget_boundary=None;io.stage='search';io.state['initialization_feedback_attempts']=io.budget.started
  memory=[];position=0;repairs=0;pair_used={}
  write(io.root/'EDIT_SEARCH_START.json',dict(initialization_evaluations=io.budget.started,memory=[],archive=sorted(io.state['programs'])))
  while not io.evaluation_exhausted():
   io.budget.guard_request();require_open(io.root)
   numbers=[position+1];pair=(numbers[0]-1)//2
   pre=view.prepare(io,memory,numbers);io.current_pre=pre
   step=io.root/'batches'/f'{numbers[0]:06}';ev.freeze(step/'PRE.json',pre)
   plans,index,rid=sp._send(io,pre,step/'initial')
   slots,records=sp._evaluate_slots(io,pre,plans,index,rid,step/'initial');position=numbers[-1]
   eligible=next((s for s in slots if s.get('repairable') and not s.get('status','').startswith('DUPLICATE_')),None)
   repaired=None
   if eligible and not pair_used.get(pair,False) and not io.evaluation_exhausted():
    io.budget.guard_request();require_open(io.root)
    rpre=view.prepare_repair(pre,eligible['slot_id']);io.current_pre=rpre
    plans,ix,rrid=sp._send(io,rpre,step/'repair',repair_slot=eligible['slot_id'],failed=eligible)
    pair_used[pair]=True
    rslots,rrecords=sp._evaluate_slots(io,rpre,plans,ix,rrid,step/'repair',repair=True)
    records+=rrecords;repairs+=1;repaired=rslots[0]['proposal_id']
   memory.extend(records);io.current_pre=None
   ev.freeze(step/'RESULT.json',dict(normal_slots=numbers,pair_index=pair,repair_allowance_used=bool(pair_used.get(pair,False)),shared_parent=pre['parent_sha'],initial_request_id=rid,
      repaired_proposal_id=repaired,memory_evidence_ids=[r['evidence_id'] for r in memory],fully_settled=True,
      evaluation_frontier=io.budget.snapshot(),physical_requests=len(io.transport.requests),repairs_so_far=repairs))
   io.checkpoint(f'BATCH-{position:06}')
  status='EVALUATION_BUDGET_COMPLETED'
 except BudgetEnd:status='EVALUATION_BUDGET_COMPLETED' if io and io.budget.started==200 else 'BUDGET_STOP_INCOMPLETE'
 except RunSafetyLimit as exc:status='SAFETY_STOPPED';error=str(exc)
 except BaseException as exc:
  status='INFRASTRUCTURE_STOPPED';error=f'{type(exc).__name__}: {exc}';stop(error)
  import traceback
  if io:ev.freeze_bytes(io.root/'ERROR_TRACE.txt',traceback.format_exc().encode())
 finally:
  run=ROOT/'runs'/spec['id']
  if io:io.checkpoint('FINAL')
  end=dict(spec,status=status,error=error,unix=time.time(),evaluation_budget=io.budget.snapshot() if io else dict(started=0,completed=0,limit=200),
      cost=io.own_cost_snapshot() if io else None,normal_completion=status=='EVALUATION_BUDGET_COMPLETED')
  write(run/'END.json',end)
 return end

if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--send':
  # Intent is already committed: honor its deadline even if another run has since stopped dispatch.
  folder=Path(sys.argv[2]).resolve()
  if ROOT!=REMOTE or not folder.is_relative_to(ROOT) or not (folder/'intent.json').exists():raise RuntimeError('SENDER_SCOPE')
  credentials(CREDENTIAL);tr.sender(folder,CREDENTIAL)
 else:raise SystemExit('Use run_study.py')
