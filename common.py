from __future__ import annotations
import contextlib
import random
import numpy as np
import os, sys, json, hashlib, time, threading, shlex, urllib.request, urllib.error, math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from fuse_patch import rpc_host as rpc
from fuse_addon import evaluation as ev
from fuse_ig.recovery_credentials import CREDENTIAL_FILE, BASE_URL

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)
 with path.open('x') as f:f.write(data+'\n')
 return value

def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
class BudgetEnd(BaseException):pass
class ServiceStop(BaseException):pass

def setup_rpc():
 cfg=read(ROOT/'engineering/RUNTIME.json')
 ev.IMAGE=cfg['image'];ev.RUNTIME=cfg.get('runtime',ev.RUNTIME);rpc.REMOTE_ROOT=ROOT;rpc.CPUS=tuple(str(x) for x in cfg['cpus'])
 def guard(source,output,public):
  if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):raise RuntimeError('BASELINE_NAMESPACE_REQUIRED')
  if not (ROOT/'engineering/AUTHORITY.json').exists():raise RuntimeError('AUTHORITY_MISSING')
 rpc._guard=guard

class Evaluator:
 def __init__(self,run,task,group,cpu):
  setup_rpc();self.run=Path(run);self.task=task;self.group=group;self.cpu=str(cpu)
  self.records=[];self.lock=threading.RLock()
 def evaluate(self,source,split='feedback',owner=None):
  with self.lock:
   if split=='screen' and not (ROOT/'SEARCH_LOCK.json').exists():raise RuntimeError('SEARCH_NOT_LOCKED')
   if split=='test' and not (ROOT/'SELECTION_LOCK.json').exists():raise RuntimeError('SELECTION_NOT_LOCKED')
   h=digest(source.encode());p=self.run/'programs'/f'{h}.py';p.parent.mkdir(exist_ok=True)
   if not p.exists():p.write_text(source)
   cache=self.run/'evaluations'/split/h
   if (cache/'SUMMARY.json').exists():
    prior=read(cache/'SUMMARY.json');result=dict(prior,duplicate=True,owner=owner)
   else:
    rows=[];complete=True;error=None;t0=time.time()
    panels=sorted((ROOT/'panels'/self.task/self.group).glob(split+'-*.json'))
    if not panels:raise RuntimeError('PANEL_MISSING')
    for panel_path in panels:
     panel=read(panel_path)
     deadline=time.monotonic()+500.0 if split=='feedback' else None
     for i,item in enumerate(panel['instances']):
      row=rpc.run_instance(p,item,task='TSP-Construct' if self.task=='tsp' else 'CVRP-Construct',cpu=self.cpu,output_dir=cache/panel_path.stem/f'{i:03}',limits={'load_seconds':10,'route_seconds':None if split=='feedback' else 300,'panel_seconds':500 if split=='feedback' else None},panel_deadline=deadline)
      rows.append(row)
      if row.get('hard_stop'):
       write(cache/'INFRASTRUCTURE_FAILURE.json',row)
       try:write(ROOT/'STOP_DISPATCH.json',{'reason':'EVALUATOR_INFRASTRUCTURE_UNKNOWN','run':str(self.run)})
       except FileExistsError:pass
       raise ServiceStop('EVALUATOR_INFRASTRUCTURE_UNKNOWN')
      if row.get('status')!='EVALUATED':complete=False;error=row.get('error_type') or row['status'];break
     if not complete:break
    vals=[r['length'] for r in rows if r.get('length') is not None]
    expected=sum(len(read(x)['instances']) for x in panels)
    complete=complete and len(vals)==expected
    result={'source_sha256':h,'split':split,'loss':sum(vals)/len(vals) if complete else None,'lengths':vals if complete else None,'valid':complete,'error':error,'evaluated_instances':len(rows),'expected_instances':expected,'wall_seconds':time.time()-t0+sum(r.get('wall_seconds',0) for r in rows if r.get('recovery_reused_instance')), 'duplicate':False,'owner':owner,'cpu_seconds':None,'cpu_note':'see per-instance host and container accounting; no missing values imputed'}
    write(cache/'SUMMARY.json',result)
   rec=dict(result,index=len(self.records))
   write(self.run/'events'/f'{len(self.records):06}.json',rec);self.records.append(rec)
   return result

class ResponseText(str):
 def __reduce__(self):return (ResponseText,(str(self),self.request_id))
 def __new__(cls,s,request_id):
  x=super().__new__(cls,s);x.request_id=request_id;return x

class Client:
 def __init__(self,run,budget):
  self.run=Path(run);self.budget=budget;self.tokens=0;self.requests=[];self.lock=threading.RLock();self.temperature=1.0;self.model='DeepSeek-V4-Flash';self.local=threading.local();self.stop_reason=None
  credential=CREDENTIAL_FILE
  values={}
  for line in credential.read_text().splitlines():
   if not line.strip() or line.lstrip().startswith('#'):continue
   k,_,v=line.removeprefix('export ').partition('=');values[k.strip()]=shlex.split(v)[0]
  assert values['BASE_URL']==BASE_URL and values['MODEL_NAME']==self.model
  self.url=values['BASE_URL']+'/chat/completions';self.key=values['API_KEY']
 def ask(self,messages,temperature=None):
  with self.lock:
   with self.evaluation_budget.admission_lock if hasattr(self,'evaluation_budget') else contextlib.nullcontext():
    if self.stop_reason:raise ServiceStop(self.stop_reason)
    if hasattr(self,'evaluation_budget'):self.evaluation_budget.guard_request()
    if self.budget is not None and self.tokens>=self.budget:raise BudgetEnd()
    if (ROOT/'STOP_DISPATCH.json').exists():raise ServiceStop('USER_OR_SERVICE_STOP')
    idx=len(self.requests);folder=self.run/'requests'/f'{idx:06}'
    body={'model':self.model,'messages':messages,'temperature':1.0 if temperature is None else float(temperature),'top_p':1.0,'max_tokens':8192,'stream':False,'thinking':{'type':'disabled'}}
    write(folder/'payload.json',body);write(folder/'INTENT.json',{'request_id':idx,'submitted_unix':time.time(),'tokens_before':self.tokens,'attempt':1,'python_rng_state':random.getstate(),'numpy_rng_state':{'algorithm':np.random.get_state()[0],'state':np.random.get_state()[1].tolist(),'position':int(np.random.get_state()[2]),'has_gauss':int(np.random.get_state()[3]),'cached_gaussian':float(np.random.get_state()[4])},'unknown_not_automatically_retried':True})
   req=urllib.request.Request(self.url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
   start=time.time();status=None;raw=b'';error=None
   try:
    with urllib.request.urlopen(req,timeout=300) as response:status=response.status;raw=response.read()
   except urllib.error.HTTPError as exc:status=exc.code;raw=exc.read();error='HTTP_'+str(status)
   except Exception as exc:error=type(exc).__name__
   (folder/'response.body').write_bytes(raw)
   try:parsed=json.loads(raw)
   except Exception:parsed={}
   usage=parsed.get('usage');total=usage.get('total_tokens') if isinstance(usage,dict) else None
   if isinstance(total,int) and total>0:self.tokens+=total
   else:total=None
   choices=parsed.get('choices') or [];content=(choices[0].get('message') or {}).get('content') if choices else None
   rec={'request_id':idx,'http_status':status,'error':error,'usage':usage,'total_tokens':total,'known_cumulative_tokens':self.tokens,'elapsed_seconds':time.time()-start,'finish_reason':choices[0].get('finish_reason') if choices else None,'response_sha256':digest(raw),'success':status==200 and isinstance(content,str),'currency':None}
   self.requests.append(rec);write(folder/'RESULT.json',rec);self.local.last_request=idx
   self.local.request_ids=getattr(self.local,'request_ids',[])+[idx]
   if not rec['success']:
    self.stop_reason=error or 'UNRESOLVED_OR_INVALID_RESPONSE';write(self.run/'SERVICE_STOP.json',rec)
    try:write(ROOT/'STOP_DISPATCH.json',{'reason':self.stop_reason,'run':str(self.run),'request':idx})
    except FileExistsError:pass
    raise ServiceStop(self.stop_reason)
   if total is None:
    self.stop_reason='USAGE_MISSING_COST_NOT_IDENTIFIABLE';write(self.run/'SERVICE_STOP.json',rec)
    try:write(ROOT/'STOP_DISPATCH.json',{'reason':self.stop_reason,'run':str(self.run),'request':idx})
    except FileExistsError:pass
   return ResponseText(content,idx)
 def chat_completion(self,n,messages,temperature=None):
  return [SimpleNamespace(message=SimpleNamespace(content=self.ask(messages,temperature))) for _ in range(n)]
 def multi_chat_completion(self,messages_list,n=1,temperature=None):
  if not isinstance(messages_list[0],list):messages_list=[messages_list]
  result=[]
  for messages in messages_list:
   for _ in range(n):
    try:result.append(self.ask(messages,temperature))
    except BudgetEnd:
     if result:return result
     raise
  return result

TSP_SEED='import numpy as np\ndef select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):\n return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])\n'
CVRP_SEED='def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):\n feasible=[int(n) for n in unvisited_nodes if demands[int(n)]<=rest_capacity]\n return min(feasible,key=lambda n: (distance_matrix[current_node,n],n)) if feasible else int(depot)\n'
TSP_DESCRIPTION='Design a step-by-step constructive heuristic for the Euclidean Traveling Salesman Problem with 50 nodes, minimizing total closed-tour distance.'
CVRP_DESCRIPTION='Design a step-by-step constructive heuristic for the Capacitated Vehicle Routing Problem with 50 customers, uniform Euclidean coordinates, integer customer demands 1 to 9 and vehicle capacity 40, minimizing total travel distance.'
TSP_INPUTS=['current_node','destination_node','unvisited_nodes','distance_matrix']
CVRP_INPUTS=['current_node','depot','unvisited_nodes','rest_capacity','demands','distance_matrix']
TSP_CONTRACT='current_node and destination_node are integer node IDs. destination_node is 0. unvisited_nodes is a NumPy array in distance-neighbor order; distance_matrix is the complete NumPy distance matrix. Return an integer next_node from unvisited_nodes, never bool.'
CVRP_CONTRACT='current_node and depot are integer node IDs. depot is 0. unvisited_nodes contains ALL unvisited customers, including those that may not fit. rest_capacity is the remaining capacity, demands is the full NumPy demand array and distance_matrix the full NumPy distance matrix. Return an integer next_node: an unvisited customer whose demand fits rest_capacity, or depot only when current_node is not depot. Never bool. No split deliveries.'
