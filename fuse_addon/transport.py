"""One-shot model calls and a small shared durable availability circuit.

Request bodies/intents precede I/O. Existing intents are reconciled, never sent
again. SQLite transactions keep the shared breaker consistent without retaining
quadratic copies of every historical response in every state snapshot.
"""
from __future__ import annotations
import argparse, hashlib, http.client, json, os, shlex, sqlite3, subprocess, sys, time
from pathlib import Path
from urllib.parse import urlsplit

MODEL='DeepSeek-V4-Flash'

def sha(data):
    return hashlib.sha256(data if isinstance(data,bytes) else data.encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def write_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('IMMUTABLE_RECORD_CONFLICT: '+str(path))
        return data
    tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.'+os.urandom(6).hex()+'.pending')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    try:os.link(tmp,path);fsync_directory(path.parent)
    except FileExistsError:
        if path.read_bytes()!=data:raise RuntimeError('IMMUTABLE_RECORD_CONFLICT: '+str(path))
    finally:tmp.unlink(missing_ok=True);fsync_directory(path.parent)
    return data
def write(path,value):
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    write_bytes(path,data)
    return value

def credentials(path):
    vals={}
    for line in Path(path).read_text().splitlines():
        k,sep,v=line.strip().removeprefix('export ').partition('=')
        if sep and k.strip() in ('BASE_URL','API_KEY','MODEL_NAME'):
            parsed=shlex.split(v,comments=True)
            if len(parsed)!=1:raise ValueError('INVALID_CREDENTIAL_FIELD')
            vals[k.strip()]=parsed[0]
    url=urlsplit(vals['BASE_URL'])
    if url.scheme!='https' or url.hostname!='api.example.invalid' or url.username or url.password or url.query or url.fragment:raise ValueError('UNAUTHORIZED_SERVICE_BINDING')
    if vals['MODEL_NAME'].lower()!=MODEL.lower():raise ValueError('UNAUTHORIZED_MODEL_BINDING')
    return vals,url.netloc,url.path.rstrip('/')+'/chat/completions'

def payload(prompt):
    return dict(model=MODEL,messages=[dict(role='user',content=prompt)],thinking={'type':'disabled'},max_tokens=16384,stream=False)

def sender(folder,credential_file):
    folder=Path(folder);body=(folder/'payload.json').read_bytes();h=sha(body);began=time.monotonic();conn=None;chunks=[];meta={}
    intent=read(folder/'intent.json');deadline=float(intent['submitted_unix'])+float(intent.get('timeout_seconds',150));limit=max(0.,deadline-time.time())
    write(folder/'SENDER_STARTED.json',dict(payload_sha256=h,started_unix=time.time(),pid=os.getpid(),parent_pid=os.getppid(),deadline_unix=deadline,network_boundary_entered=False))
    def boundary(phase):
        p=folder/'NETWORK_BOUNDARY_INTENT.json'
        if not p.exists():write(p,dict(payload_sha256=h,phase=phase,time_unix=time.time(),provider_execution_confirmed=False))
    class Connection(http.client.HTTPSConnection):
        def send(self,data):boundary('SEND_BEFORE_NETWORK');return super().send(data)
        def connect(self):boundary('CONNECT_BEFORE_NETWORK');return super().connect()
    try:
        if limit<=0:raise TimeoutError('ORIGINAL_DEADLINE_BEFORE_SENDER_START')
        vals,host,path=credentials(credential_file);conn=Connection(host,timeout=min(15,limit))
        conn.request('POST',path,body=body,headers={'Authorization':'Bearer '+vals['API_KEY'],'Content-Type':'application/json'})
        conn.sock.settimeout(max(.001,limit-(time.monotonic()-began)));response=conn.getresponse()
        meta=dict(http_status=response.status,request_id=response.getheader('x-request-id'),retry_after=response.getheader('Retry-After'))
        while True:
            remain=limit-(time.monotonic()-began)
            if remain<=0:raise TimeoutError('TOTAL_REQUEST_DEADLINE')
            conn.sock.settimeout(remain);chunk=response.read1(65536)
            if not chunk:break
            chunks.append(chunk)
        data=b''.join(chunks);write_bytes(folder/'response.body',data)
        write(folder/'response.metadata.json',dict(**meta,wall_seconds=time.monotonic()-began,body_complete=True))
    except Exception as exc:
        if chunks:write_bytes(folder/'response.partial.body',b''.join(chunks))
        error=dict(exception_type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-began,**meta)
        write(folder/'sender.error.json',error)
        if not (folder/'NETWORK_BOUNDARY_INTENT.json').exists():write(folder/'SENDER_HOST_FAILURE.json',dict(**error,definitely_pre_http=True,payload_sha256=h))
    finally:
        if conn is not None:conn.close()

def classify(folder):
    folder=Path(folder)
    if not (folder/'response.body').exists() or not (folder/'response.metadata.json').exists():return dict(status='UNKNOWN',service_success=False,content=None,usage=None,fatal_reason=None,quality=None)
    meta=read(folder/'response.metadata.json');status=meta['http_status'];fatal=None
    try:obj=json.loads((folder/'response.body').read_bytes())
    except (ValueError,UnicodeError):obj=None
    detail=json.dumps(obj.get('error') if isinstance(obj,dict) else None,ensure_ascii=False).lower()
    if status in (401,403):fatal='SERVICE_AUTH_REJECTED'
    elif status==402 or any(x in detail for x in ('insufficient balance','insufficient_balance','quota exhausted','\u4f59\u989d\u4e0d\u8db3')):fatal='SERVICE_BALANCE_REJECTED'
    elif status==413 or any(x in detail for x in ('context_length_exceeded','context length','context window','input too long')):fatal='SERVICE_CONTEXT_REJECTED'
    elif any(x in detail for x in ('model_not_found','model not found','invalid model')):fatal='SERVICE_MODEL_REJECTED'
    result=dict(status='SERVICE_ERROR',service_success=False,content=None,usage=None,fatal_reason=fatal,**meta)
    if not isinstance(obj,dict):return dict(result,error='MALFORMED_SERVICE_ENVELOPE')
    result.update(usage=obj.get('usage'),response_id=obj.get('id'),response_model=obj.get('model'),error=obj.get('error'))
    if obj.get('model') and str(obj['model']).lower() not in (MODEL.lower(),'deepseek-v4.1-flash'):result['fatal_reason']='SERVICE_RESPONSE_MODEL_MISMATCH'
    choices=obj.get('choices')
    if status==200 and not obj.get('error') and isinstance(choices,list) and choices and isinstance(choices[0],dict) and isinstance(choices[0].get('message'),dict) and isinstance(choices[0]['message'].get('content'),str) and not result['fatal_reason']:
        result.update(status='RESPONSE_RECEIVED',service_success=True,content=choices[0]['message']['content'],finish_reason=choices[0].get('finish_reason'))
    return result

def settle_existing(folder):
    """Reconcile the original sender up to its original deadline, never resend.

    A controller can die while its sender remains alive.  Do not create an
    UNKNOWN merely because the replacement controller arrived before that
    original external action's deadline or persisted process completion.
    """
    folder=Path(folder);intent=read(folder/'intent.json')
    deadline=float(intent['submitted_unix'])+float(intent.get('timeout_seconds',150))
    while time.time()<deadline:
        settled=((folder/'response.body').exists() and (folder/'response.metadata.json').exists())
        settled=settled or any((folder/name).exists() for name in ('process.json','sender.error.json','SENDER_HOST_FAILURE.json'))
        if settled:return
        time.sleep(min(.2,max(0,deadline-time.time())))

class ServiceStopped(RuntimeError):pass
class HostFailure(RuntimeError):pass
class Circuit:
    def __init__(self,root,clock=time.time):
        self.path=Path(root)/'service.sqlite3';self.path.parent.mkdir(parents=True,exist_ok=True);self.clock=clock
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS observations (id TEXT PRIMARY KEY, hash TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL, id TEXT, kind TEXT, value TEXT)')
            db.execute('INSERT OR IGNORE INTO state VALUES (1,?)',(canonical(dict(status='CLOSED',failures=0,inflight={},waiters={},recovered=False,retry_after=0,cooldown_until=None,stop_reason=None)),))
    def connect(self):
        db=sqlite3.connect(self.path,timeout=60);db.execute('PRAGMA synchronous=FULL');return db
    def transact(self,fn):
        db=self.connect()
        try:
            db.execute('BEGIN IMMEDIATE');s=json.loads(db.execute('SELECT value FROM state WHERE id=1').fetchone()[0]);result,event=fn(db,s)
            db.execute('UPDATE state SET value=? WHERE id=1',(canonical(s),))
            if event:db.execute('INSERT INTO events(time,id,kind,value) VALUES(?,?,?,?)',(self.clock(),event[0],event[1],canonical(s)))
            db.commit();return result
        except BaseException:db.rollback();raise
        finally:db.close()
    def snapshot(self):
        with self.connect() as db:return json.loads(db.execute('SELECT value FROM state WHERE id=1').fetchone()[0])
    def before(self,logical,folder,identity):
        def fn(db,s):
            known=db.execute('SELECT hash FROM observations WHERE id=?',(logical,)).fetchone()
            pending=s['inflight'].get(logical)
            if pending and pending['identity']!=identity:raise RuntimeError('SERVICE_REQUEST_IDENTITY_CONFLICT')
            if known or (folder/'intent.json').exists():return ('RECONCILE',None),None
            if s['status']=='STOPPED':raise ServiceStopped(s['stop_reason'])
            if pending:return ('SEND',None),None
            if s['status'] in ('DRAINING','COOLDOWN','HALF_OPEN'):
                if logical in s['waiters'] and s['waiters'][logical]!=identity:raise RuntimeError('WAITING_REQUEST_CHANGED')
                s['waiters'][logical]=identity
                if not(s['status']=='COOLDOWN' and not s['inflight'] and self.clock()>=s['cooldown_until'] and logical==min(s['waiters'])):return ('WAIT',s.get('cooldown_until')),None
                s['status']='HALF_OPEN'
            half=s['status']=='HALF_OPEN';s['waiters'].pop(logical,None);s['inflight'][logical]=dict(identity=identity,half_open=half)
            return ('SEND',None),(logical,'RESERVED_HALF_OPEN' if half else 'RESERVED')
        return self.transact(fn)
    def observe(self,logical,result,folder):
        persisted=read(folder/'terminal.json')
        if canonical(persisted)!=canonical(result):raise RuntimeError('TERMINAL_NOT_PERSISTED')
        terminal_hash=sha((folder/'terminal.json').read_bytes())
        def fn(db,s):
            prior=db.execute('SELECT hash FROM observations WHERE id=?',(logical,)).fetchone()
            if prior:
                if prior[0]!=terminal_hash:raise RuntimeError('OBSERVED_TERMINAL_CHANGED')
                return s,None
            pending=s['inflight'].pop(logical,None)
            if pending is None:raise RuntimeError('TERMINAL_WITHOUT_RESERVATION')
            if result.get('identity')!=pending['identity'] or result.get('logical_id')!=logical:raise RuntimeError('TERMINAL_REQUEST_IDENTITY_CONFLICT')
            db.execute('INSERT INTO observations VALUES (?,?)',(logical,terminal_hash))
            success=result['service_success'];fatal=result.get('fatal_reason')
            if fatal:s.update(status='STOPPED',stop_reason=fatal)
            elif s['status']=='STOPPED':pass
            elif pending['half_open']:
                if success:s.update(status='CLOSED',failures=0,recovered=False,retry_after=0,cooldown_until=None)
                else:s.update(status='STOPPED',stop_reason='SERVICE_UNAVAILABLE',failures=s['failures']+1)
            elif success:
                s['failures']=0
                if s['status']=='DRAINING':s['recovered']=True
                else:s['retry_after']=0
            else:
                s['failures']+=1
                try:ra=max(0,float(result.get('retry_after') or 0))
                except (ValueError,TypeError):
                    try:
                        from email.utils import parsedate_to_datetime
                        ra=max(0,parsedate_to_datetime(result['retry_after']).timestamp()-self.clock())
                    except Exception:ra=0
                s['retry_after']=max(s['retry_after'],ra)
                if s['failures']>=2:s.update(status='DRAINING',recovered=False)
            if s['status']=='DRAINING' and not s['inflight']:
                if s['recovered'] and s['failures']<2:s.update(status='CLOSED',recovered=False,retry_after=0,cooldown_until=None)
                else:s.update(status='COOLDOWN',cooldown_until=self.clock()+max(60,s['retry_after']))
            return s,(logical,'TERMINAL_'+result['status'])
        return self.transact(fn)

class NativeTransport:
    def __init__(self,run_dir,service_root,credential_file,run_id=None):
        self.root=Path(run_dir);self.service=Circuit(service_root);self.credentials=str(credential_file);credentials(self.credentials)
        self.run_id=run_id or self.root.name;self.counter=0;self.records=[]
    def __call__(self,prompt,metadata):
        number=self.counter;self.counter+=1;request_id=f'{number:06d}';folder=self.root/'requests'/request_id;folder.mkdir(parents=True,exist_ok=True)
        body=canonical(payload(prompt)).encode();frozen=dict(prompt=prompt,metadata=metadata,request_index=number,run_id=self.run_id)
        identity=dict(payload_sha256=sha(body),frozen_sha256=sha(canonical(frozen)));logical=self.run_id+'/'+request_id
        write(folder/'FROZEN_INPUT.json',frozen)
        p=folder/'payload.json'
        if p.exists():
            if p.read_bytes()!=body:raise RuntimeError('REQUEST_PAYLOAD_IDENTITY_CONFLICT')
        else:write_bytes(p,body)
        if (folder/'intent.json').exists():
            intent=read(folder/'intent.json')
            if intent['identity']!=identity:raise RuntimeError('REQUEST_FROZEN_IDENTITY_CONFLICT')
        while True:
            action,until=self.service.before(logical,folder,identity)
            if action!='WAIT':break
            time.sleep(min(1,max(.05,(until or time.time()+1)-time.time())))
        if not (folder/'terminal.json').exists():
            if action=='SEND':
                write(folder/'intent.json',dict(identity=identity,submitted_unix=time.time(),logical_id=logical,submission=1,transport_attempt=1,timeout_seconds=150,retries=0,metadata=metadata))
                t=time.monotonic()
                try:
                    done=subprocess.run([sys.executable,'-m','fuse_addon.transport','--submit',str(folder),'--credentials',self.credentials],capture_output=True,timeout=150)
                    process=dict(returncode=done.returncode,stderr=done.stderr.decode(errors='replace'),wall_seconds=time.monotonic()-t)
                except subprocess.TimeoutExpired:process=dict(returncode=None,status='REQUEST_PROCESS_DEADLINE',wall_seconds=time.monotonic()-t)
                except OSError as exc:
                    process=dict(returncode=None,status='SENDER_PROCESS_START_FAILED',exception_type=type(exc).__name__,message=str(exc),wall_seconds=time.monotonic()-t)
                    write(folder/'SENDER_HOST_FAILURE.json',dict(**process,definitely_pre_http=True,payload_sha256=identity['payload_sha256']))
                write(folder/'process.json',process)
            else:settle_existing(folder)
            if (folder/'SENDER_HOST_FAILURE.json').exists():raise HostFailure('PROVEN_PRE_HTTP_HOST_FAILURE: '+str(folder))
            result=classify(folder);intent=read(folder/'intent.json');result.update(identity=identity,logical_id=logical,request_path=str(folder),submitted_unix=intent['submitted_unix'],metadata=metadata)
            write(folder/'terminal.json',result)
        result=read(folder/'terminal.json')
        if result.get('identity')!=identity or result.get('logical_id')!=logical:raise RuntimeError('TERMINAL_REQUEST_IDENTITY_CONFLICT')
        self.service.observe(logical,result,folder);self.records.append(result)
        if result.get('fatal_reason'):raise ServiceStopped(result['fatal_reason'])
        return result['content'] if result['service_success'] else None

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',required=True);p.add_argument('--credentials',required=True);a=p.parse_args();sender(a.submit,a.credentials)
