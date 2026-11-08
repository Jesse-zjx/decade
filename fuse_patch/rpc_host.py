"""V3.1 trusted outside-container single-instance route controller.

Never imports/executes a candidate. Only this host constructs routes, validates
node frames, clocks deadlines and writes trusted result/cost records. The
candidate interpreter and its wrapper are both untrusted. Public tests exercise
this exact boundary remotely; local tests are pure protocol/config fixtures.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import selectors
import shutil
import signal
import subprocess
import time
import uuid

from fuse_addon import evaluation as ev

REMOTE_ROOT = Path('/opt/decade/patch')
CPUS = ('20', '22', '24', '26')
FRAME_BYTES = 8192
DIAGNOSTIC_BYTES = 8192
LOAD_SECONDS = 10.
HIDDEN_ROUTE_SECONDS = 300.
SEARCH_PANEL_SECONDS = 500.
HIDDEN_ENVELOPE_SECONDS = 310.
TASKS = ('TSP-Construct', 'CVRP-Construct')
WRAPPER = Path(__file__).with_name('rpc_worker.py')


class ProtocolFailure(Exception):
    pass


class CandidateDeadline(Exception):
    def __init__(self, phase):
        self.phase = phase
        super().__init__(phase)


class InfrastructureFailure(Exception):
    pass


def _reject_constant(value):
    raise ProtocolFailure('NONFINITE_JSON_CONSTANT:'+value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolFailure('DUPLICATE_JSON_KEY:'+key)
        result[key] = value
    return result


def parse_frame(data, *, ready=False, seq=None, allowed=None):
    """One bounded ordinary-JSON line, exact key set, integer node/sequence."""
    if not isinstance(data, bytes) or len(data)>FRAME_BYTES or not data.endswith(b'\n') or data.count(b'\n')!=1:
        raise ProtocolFailure('FRAME_LENGTH_OR_LINE_COUNT')
    try:
        value = json.loads(data[:-1].decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, ProtocolFailure): raise
        raise ProtocolFailure('INVALID_JSON_FRAME') from exc
    if ready:
        if not isinstance(value, dict) or set(value)!={'ready'} or value['ready'] is not True:
            raise ProtocolFailure('EXPECTED_ONE_READY_FRAME')
        return value
    if not isinstance(value, dict) or set(value)!={'seq','node'}:
        raise ProtocolFailure('EXACT_NODE_RESPONSE_FIELDS_REQUIRED')
    if type(value['seq']) is not int or value['seq']!=seq:
        raise ProtocolFailure('WRONG_OR_NONINTEGER_SEQUENCE')
    if type(value['node']) is not int:
        raise ProtocolFailure('NODE_MUST_BE_JSON_INTEGER_NOT_BOOL')
    if value['node'] not in allowed:
        raise ProtocolFailure('NODE_OUTSIDE_ORIGINAL_ALLOWED_SET')
    return value['node']


def normalize_limits(limits, public_acceptance=False):
    value=dict(limits or dict(load_seconds=10, route_seconds=300, panel_seconds=None))
    load=value.get('load_seconds');route=value.get('route_seconds');panel=value.get('panel_seconds')
    numeric=lambda x:type(x) in (int,float) and math.isfinite(x) and x>0
    if not numeric(load) or (route is not None and not numeric(route)) or (panel is not None and not numeric(panel)):
        raise ValueError('INVALID_RPC_LIMITS')
    if public_acceptance:
        if load>LOAD_SECONDS or (route is not None and route>HIDDEN_ROUTE_SECONDS) or (panel is not None and panel>SEARCH_PANEL_SECONDS):
            raise ValueError('PUBLIC_LIMITS_CANNOT_EXCEED_PRODUCTION')
    elif load!=LOAD_SECONDS or (route,panel) not in ((None,SEARCH_PANEL_SECONDS),(HIDDEN_ROUTE_SECONDS,None)):
        raise ValueError('UNFROZEN_PRODUCTION_RPC_LIMITS')
    return dict(load_seconds=float(load),route_seconds=None if route is None else float(route),
                panel_seconds=None if panel is None else float(panel))


def deadlines(t0, t_route, limits, panel_deadline=None):
    panel=math.inf if panel_deadline is None else panel_deadline
    init=min(t0+limits['load_seconds'],panel)
    if limits['route_seconds'] is None:
        route=panel
    else:
        route=min(panel,t0+limits['load_seconds']+limits['route_seconds'],
                  math.inf if t_route is None else t_route+limits['route_seconds'])
    return init,route


def _wire(value):
    return json.dumps(value,allow_nan=False,ensure_ascii=False,separators=(',',':')).encode('utf-8')+b'\n'


def _host_cpu():
    kind=getattr(resource,'RUSAGE_THREAD',resource.RUSAGE_SELF)
    result=resource.getrusage(kind)
    return result.ru_utime,result.ru_stime


class _Process:
    """Pipes with bounded capture and per-PID wait4 CPU (safe across host threads)."""
    def __init__(self, command, *, interactive=False):
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,bufsize=0,close_fds=True)
        self.selector=selectors.DefaultSelector()
        for name,pipe in (('stdout',self.proc.stdout),('stderr',self.proc.stderr)):
            os.set_blocking(pipe.fileno(),False);self.selector.register(pipe,selectors.EVENT_READ,name)
        if interactive:os.set_blocking(self.proc.stdin.fileno(),False)
        self.returncode=None;self.usage=None

    def reap(self):
        if self.returncode is not None:return
        try:pid,status,usage=os.wait4(self.proc.pid,os.WNOHANG)
        except ChildProcessError as exc:raise InfrastructureFailure('DOCKER_CLI_REAP_LOST') from exc
        if pid:
            self.returncode=os.waitstatus_to_exitcode(status);self.proc.returncode=self.returncode;self.usage=usage

    def read(self, timeout):
        rows=[]
        for key,_ in self.selector.select(max(0.,timeout)):
            try:block=os.read(key.fileobj.fileno(),65536)
            except BlockingIOError:continue
            if block:rows.append((key.data,block))
            else:self.selector.unregister(key.fileobj)
        self.reap()
        return rows

    def terminate_cli(self):
        self.reap()
        if self.returncode is None:
            try:os.killpg(self.proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:pid,status,usage=os.wait4(self.proc.pid,0)
            except ChildProcessError:pass
            else:self.returncode=os.waitstatus_to_exitcode(status);self.proc.returncode=self.returncode;self.usage=usage

    def close(self):
        self.terminate_cli();self.selector.close()
        for stream in (self.proc.stdin,self.proc.stdout,self.proc.stderr):
            if stream is not None:
                try:stream.close()
                except OSError:pass


class _Commands:
    def __init__(self,folder):
        self.folder=Path(folder);self.index=0;self.cpu_user=0.;self.cpu_system=0.;self.cpu_missing=0
    def account(self,process):
        if process.usage is None:self.cpu_missing+=1
        else:self.cpu_user+=process.usage.ru_utime;self.cpu_system+=process.usage.ru_stime
    def run(self,args,*,timeout=10):
        self.index+=1;began=time.monotonic();output=bytearray();errors=bytearray();overflow=False
        process=_Process(args)
        timed_out=False
        try:
            while process.returncode is None or process.selector.get_map():
                remaining=began+max(.05,timeout)-time.monotonic()
                if remaining<=0:
                    timed_out=True;process.terminate_cli();break
                for stream,block in process.read(min(.05,remaining)):
                    target=output if stream=='stdout' else errors
                    if len(target)+len(block)>1024*1024:overflow=True
                    target.extend(block[:max(0,1024*1024-len(target))])
            result=dict(command=list(args),returncode=process.returncode,stdout=output.decode('utf-8',errors='replace'),
                stderr=errors.decode('utf-8',errors='replace'),timed_out=timed_out,output_truncated=overflow,
                wall_seconds=time.monotonic()-began)
        finally:
            process.close();self.account(process)
        ev.freeze(self.folder/'docker_commands'/f'{self.index:03d}.json',result)
        if overflow:raise InfrastructureFailure('DOCKER_CONTROL_OUTPUT_OVERFLOW')
        return result


def create_command(source,wrapper,cpu,name,nonce):
    """Exactly three readonly binds; no evaluator, panel, socket or host project."""
    if str(cpu) not in CPUS:raise ValueError('CPU_OUTSIDE_V3_RESERVED_SLOTS')
    command=['docker','create','--interactive','--name',name,'--hostname',nonce[:12],
        '--network','none','--ipc','private','--cgroupns','private','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--user','65534:65534','--pids-limit','64',
        '--memory','8g','--memory-swap','8g','--cpuset-cpus',str(cpu),
        '--tmpfs','/tmp:rw,nosuid,nodev,size=128m,mode=1777',
        '--tmpfs','/workspace:ro,nosuid,nodev,noexec,size=4k,mode=0555','--workdir','/tmp',
        '--label','fuse-rpc-owner='+nonce]
    for host,target in ((Path(ev.RUNTIME).resolve(),ev.RUNTIME),(Path(source).resolve(),'/candidate/source.py'),
                        (Path(wrapper).resolve(),'/rpc/worker.py')):
        command += ['--mount',f'type=bind,src={host},dst={target},readonly']
    command += ['--entrypoint','/usr/bin/env',ev.IMAGE,'-i','PATH=/usr/local/bin:/usr/bin:/bin','HOME=/tmp',
        'PYTHONDONTWRITEBYTECODE=1','PYTHONHASHSEED=0','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
        'MKL_NUM_THREADS=1','NUMBA_NUM_THREADS=1','NUMBA_CACHE_DIR=/tmp/numba-cache',
        ev.RUNTIME+'/bin/python','/rpc/worker.py']
    return command


def gpfs_mount_mapping(mountinfo):
    """One narrow Docker-proxy path rewrite witnessed by host mountinfo.

    The authorized host exposes the GPFS filesystem as /mnt/finder, while
    its Docker proxy names that same filesystem /mnt/<source without fs_>.
    No arbitrary alias, different filesystem, submount or suffix is accepted.
    """
    witness=dict(schema='V3_1_GPFS_DOCKER_SOURCE_MAPPING',available=False,
        mountinfo_sha256=ev.digest(mountinfo.encode('utf-8')),matched_line=None,
        requested_prefix='/mnt/finder',daemon_prefix=None,mountpoints=[])
    matches=[]
    for line in mountinfo.splitlines():
        fields=line.split()
        try:separator=fields.index('-')
        except ValueError:continue
        if separator<6 or len(fields)<separator+4:continue
        decode=lambda value:re.sub(r'\\([0-7]{3})',lambda m:chr(int(m.group(1),8)),value)
        root,mountpoint=decode(fields[3]),decode(fields[4])
        if mountpoint.startswith('/'):witness['mountpoints'].append(mountpoint)
        fstype,source=fields[separator+1:separator+3]
        found=re.fullmatch(r'fs_(vepfs-[a-z0-9]+)',source)
        if root=='/' and mountpoint=='/mnt/finder' and fstype=='gpfs' and found:
            matches.append((line,'/mnt/'+found.group(1)))
    if len(matches)==1:
        witness.update(available=True,matched_line=matches[0][0],daemon_prefix=matches[0][1])
    else:witness['reason']='EXPECTED_EXACTLY_ONE_TRUSTED_GPFS_MOUNT'
    return witness


def mount_source_receipt(requested,actual,mapping):
    receipt=dict(requested_source=requested,actual_source=actual,match='REJECTED')
    if actual==requested:return dict(receipt,match='EXACT')
    if not mapping.get('available'):return receipt
    prefix=mapping['requested_prefix']
    if not (requested==prefix or requested.startswith(prefix+'/')):return receipt
    # A more specific mount cannot inherit the parent filesystem's mapping.
    covering=[point for point in mapping['mountpoints'] if requested==point or requested.startswith(point.rstrip('/')+'/')]
    if not covering or max(covering,key=len)!=prefix:return receipt
    mapped=mapping['daemon_prefix']+requested[len(prefix):]
    receipt['mapped_source']=mapped
    if actual==mapped:receipt['match']='EXACT_GPFS_PROXY_REWRITE'
    return receipt


def inspect_contract(record,*,source,wrapper,cpu,name,nonce):
    """Validate actual Docker config before starting untrusted code."""
    h=record.get('HostConfig',{});c=record.get('Config',{});errors=[]
    expected=dict(NetworkMode='none',IpcMode='private',CgroupnsMode='private',ReadonlyRootfs=True,
        Privileged=False,PidsLimit=64,Memory=8*1024**3,MemorySwap=8*1024**3,CpusetCpus=str(cpu))
    for key,value in expected.items():
        if h.get(key)!=value:errors.append('HOSTCONFIG_'+key)
    if h.get('PidMode') not in ('',None):errors.append('HOST_PID_NAMESPACE')
    if h.get('UTSMode') not in ('',None):errors.append('HOST_UTS_NAMESPACE')
    if set(h.get('CapDrop') or [])!={'ALL'} or h.get('CapAdd'):errors.append('CAPABILITIES')
    opts=h.get('SecurityOpt') or []
    if not any(x in ('no-new-privileges','no-new-privileges=true') for x in opts):errors.append('NO_NEW_PRIVILEGES')
    if any('unconfined' in x for x in opts):errors.append('UNCONFINED_SECURITY')
    if h.get('Devices') or h.get('DeviceRequests'):errors.append('EXTRA_DEVICES')
    if c.get('User')!='65534:65534' or c.get('Hostname')!=nonce[:12]:errors.append('USER_HOSTNAME')
    if c.get('Labels',{}).get('fuse-rpc-owner')!=nonce or record.get('Name')!='/'+name:errors.append('OWNER_IDENTITY')
    if c.get('OpenStdin') is not True or c.get('Tty') is True:errors.append('STDIN_TTY')
    if c.get('WorkingDir')!='/tmp':errors.append('WORKDIR')
    if c.get('Entrypoint')!=['/usr/bin/env'] or not isinstance(c.get('Cmd'),list) or not c['Cmd'] or c['Cmd'][0]!='-i':errors.append('CLEAN_ENV_ENTRYPOINT')
    requested=create_command(source,wrapper,cpu,name,nonce)
    if c.get('Cmd')!=requested[requested.index(ev.IMAGE)+1:]:errors.append('EXACT_WRAPPER_COMMAND')
    binds={ev.RUNTIME:str(Path(ev.RUNTIME).resolve()),'/candidate/source.py':str(Path(source).resolve()),'/rpc/worker.py':str(Path(wrapper).resolve())}
    mount_rows=record.get('Mounts',[])
    # Docker versions differ in whether --tmpfs is repeated in Mounts. It is
    # always checked through HostConfig.Tmpfs, and no other mount is permitted.
    actual={m.get('Destination'):m for m in mount_rows if m.get('Type')!='tmpfs'}
    tmpfs_rows=[m for m in mount_rows if m.get('Type')=='tmpfs']
    tmpfs_modes={'/tmp':True,'/workspace':False}
    if (set(actual)!=set(binds) or len(actual)!=len(mount_rows)-len(tmpfs_rows) or
            len({m.get('Destination') for m in tmpfs_rows})!=len(tmpfs_rows) or
            any(m.get('Destination') not in tmpfs_modes or m.get('RW') is not tmpfs_modes.get(m.get('Destination'))
                for m in tmpfs_rows)):
        errors.append('EXTRA_OR_MISSING_MOUNTS')
    mapping=dict(available=False,reason='ALL_SOURCES_MATCH_REQUESTED_PATHS')
    if any(actual.get(target,{}).get('Source')!=path for target,path in binds.items()):
        try:mapping=gpfs_mount_mapping(Path('/proc/self/mountinfo').read_text(encoding='utf-8'))
        except (OSError,UnicodeError) as exc:mapping=dict(available=False,reason='HOST_MOUNTINFO_UNAVAILABLE:'+type(exc).__name__)
    source_receipts={}
    for target,path in binds.items():
        m=actual.get(target,{})
        source_receipts[target]=mount_source_receipt(path,m.get('Source'),mapping)
        if m.get('Type')!='bind' or m.get('RW') is not False or source_receipts[target]['match']=='REJECTED':errors.append('MOUNT_'+target)
    tmp=h.get('Tmpfs') or {}
    tmp_options=set(tmp.get('/tmp','').split(','))
    if (set(tmp)!={'/tmp','/workspace'} or not ({'size=128m','size=134217728'} & tmp_options) or
            not {'nosuid','nodev','rw','mode=1777'}<=tmp_options):
        errors.append('PRIVATE_TMPFS')
    workspace_options=set(tmp.get('/workspace','').split(','))
    normalized_workspace={{'size=4096':'size=4k','mode=555':'mode=0555'}.get(value,value)
                          for value in workspace_options}
    if normalized_workspace!={'ro','nosuid','nodev','noexec','size=4k','mode=0555'}:
        errors.append('EMPTY_READONLY_WORKSPACE_MASK')
    if record.get('Image')!=ev.IMAGE:errors.append('IMAGE_IDENTITY')
    return dict(passed=not errors,errors=errors,actual_image=record.get('Image'),
        apparmor_profile=record.get('AppArmorProfile'),security_options=opts,
        readonly_bind_count=len(actual),opaque_owner=nonce,absolute_security_proof=False,
        mount_source_mapping=mapping,mount_source_receipts=source_receipts,
        workspace_mask=dict(destination='/workspace',kind='empty readonly tmpfs overlay',
            size_bytes=4096,actual_options=tmp.get('/workspace'),preexisting_image_contents_read=False))


def _inspect(commands,name,*,timeout=5):
    result=commands.run(['docker','inspect',name],timeout=timeout)
    if result['timed_out']:raise InfrastructureFailure('DOCKER_INSPECT_TIMEOUT')
    if result['returncode']!=0:
        text=result['stderr'].lower()
        if 'no such object' in text or 'no such container' in text:return None
        raise InfrastructureFailure('DOCKER_INSPECT_FAILED:'+result['stderr'][:500])
    try:rows=json.loads(result['stdout'])
    except ValueError as exc:raise InfrastructureFailure('INVALID_DOCKER_INSPECT_JSON') from exc
    if not isinstance(rows,list) or len(rows)!=1:raise InfrastructureFailure('DOCKER_INSPECT_COUNT')
    return rows[0]


def _cgroup_cpu(inspect):
    """Read actual container cgroup counters, never estimate from CPU percentage."""
    pid=(inspect or {}).get('State',{}).get('Pid')
    if type(pid) is not int or pid<=0:return dict(available=False,reason='NO_LIVE_CONTAINER_PID')
    try:
        groups=Path(f'/proc/{pid}/cgroup').read_text().splitlines()
        for line in groups:
            hierarchy,controllers,relative=line.split(':',2)
            if hierarchy=='0' and controllers=='':
                path=(Path('/sys/fs/cgroup')/relative.lstrip('/')).resolve()
                if not path.is_relative_to('/sys/fs/cgroup'):raise ValueError('CGROUP_PATH_ESCAPE')
                fields=dict(row.split() for row in (path/'cpu.stat').read_text().splitlines())
                return dict(available=True,kind='cgroup-v2',path=str(path/'cpu.stat'),
                    total_seconds=int(fields['usage_usec'])/1e6,
                    user_seconds=int(fields['user_usec'])/1e6 if 'user_usec' in fields else None,
                    system_seconds=int(fields['system_usec'])/1e6 if 'system_usec' in fields else None,
                    observed_monotonic=time.monotonic(),raw=fields)
            if 'cpuacct' in controllers.split(','):
                path=(Path('/sys/fs/cgroup/cpuacct')/relative.lstrip('/')).resolve()
                if not path.is_relative_to('/sys/fs/cgroup'):raise ValueError('CGROUP_PATH_ESCAPE')
                ticks=os.sysconf('SC_CLK_TCK');fields=dict(row.split() for row in (path/'cpuacct.stat').read_text().splitlines())
                return dict(available=True,kind='cgroup-v1',path=str(path/'cpuacct.usage'),
                    total_seconds=int((path/'cpuacct.usage').read_text())/1e9,
                    user_seconds=int(fields['user'])/ticks,system_seconds=int(fields['system'])/ticks,
                    observed_monotonic=time.monotonic(),raw=fields)
    except (OSError,ValueError,KeyError) as exc:
        return dict(available=False,reason=type(exc).__name__+':'+str(exc))
    return dict(available=False,reason='CPU_CGROUP_NOT_FOUND')


def _read_prior_cgroup(sample):
    if not sample.get('available'):return dict(available=False,reason='NO_PRIOR_CGROUP_PATH')
    try:
        path=Path(sample['path'])
        if sample['kind']=='cgroup-v2':
            fields=dict(row.split() for row in path.read_text().splitlines())
            return dict(available=True,kind='cgroup-v2',path=str(path),total_seconds=int(fields['usage_usec'])/1e6,
                user_seconds=int(fields['user_usec'])/1e6 if 'user_usec' in fields else None,
                system_seconds=int(fields['system_usec'])/1e6 if 'system_usec' in fields else None,
                observed_monotonic=time.monotonic(),raw=fields)
        fields=dict(row.split() for row in path.with_name('cpuacct.stat').read_text().splitlines())
        ticks=os.sysconf('SC_CLK_TCK')
        return dict(available=True,kind='cgroup-v1',path=str(path),total_seconds=int(path.read_text())/1e9,
            user_seconds=int(fields['user'])/ticks,system_seconds=int(fields['system'])/ticks,
            observed_monotonic=time.monotonic(),raw=fields)
    except (OSError,ValueError,KeyError) as exc:return dict(available=False,reason=type(exc).__name__+':'+str(exc))


def _cleanup(commands,intent,*,kill_first=False):
    """Only this nonce-owned container; successful absence required before return."""
    began=time.monotonic();name=intent['container_name'];samples=[];errors=[];last=None;complete_cpu=False
    try:
        # This path is only for a container whose actual nonce ownership was
        # checked by this invocation before start. Do not delay a deadline kill
        # behind an observational inspect; lost-intent recovery verifies first.
        if kill_first:
            commands.run(['docker','kill','--signal','KILL',name],timeout=5)
        last=_inspect(commands,name)
        if last is not None:
            if last.get('Config',{}).get('Labels',{}).get('fuse-rpc-owner')!=intent['nonce']:
                raise InfrastructureFailure('CLEANUP_CONTAINER_OWNER_CONFLICT')
            ev.freeze(commands.folder/'INSPECT_BEFORE_CLEANUP.json',last)
            sample=_cgroup_cpu(last);samples.append(sample)
            if last.get('State',{}).get('Running'):
                commands.run(['docker','kill','--signal','KILL',name],timeout=5)
            stopped=_inspect(commands,name)
            if stopped is not None and stopped.get('State',{}).get('Running'):
                raise InfrastructureFailure('CONTAINER_STILL_RUNNING_AFTER_KILL')
            final_sample=_read_prior_cgroup(sample);samples.append(final_sample)
            complete_cpu=bool(stopped is not None and not stopped.get('State',{}).get('Running') and final_sample.get('available'))
            if stopped is not None:
                last=stopped;ev.freeze(commands.folder/'INSPECT_FINAL.json',stopped)
            commands.run(['docker','rm','--force',name],timeout=5)
        absent=_inspect(commands,name)
        if absent is not None:raise InfrastructureFailure('CONTAINER_NOT_REMOVED')
        confirmed=True
    except (InfrastructureFailure,OSError,ValueError,TypeError,KeyError) as exc:
        errors.append(type(exc).__name__+':'+str(exc));confirmed=False
    never_started=bool(last and last.get('State',{}).get('Status')=='created' and
        last.get('State',{}).get('StartedAt') in ('0001-01-01T00:00:00Z',None,''))
    if never_started:complete_cpu=True
    valid=[s for s in samples if s.get('available')]
    measured=max(valid,key=lambda s:s['total_seconds']) if valid else None
    return dict(confirmed_removed=confirmed,errors=errors,wall_seconds=time.monotonic()-began,
        final_inspect=last,deadline_kill_dispatched_before_inspect=kill_first,
        container_cpu_samples=samples,container_cpu_complete=complete_cpu,
        container_cpu_seconds=measured['total_seconds'] if complete_cpu and measured else (0. if never_started else None),
        container_cpu_seconds_known=measured['total_seconds'] if measured else 0.,
        container_cpu_seconds_missing=0 if complete_cpu else 1,
        child_cpu_user_seconds=measured.get('user_seconds') if complete_cpu and measured else (0. if never_started else None),
        child_cpu_system_seconds=measured.get('system_seconds') if complete_cpu and measured else (0. if never_started else None))


class _RPC:
    def __init__(self,process,folder,t0,limits,panel_deadline):
        self.process=process;self.folder=Path(folder);self.t0=t0;self.limits=limits;self.panel_deadline=panel_deadline
        self.protocol=bytearray();self.diagnostics=bytearray();self.diagnostic_total=0
        self.ready_time=None;self.route_time=None;self.seq=0;self.ipc_seconds=0.;self.request_bytes=0;self.response_bytes=0
        self.protocol_failure_prefix=b''
        self.trace=(self.folder/'RPC_TRACE.jsonl').open('xb')

    def _ingest(self,rows):
        for stream,block in rows:
            if stream=='stderr':
                self.diagnostic_total+=len(block);self.diagnostics.extend(block[:max(0,DIAGNOSTIC_BYTES-len(self.diagnostics))])
            else:
                self.response_bytes+=len(block);self.protocol.extend(block)
                self.protocol_failure_prefix=bytes(self.protocol[:FRAME_BYTES])
                if len(self.protocol)>FRAME_BYTES:raise ProtocolFailure('PROTOCOL_FRAME_OVER_8192_BYTES')

    def quiet(self):
        self._ingest(self.process.read(0))
        if self.protocol:raise ProtocolFailure('UNSOLICITED_OR_MULTIPLE_PROTOCOL_FRAMES')

    def send(self,value,deadline,phase):
        block=_wire(value);offset=0;start=time.monotonic()
        try:
            self.quiet()
            while offset<len(block):
                if time.monotonic()>=deadline:raise CandidateDeadline(phase)
                try:count=os.write(self.process.proc.stdin.fileno(),block[offset:])
                except BlockingIOError:
                    self._ingest(self.process.read(min(.01,max(0,deadline-time.monotonic()))));continue
                except BrokenPipeError as exc:raise ProtocolFailure('CANDIDATE_CLOSED_INPUT') from exc
                offset+=count
        finally:
            self.request_bytes+=offset;self.ipc_seconds+=time.monotonic()-start

    def receive(self,deadline,phase,*,ready=False,seq=None,allowed=None):
        start=time.monotonic()
        try:
            while True:
                # Buffered bytes and selector reads do not extend a deadline.
                now=time.monotonic()
                if now>=deadline:raise CandidateDeadline(phase)
                if b'\n' in self.protocol:
                    split=self.protocol.index(10)+1
                    if split!=len(self.protocol):raise ProtocolFailure('MULTIPLE_OR_TRAILING_PROTOCOL_OUTPUT')
                    frame=bytes(self.protocol);self.protocol.clear()
                    result=parse_frame(frame,ready=ready,seq=seq,allowed=allowed)
                    self.quiet()
                    if time.monotonic()>=deadline:raise CandidateDeadline(phase)
                    self.protocol_failure_prefix=b''
                    return result
                rows=self.process.read(min(.02,deadline-now));self._ingest(rows)
                if self.process.returncode is not None and not self.process.selector.get_map() and b'\n' not in self.protocol:
                    raise ProtocolFailure('CANDIDATE_EXITED_WITH_TRUNCATED_RESPONSE' if self.protocol else 'CANDIDATE_EXITED_WITHOUT_RESPONSE')
        finally:self.ipc_seconds+=time.monotonic()-start

    def initialize(self,item,task):
        message=dict(task=task,execution_seed=item['execution_seed'],distance_matrix=item['distance_matrix'])
        if task=='CVRP-Construct':
            message['demands']=item['demands']
            # Preserve cvrp_route's exact original callback matrix. The panel's
            # NumPy norm matrix can differ by a final floating-point bit.
            coords=item['coordinates']
            message['distance_matrix']=[[math.dist(a,b) for b in coords] for a in coords]
        ev.freeze(self.folder/'INIT_REQUEST.json',message)
        deadline,_=deadlines(self.t0,None,self.limits,self.panel_deadline)
        self.send(message,deadline,'INIT_TIMEOUT');self.receive(deadline,'INIT_TIMEOUT',ready=True)
        self.ready_time=time.monotonic()
        if self.ready_time>=deadline:raise CandidateDeadline('INIT_TIMEOUT')

    def node(self,task,current,destination,unvisited,remaining=None,demands=None):
        if self.route_time is None:self.route_time=time.monotonic()
        _,deadline=deadlines(self.t0,self.route_time,self.limits,self.panel_deadline)
        allowed=[int(x) for x in unvisited]
        if task=='CVRP-Construct':
            allowed=[node for node in allowed if int(demands[node])<=remaining]
            if current!=destination:allowed.append(int(destination))
        request=dict(seq=self.seq,current=int(current),destination=int(destination),unvisited=[int(x) for x in unvisited])
        if task=='CVRP-Construct':request['remaining_capacity']=int(remaining)
        self.send(request,deadline,'ROUTE_TIMEOUT')
        node=self.receive(deadline,'ROUTE_TIMEOUT',seq=self.seq,allowed=set(allowed))
        self.trace.write(_wire(dict(request=request,response=dict(seq=self.seq,node=node),original_allowed=allowed)))
        self.trace.flush();self.seq+=1
        return node

    def close(self):
        self.trace.close();ev.freeze_bytes(self.folder/'DIAGNOSTIC.bin',bytes(self.diagnostics))
        if self.protocol_failure_prefix:
            ev.freeze_bytes(self.folder/'PROTOCOL_FAILURE_PREFIX.bin',self.protocol_failure_prefix)


def _guard(source,output,public):
    if platform.system()!='Linux' or not source.is_relative_to(REMOTE_ROOT) or not output.is_relative_to(REMOTE_ROOT):
        raise RuntimeError('RPC_CANDIDATE_EXECUTION_ONLY_AUTHORIZED_REMOTE_NEW_CHECKOUT')
    if not Path(ev.RUNTIME+'/bin/python').is_file():raise RuntimeError('FROZEN_RUNTIME_MISSING')
    if public:
        study=REMOTE_ROOT/'runs'/'fuse-forward-continuation-diagnostic-20260928'
        permitted=(study/'engineering'/'public-acceptance').resolve()
        authority=ev.read(study/'authorization'/'V3_1_USER_AUTHORIZATION.json')
        if (not output.is_relative_to(permitted) or authority.get('user_authorized') is not True or
                authority.get('public_acceptance_then_direct_scientific_dispatch') is not True):
            raise RuntimeError('PUBLIC_ACCEPTANCE_OUTSIDE_EXPLICIT_V3_1_AUTHORIZATION')
        if (study/'RESULT_SEALED.json').exists():raise RuntimeError('V3_1_STUDY_ALREADY_SEALED')
    else:
        from .runtime import _require_remote
        _require_remote(output)


def _identity(source_path,item,task,cpu,limits,public):
    return dict(source_sha256=ev.digest(source_path.read_bytes()),instance_sha256=ev.digest(ev.encoded(item)),
        instance_id=item['instance_id'],task=task,cpu=str(cpu),limits=limits,public_acceptance=bool(public),
        wrapper_sha256=ev.digest(WRAPPER.read_bytes()),host_sha256=ev.digest(Path(__file__).read_bytes()),
        image=ev.IMAGE,runtime=ev.RUNTIME)


def _row(intent,status,*,error=None):
    return dict(instance_id=intent['identity']['instance_id'],type='ROUTE_RESULT',status=status,length=None,
        source_sha256=intent['identity']['source_sha256'],hard_stop=status=='UNKNOWN',
        infrastructure_error=error if status=='UNKNOWN' else None,error_type=error,
        load_seconds=None,route_seconds=None,container_started=False,
        initialization_contract='host-observed protocol READY deadline, not proof of Python source load duration',
        ordinary_candidate_clocks_allowed=True)


def _finish(folder,intent,row,commands,process,rpc,started_cpu,t0,*,recovery=False):
    kill_first=not recovery and process is not None and row['status'] in ('INIT_TIMEOUT','ROUTE_TIMEOUT')
    cleanup=_cleanup(commands,intent,kill_first=kill_first)
    def tail_failure(error):
        row['protocol_tail_error']=error
        if row['status']=='EVALUATED':
            row.update(status='PROTOCOL_FAILURE',length=None,error_type=error,hard_stop=False,infrastructure_error=None)
    if process is not None:
        try:
            # Always drain diagnostics after whole-container termination. The
            # protocol tail is still untrusted and invalidates extra frames.
            stop=time.monotonic()+2
            while process.selector.get_map() and time.monotonic()<stop:
                parts=process.read(.02)
                if rpc is not None:
                    try:rpc._ingest(parts)
                    except ProtocolFailure as exc:
                        tail_failure(str(exc))
            if rpc is not None and rpc.protocol:
                tail_failure('TRAILING_PROTOCOL_OUTPUT')
        except (InfrastructureFailure,OSError) as exc:
            row.update(status='UNKNOWN',length=None,hard_stop=True,infrastructure_error=str(exc))
        finally:
            try:process.close()
            except (InfrastructureFailure,OSError) as exc:
                row.update(status='UNKNOWN',length=None,hard_stop=True,infrastructure_error=str(exc))
            commands.account(process)
    if rpc is not None:rpc.close()
    if not cleanup['confirmed_removed']:
        row.update(status='UNKNOWN',hard_stop=True,infrastructure_error='CONTAINER_CLEANUP_UNCONFIRMED',length=None)
    inspected=cleanup.get('final_inspect') or {}
    state=inspected.get('State',{})
    started_at=state.get('StartedAt')
    if state.get('Running') or (type(state.get('Pid')) is int and state['Pid']>0) or (started_at and not started_at.startswith('0001-')):
        row['container_started']=True
    elif state.get('Status')=='created' and row['container_started'] is not True:
        row['container_started']=False
    if inspected.get('State',{}).get('OOMKilled'):
        row.update(candidate_container_oom=True)
    if cleanup['confirmed_removed']:
        stage=Path(intent['staging_directory']).resolve()
        base=(REMOTE_ROOT/'.rpc-material').resolve()
        if stage.is_relative_to(base) and stage.name==intent['nonce'] and stage.exists():
            try:shutil.rmtree(stage)
            except OSError as exc:row['staging_cleanup_error']=str(exc)
    cpu_end=_host_cpu();row.update(cleanup=cleanup,
        child_cpu_user_seconds=cleanup['child_cpu_user_seconds'],child_cpu_system_seconds=cleanup['child_cpu_system_seconds'],
        container_cpu_seconds=cleanup['container_cpu_seconds'],container_cpu_seconds_known=cleanup['container_cpu_seconds_known'],
        container_cpu_seconds_missing=cleanup['container_cpu_seconds_missing'],
        host_cpu_user_seconds=max(0.,cpu_end[0]-started_cpu[0])+commands.cpu_user,
        host_cpu_system_seconds=max(0.,cpu_end[1]-started_cpu[1])+commands.cpu_system,
        host_thread_cpu_user_seconds=max(0.,cpu_end[0]-started_cpu[0]),host_thread_cpu_system_seconds=max(0.,cpu_end[1]-started_cpu[1]),
        docker_cli_cpu_user_seconds=commands.cpu_user,docker_cli_cpu_system_seconds=commands.cpu_system,
        docker_cli_cpu_missing=commands.cpu_missing,wall_seconds=time.monotonic()-t0,
        host_cpu_complete=not recovery and not commands.cpu_missing,
        host_cpu_seconds_missing=(1 if recovery else 0)+commands.cpu_missing,
        active_seconds=time.monotonic()-t0,child_wall_seconds=time.monotonic()-t0,
        startup_seconds=(rpc.ready_time-t0 if rpc and rpc.ready_time is not None else None),
        ipc_seconds=rpc.ipc_seconds if rpc else None,cleanup_seconds=cleanup['wall_seconds'],
        diagnostic_bytes_observed=rpc.diagnostic_total if rpc else None,
        diagnostic_bytes_retained=len(rpc.diagnostics) if rpc else None,
        protocol_request_bytes=rpc.request_bytes if rpc else None,protocol_response_bytes=rpc.response_bytes if rpc else None,
        resumed_existing_intent=recovery,physical_work_repeated=False,terminal_unix=time.time())
    ev.freeze(folder/'RESULT.json',row)
    return row


def run_instance(source_path,item,*,task,cpu,output_dir,panel_deadline=None,limits=None,public_acceptance=False):
    """One fresh container per instance. Existing unknown intents are not rerun.

    On recovery, return an existing immutable RESULT; otherwise only inspect and
    remove the original nonce-owned container, then record UNKNOWN. The caller
    stops a recovered incomplete panel instead of dispatching its remaining
    instances. This function never resumes a lost interactive candidate stream.
    """
    source_path=Path(source_path).resolve();folder=Path(output_dir).resolve()
    if task not in TASKS or str(cpu) not in CPUS:raise ValueError('UNREGISTERED_RPC_TASK_OR_CPU')
    limits=normalize_limits(limits,public_acceptance)
    if limits['panel_seconds'] is not None and panel_deadline is None:raise ValueError('SEARCH_ABSOLUTE_PANEL_DEADLINE_REQUIRED')
    if panel_deadline is not None and (type(panel_deadline) not in (int,float) or not math.isfinite(panel_deadline)):
        raise ValueError('INVALID_PANEL_DEADLINE')
    identity=_identity(source_path,item,task,cpu,limits,public_acceptance)
    folder.mkdir(parents=True,exist_ok=True)
    intent_path=folder/'INTENT.json'
    if intent_path.exists():
        intent=ev.read(intent_path)
        if intent['identity']!=identity:raise RuntimeError('RPC_INSTANCE_FROZEN_IDENTITY_CONFLICT')
        if (folder/'RESULT.json').exists():return ev.read(folder/'RESULT.json')
        _guard(source_path,folder,public_acceptance)
        # Never create/start a container or resubmit an RPC after an unknown
        # prior intent. A separate recovery log namespace preserves old control
        # receipts and immutable stdout/diagnostics.
        recovery=folder/('recovery-'+uuid.uuid4().hex);recovery.mkdir()
        t0=time.monotonic();commands=_Commands(recovery)
        row=_row(intent,'UNKNOWN',error='LOST_INTERACTIVE_INSTANCE_OUTCOME_NOT_REPLAYED')
        row['container_started']=None
        return _finish(folder,intent,row,commands,None,None,_host_cpu(),t0,recovery=True)
    _guard(source_path,folder,public_acceptance)
    t0=time.monotonic();cpu0=_host_cpu()
    if panel_deadline is not None and t0>=panel_deadline:
        # This is host scheduling, not a started candidate; no container intent.
        return dict(instance_id=item['instance_id'],status='NOT_EXECUTED',length=None,hard_stop=False,
                    container_started=False,error_type='PANEL_DEADLINE_BEFORE_INSTANCE')
    nonce=uuid.uuid4().hex;name='rpc-'+nonce;stage=REMOTE_ROOT/'.rpc-material'/nonce
    stage.mkdir(parents=True,exist_ok=False)
    staged_source=stage/'module.py';staged_wrapper=stage/'worker.py'
    staged_source.write_bytes(source_path.read_bytes());staged_wrapper.write_bytes(WRAPPER.read_bytes())
    if (ev.digest(staged_source.read_bytes())!=identity['source_sha256'] or
            ev.digest(staged_wrapper.read_bytes())!=identity['wrapper_sha256']):
        shutil.rmtree(stage)
        raise RuntimeError('RPC_SOURCE_OR_WRAPPER_CHANGED_BEFORE_DISPATCH')
    staged_source.chmod(0o644);staged_wrapper.chmod(0o644)
    command=create_command(staged_source,staged_wrapper,cpu,name,nonce)
    intent=ev.freeze(intent_path,dict(schema='V3_1_EXTERNAL_RPC_INSTANCE',identity=identity,nonce=nonce,
        container_name=name,staging_directory=str(stage),create_command=command,submitted_unix=time.time(),
        t0_monotonic=t0,panel_deadline_monotonic=panel_deadline,original_source_path=str(source_path),
        mounts_exclude_panels_controllers_credentials=True,unknown_never_replayed=True))
    commands=_Commands(folder);process=rpc=None;row=_row(intent,'UNKNOWN',error='UNFINISHED_HOST_INSTANCE')
    try:
        init_deadline,_=deadlines(t0,None,limits,panel_deadline)
        created=commands.run(command,timeout=max(.05,init_deadline-time.monotonic()))
        if created['timed_out']:raise InfrastructureFailure('CONTAINER_CREATE_OUTCOME_UNRESOLVED_AT_INIT_DEADLINE')
        if created['returncode']!=0:raise InfrastructureFailure('CONTAINER_CREATE_FAILED:'+created['stderr'][:500])
        if time.monotonic()>=init_deadline:raise CandidateDeadline('INIT_TIMEOUT')
        actual=_inspect(commands,name,timeout=max(.05,init_deadline-time.monotonic()))
        if actual is None:raise InfrastructureFailure('CREATED_CONTAINER_NOT_FOUND')
        ev.freeze(folder/'INSPECT_CREATED.json',actual)
        checked=inspect_contract(actual,source=staged_source,wrapper=staged_wrapper,cpu=cpu,name=name,nonce=nonce)
        ev.freeze(folder/'INSPECT_CONTRACT.json',checked)
        if not checked['passed']:raise InfrastructureFailure('ACTUAL_CONTAINER_CONTRACT_CONFLICT:'+','.join(checked['errors']))
        if time.monotonic()>=init_deadline:raise CandidateDeadline('INIT_TIMEOUT')
        start_command=['docker','start','--attach','--interactive',name]
        ev.freeze(folder/'START_INTENT.json',dict(command=start_command,submitted_monotonic=time.monotonic()))
        row['container_started']=None
        process=_Process(start_command,interactive=True)
        rpc=_RPC(process,folder,t0,limits,panel_deadline)
        rpc.initialize(item,task);row['container_started']=True
        if task=='TSP-Construct':
            from .sandbox_worker import tsp_route
            facts=tsp_route(lambda current,destination,unvisited,matrix:
                rpc.node(task,current,destination,unvisited),item)
        else:
            from .cvrp_contract import cvrp_route
            facts=cvrp_route(lambda current,depot,unvisited,remaining,demands,matrix:
                rpc.node(task,current,depot,unvisited,remaining,demands),item)
        _,route_deadline=deadlines(t0,rpc.route_time,limits,panel_deadline)
        if time.monotonic()>=route_deadline:raise CandidateDeadline('ROUTE_TIMEOUT')
        rpc.quiet()
        row.update(facts,status='EVALUATED',hard_stop=False,infrastructure_error=None,error_type=None)
    except CandidateDeadline as exc:
        row.update(status=exc.phase,length=None,hard_stop=False,infrastructure_error=None,error_type=exc.phase)
    except ProtocolFailure as exc:
        row.update(status='PROTOCOL_FAILURE',length=None,hard_stop=False,infrastructure_error=None,error_type=str(exc))
    except ValueError as exc:
        if task=='CVRP-Construct' and str(exc)=='CVRP_ITERATION_LIMIT':
            row.update(status='ALGORITHM_FAILURE',length=None,hard_stop=False,infrastructure_error=None,error_type=str(exc))
        else:
            row.update(status='UNKNOWN',length=None,hard_stop=True,infrastructure_error=type(exc).__name__+':'+str(exc),error_type=type(exc).__name__)
    except (InfrastructureFailure,OSError,TypeError,KeyError) as exc:
        # Value/type/key failures in trusted task math or registered fixtures are
        # evaluator infrastructure; invalid candidate nodes were already caught
        # at the ordinary-JSON/allowed-set boundary above.
        row.update(status='UNKNOWN',length=None,hard_stop=True,infrastructure_error=type(exc).__name__+':'+str(exc),error_type=type(exc).__name__)
    finally:
        if rpc is not None:
            row['load_seconds']=rpc.ready_time-t0 if rpc.ready_time is not None else None
            row['route_seconds']=time.monotonic()-rpc.route_time if rpc.route_time is not None else None
        row=_finish(folder,intent,row,commands,process,rpc,cpu0,t0)
    return row
