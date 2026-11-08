"""Durable common evaluator for official native EoH/HiFo and FUSE endpoints.

Exact cache is scoped to one search run. A persisted intent is never replayed:
recovery collects the same named container, or records uncertainty. No source
static admissibility rule specific to FUSE is applied to ordinary baselines.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time

IMAGE = 'sha256:f0ee0d1b7f8eb46fc35919693e8f5f80462d15be0f48e92be001144726e059ae'
RUNTIME = '/opt/decade/venv'
RECORD_PREFIX = 'FUSE_NATIVE_JSON '


class EvaluationInfrastructureError(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def freeze_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError('IMMUTABLE_BYTES_CONFLICT: ' + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError('IMMUTABLE_BYTES_CONFLICT: ' + str(path))
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def freeze(path, value):
    freeze_bytes(path, encoded(value)+b'\n')
    return value


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def container_command(worker_path, source_path, panel_path, name, cpu, *, outer_seconds,
                      runtime=RUNTIME, image=IMAGE):
    if not str(cpu).isdigit():
        raise ValueError('EXACTLY_ONE_CPU_REQUIRED')
    command = ['docker', 'run', '--name', name, '--network', 'none', '--read-only',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534',
               '--pids-limit', '64', '--memory', '8g', '--memory-swap', '8g',
               '--cpuset-cpus', str(cpu), '--tmpfs', '/tmp:rw,nosuid,nodev,size=128m,mode=1777',
               '--workdir', '/tmp']
    for host, target in ((Path(worker_path).resolve(), '/worker/sandbox_worker.py'),
                         (Path(source_path).resolve(), '/candidate/source.py'),
                         (Path(panel_path).resolve(), '/panel/registered.json'),
                         (Path(runtime), runtime)):
        command += ['--mount', f'type=bind,src={host},dst={target},readonly']
    command += ['--entrypoint', '/usr/bin/env', image, '-i',
                'PATH=/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1',
                'PYTHONHASHSEED=0',
                'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
                'NUMBA_NUM_THREADS=1', 'NUMBA_CACHE_DIR=/tmp/numba-cache',
                '/usr/bin/timeout', '--signal=TERM', '--kill-after=2s', f'{outer_seconds}s',
                runtime+'/bin/python', '/worker/sandbox_worker.py']
    return command


def parse_worker_log(text):
    events = []
    for line in text.splitlines():
        if not line.startswith(RECORD_PREFIX):
            continue
        try:
            item = json.loads(line[len(RECORD_PREFIX):])
        except (ValueError, TypeError):
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def aggregate_worker(events, panel, *, source_hash, raw_path, container_state=None,
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
        route = row.get('route')
        n = len(item['coordinates'])
        if (not isinstance(route, list) or len(route) != n or
                any(type(x) is not int for x in route) or sorted(route) != list(range(n))):
            integrity_errors.append(dict(instance_id=item['instance_id'], reason='HOST_INVALID_ROUTE'))
            continue
        coords = item['coordinates']
        edges = [math.dist(coords[a], coords[b]) for a,b in zip(route, route[1:]+route[:1])]
        independent = math.fsum(edges)
        row['host_independent_length'] = independent
        length = row.get('length')
        if (not isinstance(length, (int,float)) or not math.isfinite(length) or
                not math.isclose(length, independent, rel_tol=1e-12, abs_tol=1e-12)):
            integrity_errors.append(dict(instance_id=item['instance_id'], reason='HOST_ROUTE_LENGTH_MISMATCH'))
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


class NativeEvaluator:
    def __init__(self, run_dir, cpu, config=None):
        self.root = Path(run_dir).resolve()
        self.cpu = str(cpu)
        self.config = dict(config or {})
        self.worker = Path(__file__).with_name('sandbox_worker.py').resolve()
        self.guard = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.binding = dict(version='NATIVE-EVALUATION-01', cache_scope=str(self.root),
                            cpu=self.cpu, cpu_count=1, memory='8g', threads=1,
                            image=IMAGE, runtime=RUNTIME,
                            worker_sha256=digest(self.worker.read_bytes()),
                            host_evaluator_sha256=digest(Path(__file__).read_bytes()),
                            load_seconds=10, search_panel_seconds=500,
                            validation_test_route_seconds=300,
                            isolation='network-none/read-only/nonroot/fresh-route-process',
                            search_timer='entire Search8 panel including all instance loads',
                            runtime_identity=self.config.get('runtime_identity', {}))
        freeze(self.root/'NATIVE_EVALUATOR_BINDING.json', self.binding)

    def _validate_panel_access(self, panel, panel_path):
        role = panel['role']
        expected = {'Search8':(8,50), 'Val128':(128,50), 'Audit128':(128,50), 'Test50':(100,50),
                    'Test100':(100,100), 'Test200':(100,200)}
        if panel.get('identity',{}).get('stage')=='C3':expected['Test50']=(256,50)
        if role not in expected or panel['task'] != 'TSP-Construct':
            raise ValueError('UNREGISTERED_PANEL_TASK_OR_ROLE')
        count, n = expected[role]
        if len(panel['instances']) != count or any(len(x['coordinates']) != n for x in panel['instances']):
            raise RuntimeError('PANEL_SHAPE_CONFLICT')
        required = dict(panel_seconds=500 if role=='Search8' else None,
                        route_seconds=None if role=='Search8' else 300, load_seconds=10)
        if panel['limits'] != required:
            raise RuntimeError('FROZEN_EVALUATION_LIMIT_CONFLICT')
        if role=='Audit128':
            lock=self.config.get('audit_lock_path')
            if not lock or not Path(lock).is_file():raise RuntimeError('AUDIT_SELECTION_LOCK_MISSING')
            receipt=read(lock)
            if receipt.get('all_new_Z_Search_terminal') is not True or digest(Path(panel_path).read_bytes()) not in [s['audit_panel']['sha256'] for s in receipt['sources']]:
                raise RuntimeError('AUDIT_BEFORE_FROZEN_SELECTION_OR_PANEL_CONFLICT')
        if role.startswith('Test'):
            lock = self.config.get('stage_lock_path')
            if not lock or not Path(lock).is_file():
                raise RuntimeError('TEST_LOCK_MISSING')
            receipt = read(lock)
            if receipt.get('all_outputs_and_val_locked') is not True:
                raise RuntimeError('TEST_BEFORE_ALL_OUTPUTS_AND_VAL_LOCKED')
            allowed = receipt.get('test_panel_hashes', [])
            if digest(Path(panel_path).read_bytes()) not in allowed:
                raise RuntimeError('TEST_PANEL_NOT_BOUND_TO_STAGE_LOCK')

    def key(self, source, panel_hash):
        return digest(encoded(dict(source_sha256=digest(source), panel_sha256=panel_hash,
                                   binding=self.binding, cache_scope=str(self.root))))

    def evaluate(self, source, panel_path, role):
        """role is a unique caller label (e.g. main/0001 or val/sourcehash)."""
        role_path = Path(role)
        if role_path.is_absolute() or '..' in role_path.parts or not role_path.parts:
            raise ValueError('UNSAFE_EVALUATION_ROLE')
        began = time.monotonic()
        with self.guard, exclusive(self.root/'.native-evaluation.lock'):
            wait_seconds = time.monotonic()-began
            return self._evaluate_locked(source, Path(panel_path).resolve(), role, wait_seconds)

    def _evaluate_locked(self, source, panel_path, role, wait_seconds):
        # Test access is checked from its metadata here; no generated program
        # can execute against any test bytes without the stage lock receipt.
        panel = read(panel_path)
        self._validate_panel_access(panel, panel_path)
        source_hash = digest(source)
        panel_hash = digest(panel_path.read_bytes())
        key = self.key(source, panel_hash)
        role_dir = self.root/'evaluations'/role
        role_result = role_dir/'EVALUATION.json'
        demand = dict(role=role, source_sha256=source_hash, panel_sha256=panel_hash,
                      panel_path=str(panel_path), cache_key=key, instance_count=len(panel['instances']))
        freeze(role_dir/'DEMAND.json', demand)
        if role_result.exists():
            return read(role_result)
        physical = self.root/'physical_evaluations'/key
        final_path = physical/'EVALUATION.json'
        role_intent_path = role_dir/'ROLE_INTENT.json'
        if role_intent_path.exists():
            role_intent = read(role_intent_path)
            if role_intent['cache_key'] != key:
                raise RuntimeError('ROLE_INTENT_IDENTITY_CONFLICT')
        else:
            role_intent = freeze(role_intent_path, dict(cache_key=key,
                                cache_hit_at_reservation=final_path.exists(),
                                queue_wait_seconds=wait_seconds, time_unix=time.time()))
        if final_path.exists():
            result = read(final_path)
        else:
            source_path = physical/'source.py'
            frozen_panel = physical/'panel.json'
            freeze_bytes(source_path, source.encode())
            freeze_bytes(frozen_panel, panel_path.read_bytes())
            # mkstemp defaults0600; these two nonsecret immutable mount files
            # must be readable by the sandbox's UID65534. No host directory or
            # credential mount permissions are changed.
            source_path.chmod(0o644)
            frozen_panel.chmod(0o644)
            result = self._execute(physical, source_path, frozen_panel, panel)
            freeze(final_path, result)
        hit = role_intent['cache_hit_at_reservation']
        return freeze(role_result, dict(result, role=role, panel_role=panel['role'],
                    cache_key=key, panel_sha256=panel_hash, cache_hit=hit,
                    cache_scope=str(self.root), role_responsibility=len(panel['instances']),
                    cached_responsibility=len(panel['instances']) if hit else 0,
                    physical_started_for_role=0 if hit else result['physical_started'],
                    active_seconds_for_role=0 if hit else result['active_seconds'],
                    cpu_slot_wait_seconds_for_role=0 if hit else result.get('cpu_slot_wait_seconds', 0),
                    queue_wait_seconds=role_intent['queue_wait_seconds'],
                    physical_record=str(final_path), terminal_unix=time.time()))

    def _execute(self, folder, source_path, panel_path, panel):
        """Serialize physical candidates on a shared CPU without holding it at stage barriers."""
        terminal = folder/'RAW_RESULT.json'
        if terminal.exists():
            return read(terminal)
        slots = self.config.get('cpu_slot_lock_dir')
        if not slots:
            return self._execute_locked(folder, source_path, panel_path, panel)
        began = time.monotonic()
        with exclusive(Path(slots)/('cpu-'+self.cpu+'.lock')):
            wait_path = folder/'CPU_SLOT.json'
            if wait_path.exists():
                wait = read(wait_path)
                if wait['cpu'] != self.cpu:
                    raise RuntimeError('CPU_SLOT_IDENTITY_CONFLICT')
            else:
                wait = freeze(wait_path, dict(cpu=self.cpu, wait_seconds=time.monotonic()-began,
                    acquired_unix=time.time(), scope='cross-process shared physical evaluation slot'))
            return self._execute_locked(folder, source_path, panel_path, panel,
                                        cpu_slot_wait_seconds=wait['wait_seconds'])

    def _execute_locked(self, folder, source_path, panel_path, panel, cpu_slot_wait_seconds=0):
        terminal = folder/'RAW_RESULT.json'
        if terminal.exists():
            return read(terminal)
        intent_path = folder/'INTENT.json'
        resumed = intent_path.exists()
        outer = 530 if panel['role']=='Search8' else 310*len(panel['instances'])+30
        name = 'fuse-native-' + digest(str(folder))[:24]
        command = container_command(self.worker, source_path, panel_path, name, self.cpu,
                                    outer_seconds=outer)
        if resumed:
            intent = read(intent_path)
            if intent['command'] != command:
                raise RuntimeError('EVALUATION_INTENT_COMMAND_CONFLICT')
        else:
            intent = freeze(intent_path, dict(command=command, container_name=name,
                            source_sha256=digest(source_path.read_bytes()),
                            panel_sha256=digest(panel_path.read_bytes()),
                            time_unix=time.time(), outer_seconds=outer, deadline_unix=time.time()+outer+15,
                            frozen_binding=self.binding))
        active_began = time.monotonic()
        if not resumed:
            try:
                with (folder/'client.stdout').open('xb') as out, (folder/'client.stderr').open('xb') as err:
                    submitted = subprocess.run(command, stdout=out, stderr=err, timeout=outer+15)
                freeze(folder/'SUBMISSION.json', dict(returncode=submitted.returncode,
                                                      time_unix=time.time()))
            except subprocess.TimeoutExpired:
                freeze(folder/'SUBMISSION_TIMEOUT.json', dict(timeout_seconds=outer+15,
                                                             time_unix=time.time()))
            except Exception as exc:
                freeze(folder/'SUBMISSION_ERROR.json', dict(type=type(exc).__name__, message=str(exc),
                                                           time_unix=time.time()))
        state, logs, error = self._collect_same_container(folder, intent)
        events = parse_worker_log(logs)
        witness = next((x for x in events if x.get('type')=='WORKER_STARTED'), None)
        if witness and (witness.get('source_sha256') != intent['source_sha256'] or
                        witness.get('panel_sha256') != intent['panel_sha256']):
            error = 'WORKER_SOURCE_OR_PANEL_IDENTITY_CONFLICT'
        active = None if resumed else time.monotonic()-active_began
        result = aggregate_worker(events, panel, source_hash=intent['source_sha256'],
                    raw_path=str(terminal), container_state=state,
                    infrastructure_error=error, active_seconds=active)
        result.update(events=events, resumed_existing_intent=resumed,
                      intent_path=str(intent_path), terminal_unix=time.time(),
                      cpu_slot_wait_seconds=cpu_slot_wait_seconds)
        # Save scientific terminal BEFORE any container removal. A crash in
        # collection resumes this record, never docker run or quality again.
        freeze(terminal, result)
        if state and not state.get('Running'):
            removed = subprocess.run(['docker','rm',name],capture_output=True,text=True,timeout=15)
            freeze(folder/'CONTAINER_REMOVAL.json', dict(returncode=removed.returncode,
                         stdout=removed.stdout, stderr=removed.stderr, time_unix=time.time()))
        return result

    def _collect_same_container(self, folder, intent):
        snapshot = folder/'COLLECTION_SNAPSHOT.json'
        if snapshot.exists():
            value = read(snapshot)
            return value['state'], (folder/'docker.log').read_text(), value['error']
        name = intent['container_name']
        state, error, logs = None, None, ''
        observations = []
        try:
            inspected = subprocess.run(['docker','inspect','--format','{{json .State}}',name],
                                        capture_output=True,text=True,timeout=15)
            observations.append(dict(action='inspect', returncode=inspected.returncode,
                                     stderr=inspected.stderr))
            if inspected.returncode:
                raise EvaluationInfrastructureError('INTENDED_CONTAINER_NOT_INSPECTABLE')
            state = json.loads(inspected.stdout)
            if state.get('Running'):
                remaining = max(.001, intent['deadline_unix']-time.time())
                try:
                    waited = subprocess.run(['docker','wait',name],capture_output=True,text=True,
                                            timeout=remaining)
                    observations.append(dict(action='wait', returncode=waited.returncode,
                                             stdout=waited.stdout, stderr=waited.stderr))
                except subprocess.TimeoutExpired:
                    killed = subprocess.run(['docker','kill',name],capture_output=True,text=True,timeout=15)
                    observations.append(dict(action='kill_after_original_deadline', returncode=killed.returncode,
                                             stderr=killed.stderr))
                inspected = subprocess.run(['docker','inspect','--format','{{json .State}}',name],
                                            capture_output=True,text=True,timeout=15)
                if inspected.returncode:
                    raise EvaluationInfrastructureError('CONTAINER_STATE_LOST_AFTER_WAIT')
                state = json.loads(inspected.stdout)
                if state.get('Running'):
                    raise EvaluationInfrastructureError('CONTAINER_RUNNING_AFTER_DEADLINE_KILL')
            captured = subprocess.run(['docker','logs',name],capture_output=True,text=True,timeout=20)
            logs = captured.stdout
            observations.append(dict(action='logs', returncode=captured.returncode, stderr=captured.stderr))
            if captured.returncode:
                raise EvaluationInfrastructureError('CONTAINER_LOG_COLLECTION_FAILED')
            if state.get('OOMKilled'):
                # A genuine candidate resource failure is not an infrastructure
                # stop. Worker may have no terminal; record its explicit limit.
                error = 'CANDIDATE_CONTAINER_OOM'
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
        freeze_bytes(folder/'docker.log', logs.encode())
        freeze(snapshot, dict(state=state, error=error, observations=observations,
                              time_unix=time.time()))
        return state, logs, error

    def accounting(self):
        roles = [read(p) for p in sorted((self.root/'evaluations').glob('**/EVALUATION.json'))]
        physical = [read(p) for p in sorted((self.root/'physical_evaluations').glob('*/EVALUATION.json'))]
        return dict(role_count=len(roles), role_responsibility=sum(x['role_responsibility'] for x in roles),
                    cached_responsibility=sum(x['cached_responsibility'] for x in roles),
                    physical_evaluations=len(physical),
                    physical_started_known=sum(x['physical_started'] for x in physical),
                    physical_started_uncertain=sum(bool(x.get('physical_started_uncertain')) for x in physical),
                    completed_instances=sum(x['completed'] for x in physical),
                    active_seconds_known=sum(x['active_seconds'] or 0 for x in physical),
                    active_seconds_missing=sum(x['active_seconds'] is None for x in physical),
                    load_seconds_known=sum(x['load_seconds_known'] for x in physical),
                    route_seconds_known=sum(x['route_seconds_known'] for x in physical),
                    child_cpu_user_seconds_known=sum(x.get('child_cpu_user_seconds_known',0) for x in physical),
                    child_cpu_system_seconds_known=sum(x.get('child_cpu_system_seconds_known',0) for x in physical),
                    child_cpu_missing_rows=sum(x.get('child_cpu_missing_rows',0) for x in physical),
                    failed_evaluations=sum(not x['complete'] for x in physical))


def evaluate(source, panelpath, role, run_dir, cpu, config=None):
    return NativeEvaluator(run_dir, cpu, config).evaluate(source, panelpath, role)
