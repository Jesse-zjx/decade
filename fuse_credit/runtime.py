"""Isolated v1 transport, panel registration, and full-program evaluation.

This module never changes the C3 runtime or its namespaces. Scientific I/O is
restricted to the authorized remote checkout. Unit tests replace those I/O
boundaries; they do not execute candidate code or contact a model service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

from fuse_addon import evaluation as ev
from fuse_addon import task_data
from fuse_addon import transport as tr

STUDY_ID = 'fuse-credit-action-v1-20260926'
REMOTE_ROOT = Path('/opt/decade/credit')
PANEL_SHAPES = {'Search32': (32, 50), 'Probe64': (64, 50), 'Val128': (128, 50),
                'Test50': (256, 50), 'Test100': (128, 100), 'Test200': (128, 200)}
MODEL_SETTINGS = dict(model=tr.MODEL, thinking={'type': 'disabled'}, max_tokens=8192,
                      temperature=1.0, top_p=1.0, stream=False)

# Explicit node-number tie breaking, unlike relying on the evaluator's neighbor
# ordering to settle a tie. This is a reference/scale, not an initialization gift.
NN_ANCHOR_SOURCE = '''import numpy as np\n\ndef select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:\n    return min((int(node) for node in unvisited_nodes), key=lambda node: (float(distance_matrix[current_node, node]), node))\n'''


def seed_for(phase, task, group, role, stream, *, study_id=STUDY_ID):
    """SHA256 of six UTF-8 fields joined by |, first 4 bytes unsigned big endian."""
    fields = tuple(map(str, (study_id, phase, task, group, role, stream)))
    if any('|' in field for field in fields):
        raise ValueError('SEED_NAMESPACE_DELIMITER_IN_FIELD')
    return int.from_bytes(hashlib.sha256('|'.join(fields).encode('utf-8')).digest()[:4], 'big')


def _derivation(phase, task, group, role, stream, study_id=STUDY_ID):
    seed_for(phase, task, group, role, stream, study_id=study_id)
    return '|'.join(map(str, (study_id, phase, task, group, role, stream)))


def panel_limits(role):
    return dict(panel_seconds=500 if role == 'Search32' else None,
                route_seconds=None if role == 'Search32' else 300, load_seconds=10)


def make_panel(path, *, phase, group, role, study_id=STUDY_ID,
               task='TSP-Construct', sampler_path=None):
    """Freeze a fresh arm-independent panel; no candidate or quality is run.

    Test256 is an alias for Test50 (256 instances, 50 cities). All instances and
    execution seeds are included in the panel hash. A role never shares its
    coordinate or instance-execution stream with another role or phase.
    """
    role = 'Test50' if role == 'Test256' else role
    if role not in PANEL_SHAPES or task != 'TSP-Construct':
        raise ValueError('UNREGISTERED_TSP_PANEL')
    path = Path(path).resolve()
    count, cities = PANEL_SHAPES[role]
    identity = dict(study_id=study_id, phase=str(phase), stage=str(phase), group=str(group),
                    role=role, count=count, cities=cities, task=task)
    if path.exists():
        panel = ev.read(path)
        if panel['identity'] != identity or panel['limits'] != panel_limits(role):
            raise RuntimeError('PANEL_IDENTITY_CONFLICT')
        return task_data.panel_manifest(path, panel)
    seed = seed_for(phase, task, group, role, 'coordinates', study_id=study_id)
    data, witness = task_data.sample_tsp(count, cities, seed, sampler_path)
    instances = [dict(instance_id=f'{phase}/{group}/{role}/{index:03d}',
                      coordinates=coordinates.tolist(), distance_matrix=distances.tolist(),
                      execution_seed=seed_for(phase, task, group, role, f'execution/{index:03d}',
                                              study_id=study_id))
                 for index, (coordinates, distances) in enumerate(data)]
    panel = dict(identity=identity, task=task, role=role, seed=seed,
                 seed_derivation=_derivation(phase, task, group, role, 'coordinates', study_id),
                 seed_conversion='SHA256 UTF-8 | separated fields; first4 bytes unsigned big endian',
                 execution_seed_derivation='study_id|phase|task|group|role|execution/NNN',
                 sampler=witness, instances=instances, limits=panel_limits(role),
                 independent_of_old_quality=True,
                 quality_access=('SEARCH' if role == 'Search32' else
                                 'AFTER_ALL_FOUR_GROUP_BRANCH_SEARCHES_LOCKED' if role == 'Probe64' else
                                 'POST_SEARCH_VAL' if role == 'Val128' else
                                 'AFTER_ALL_STAGE_OUTPUTS_AND_VAL_SELECTIONS_LOCKED'),
                 adopted_timeout_note='Historical Search8 whole-panel 500s retained for Search32; not 500s per route',
                 optimal_tour_reference=None,
                 gap_status='MISSING_NO_FROZEN_OPTIMAL_REFERENCE_FOR_NEW_INSTANCES')
    ev.freeze(path, panel)
    return task_data.panel_manifest(path, panel)


def _require_remote(root):
    root = Path(root).resolve()
    if platform.system() != 'Linux' or not root.is_relative_to(REMOTE_ROOT):
        raise RuntimeError('SCIENTIFIC_IO_ONLY_IN_AUTHORIZED_REMOTE_CHECKOUT')
    if not Path(ev.RUNTIME+'/bin/python').is_file():
        raise RuntimeError('AUTHORIZED_REMOTE_RUNTIME_MISSING')


class CreditEvaluator(ev.NativeEvaluator):
    """Native isolated evaluator with v1 panel barriers and exact common reuse.

    config keys: runtime_identity, cpu_slot_lock_dir, probe_lock_path,
    stage_lock_path, common_cache_roots. Common cache roots must be the current
    group's shared-state evaluator roots, not historical runs. Only complete
    matching evaluations may be reused. Their full responsibility is charged
    to the strategy; physical work remains charged to its real owner.
    """
    def __init__(self, run_dir, cpu, config=None):
        self.root = Path(run_dir).resolve()
        self.cpu = str(cpu)
        if not self.cpu.isdigit():
            raise ValueError('EXACTLY_ONE_CPU_REQUIRED')
        self.config = dict(config or {})
        self.worker = Path(ev.__file__).with_name('sandbox_worker.py').resolve()
        self.guard = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_binding = dict(version='CREDIT-EVALUATION-V1', study_id=STUDY_ID,
            cpu_count=1, memory='8g', threads=1, image=ev.IMAGE, runtime=ev.RUNTIME,
            worker_sha256=ev.digest(self.worker.read_bytes()),
            native_evaluator_sha256=ev.digest(Path(ev.__file__).read_bytes()),
            credit_runtime_sha256=ev.digest(Path(__file__).read_bytes()),
            load_seconds=10, search_panel_seconds=500, validation_probe_test_route_seconds=300,
            isolation='network-none/read-only/nonroot/fresh-route-process',
            search_timer='entire Search32 panel including all instance loads',
            historical_difference='Search8 to Search32; same 500s whole-panel limit',
            runtime_identity=self.config.get('runtime_identity', {}))
        self.binding = dict(self.execution_binding, cache_scope=str(self.root), cpu=self.cpu)
        ev.freeze(self.root/'CREDIT_EVALUATOR_BINDING.json', self.binding)

    def key(self, source, panel_hash):
        # Physical CPU slot identity is scheduling, not a source of free reuse.
        # runtime_identity must include the frozen machine/CPU class witness.
        return ev.digest(ev.encoded(dict(source_sha256=ev.digest(source), panel_sha256=panel_hash,
                                          execution_binding=self.execution_binding)))

    def _validate_panel_access(self, panel, panel_path):
        role = panel['role']
        identity = panel.get('identity', {})
        if role not in PANEL_SHAPES or panel.get('task') != 'TSP-Construct':
            raise ValueError('UNREGISTERED_PANEL_TASK_OR_ROLE')
        if identity.get('study_id') != STUDY_ID or identity.get('role') != role:
            raise RuntimeError('PANEL_STUDY_IDENTITY_CONFLICT')
        count, cities = PANEL_SHAPES[role]
        if (len(panel['instances']) != count or identity.get('count') != count or
                identity.get('cities') != cities or
                any(len(row['coordinates']) != cities for row in panel['instances'])):
            raise RuntimeError('PANEL_SHAPE_CONFLICT')
        if panel['limits'] != panel_limits(role):
            raise RuntimeError('FROZEN_EVALUATION_LIMIT_CONFLICT')
        panel_hash = ev.digest(Path(panel_path).read_bytes())
        if role == 'Probe64':
            lock = self.config.get('probe_lock_path')
            if not lock or not Path(lock).is_file():
                raise RuntimeError('PROBE_LOCK_MISSING')
            receipt = ev.read(lock)
            if (receipt.get('all_four_searches_locked') is not True or
                    set(receipt.get('branches', {})) != {'C', 'U', 'S', 'R'} or
                    any(not value for value in receipt.get('branches', {}).values()) or
                    receipt.get('probe_panel_sha256') != panel_hash or
                    any(str(receipt.get(key)) != str(identity.get(key))
                        for key in ('study_id', 'phase', 'group'))):
                raise RuntimeError('PROBE_BEFORE_FOUR_BRANCH_LOCK_OR_IDENTITY_CONFLICT')
        if role.startswith('Test'):
            lock = self.config.get('stage_lock_path')
            if not lock or not Path(lock).is_file():
                raise RuntimeError('TEST_LOCK_MISSING')
            receipt = ev.read(lock)
            if (receipt.get('all_outputs_and_val_locked') is not True or
                    receipt.get('study_id') != STUDY_ID or
                    str(receipt.get('phase')) != str(identity.get('phase')) or
                    panel_hash not in receipt.get('test_panel_hashes', [])):
                raise RuntimeError('TEST_BEFORE_ALL_OUTPUTS_AND_VAL_LOCKED_OR_IDENTITY_CONFLICT')

    def _common_result(self, key):
        for root in self.config.get('common_cache_roots', []):
            root = Path(root).resolve()
            path = root/'physical_evaluations'/key/'EVALUATION.json'
            if not path.exists():
                continue
            binding = ev.read(root/'CREDIT_EVALUATOR_BINDING.json')
            comparable = {k: v for k, v in binding.items() if k not in ('cache_scope', 'cpu')}
            if comparable != self.execution_binding:
                raise RuntimeError('COMMON_CACHE_EXECUTION_BINDING_CONFLICT')
            result = ev.read(path)
            # Unknown/incomplete original evaluations are never replayed as
            # cache work. The inherited physical intent still forbids replay.
            if result.get('complete') is True:
                return path, result
        return None

    def _evaluate_locked(self, source, panel_path, role, wait_seconds):
        panel = ev.read(panel_path)
        self._validate_panel_access(panel, panel_path)
        panel_hash = ev.digest(panel_path.read_bytes())
        key = self.key(source, panel_hash)
        role_dir = self.root/'evaluations'/role
        demand = dict(role=role, source_sha256=ev.digest(source), panel_sha256=panel_hash,
                      panel_path=str(panel_path), cache_key=key, instance_count=len(panel['instances']))
        ev.freeze(role_dir/'DEMAND.json', demand)
        if (role_dir/'EVALUATION.json').exists():
            return ev.read(role_dir/'EVALUATION.json')
        physical = self.root/'physical_evaluations'/key
        final_path = physical/'EVALUATION.json'
        reservation = role_dir/'ROLE_INTENT.json'
        if reservation.exists():
            intent = ev.read(reservation)
            if intent['cache_key'] != key:
                raise RuntimeError('ROLE_INTENT_IDENTITY_CONFLICT')
        else:
            common = self._common_result(key) if not final_path.exists() else None
            origin = str(common[0]) if common else str(final_path)
            intent = ev.freeze(reservation, dict(cache_key=key, physical_record=origin,
                cache_hit_at_reservation=bool(final_path.exists() or common),
                common_state_reuse=bool(common), queue_wait_seconds=wait_seconds, time_unix=time.time()))
        actual_path = Path(intent['physical_record'])
        if actual_path.exists():
            result = ev.read(actual_path)
            if (result['source_hash'] != ev.digest(source) or
                    result.get('panel_sha256') != panel_hash or
                    result.get('execution_binding') != self.execution_binding):
                raise RuntimeError('CACHED_SOURCE_PANEL_OR_EXECUTION_IDENTITY_CONFLICT')
        else:
            if intent['cache_hit_at_reservation']:
                raise RuntimeError('FROZEN_CACHE_RECORD_LOST')
            source_path, frozen_panel = physical/'source.py', physical/'panel.json'
            ev.freeze_bytes(source_path, source.encode())
            ev.freeze_bytes(frozen_panel, panel_path.read_bytes())
            source_path.chmod(0o644)
            frozen_panel.chmod(0o644)
            result = self._execute(physical, source_path, frozen_panel, panel)
            ev.freeze(final_path, result)
        hit = intent['cache_hit_at_reservation']
        return ev.freeze(role_dir/'EVALUATION.json', dict(result, role=role, panel_role=panel['role'],
            cache_key=key, panel_sha256=panel_hash, cache_hit=hit,
            common_state_reuse=intent['common_state_reuse'], cache_scope=str(self.root),
            role_responsibility=len(panel['instances']),
            cached_responsibility=len(panel['instances']) if hit else 0,
            physical_started_for_role=0 if hit else result['physical_started'],
            active_seconds_for_role=0 if hit else result['active_seconds'],
            cpu_slot_wait_seconds_for_role=0 if hit else result.get('cpu_slot_wait_seconds', 0),
            queue_wait_seconds=intent['queue_wait_seconds'], physical_record=str(actual_path),
            physical_record_sha256=ev.digest(actual_path.read_bytes()), terminal_unix=time.time()))

    def _execute_locked(self, folder, source_path, panel_path, panel, cpu_slot_wait_seconds=0):
        _require_remote(self.root)
        terminal = folder/'RAW_RESULT.json'
        if terminal.exists():
            return ev.read(terminal)
        intent_path = folder/'INTENT.json'
        resumed = intent_path.exists()
        outer = 530 if panel['limits']['panel_seconds'] is not None else 310*len(panel['instances'])+30
        name = 'fuse-credit-' + ev.digest(str(folder))[:24]
        command = ev.container_command(self.worker, source_path, panel_path, name, self.cpu,
                                        outer_seconds=outer)
        if resumed:
            intent = ev.read(intent_path)
            if intent['command'] != command:
                raise RuntimeError('EVALUATION_INTENT_COMMAND_CONFLICT')
        else:
            now = time.time()
            intent = ev.freeze(intent_path, dict(command=command, container_name=name,
                source_sha256=ev.digest(source_path.read_bytes()), panel_sha256=ev.digest(panel_path.read_bytes()),
                time_unix=now, outer_seconds=outer, deadline_unix=now+outer+15, frozen_binding=self.binding))
        active_began = time.monotonic()
        if not resumed:
            try:
                with (folder/'client.stdout').open('xb') as out, (folder/'client.stderr').open('xb') as err:
                    submitted = subprocess.run(command, stdout=out, stderr=err, timeout=outer+15)
                ev.freeze(folder/'SUBMISSION.json', dict(returncode=submitted.returncode, time_unix=time.time()))
            except subprocess.TimeoutExpired:
                ev.freeze(folder/'SUBMISSION_TIMEOUT.json', dict(timeout_seconds=outer+15, time_unix=time.time()))
            except Exception as exc:
                ev.freeze(folder/'SUBMISSION_ERROR.json', dict(type=type(exc).__name__, message=str(exc), time_unix=time.time()))
        state, logs, error = self._collect_same_container(folder, intent)
        events = ev.parse_worker_log(logs)
        witness = next((x for x in events if x.get('type') == 'WORKER_STARTED'), None)
        if witness and (witness.get('source_sha256') != intent['source_sha256'] or
                        witness.get('panel_sha256') != intent['panel_sha256']):
            error = 'WORKER_SOURCE_OR_PANEL_IDENTITY_CONFLICT'
        result = ev.aggregate_worker(events, panel, source_hash=intent['source_sha256'],
                    raw_path=str(terminal), container_state=state, infrastructure_error=error,
                    active_seconds=None if resumed else time.monotonic()-active_began)
        result.update(events=events, resumed_existing_intent=resumed, intent_path=str(intent_path),
                      terminal_unix=time.time(), cpu_slot_wait_seconds=cpu_slot_wait_seconds,
                      panel_sha256=intent['panel_sha256'], execution_binding=self.execution_binding)
        ev.freeze(terminal, result)
        if state and not state.get('Running'):
            removed = subprocess.run(['docker', 'rm', name], capture_output=True, text=True, timeout=15)
            ev.freeze(folder/'CONTAINER_REMOVAL.json', dict(returncode=removed.returncode,
                         stdout=removed.stdout, stderr=removed.stderr, time_unix=time.time()))
        return result


def payload(prompt, config=None):
    settings = dict(MODEL_SETTINGS)
    config = dict(config or {})
    # Omission is allowed only as a frozen provider compatibility adjustment,
    # never selected after quality. No change of model or capability is allowed.
    unsupported = tuple(config.get('unsupported_sampling_fields', ()))
    if any(field not in ('temperature', 'top_p') for field in unsupported):
        raise ValueError('UNAUTHORIZED_MODEL_SETTING_CHANGE')
    if unsupported and not config.get('provider_compatibility_evidence'):
        raise ValueError('SAMPLING_OMISSION_REQUIRES_FROZEN_PROVIDER_EVIDENCE')
    for field in unsupported:
        settings.pop(field)
    return dict(settings, messages=[dict(role='user', content=prompt)])


def payload_utf8_bytes(prompt, config=None):
    """Actual canonical JSON body bytes, excluding secret HTTP headers."""
    return len(tr.canonical(payload(prompt, config)).encode('utf-8'))


def retryable_failure(result):
    """Only a confirmed retryable rejection; no UNKNOWN or quality retry."""
    return bool(not result.get('service_success') and not result.get('fatal_reason') and
                result.get('status') == 'SERVICE_ERROR' and result.get('body_complete') is True and
                result.get('http_status') in (408, 429, 500, 502, 503, 504))


class CreditTransport:
    """One logical generation, at most three separately durable physical sends.

    A complete model response ends generation even if invalid or low quality.
    UNKNOWN is terminal and never resent. Proved pre-network host failure raises
    HostFailure for scoped engineering recovery, rather than inventing quality.
    """
    def __init__(self, run_dir, service_root, credential_file, run_id=None, config=None):
        self.root = Path(run_dir).resolve()
        self.service = tr.Circuit(service_root)
        self.credentials = str(credential_file)
        tr.credentials(self.credentials)  # validates by path; never log values
        self.run_id = run_id or self.root.name
        self.config = dict(config or {})
        self.counter = 0
        self.records = []
        self.logical_records = []
        self.guard = threading.RLock()
        self.binding = dict(study_id=STUDY_ID, model_settings={k: v for k, v in payload('', self.config).items() if k != 'messages'},
                            max_physical_attempts=3, unknown_replay=False,
                            retry_policy='complete retryable HTTP rejection only; first actual model response ends task',
                            service_namespace=str(Path(service_root).resolve()),
                            credential_file=self.credentials,
                            native_transport_sha256=tr.sha(Path(tr.__file__).read_bytes()),
                            runtime_sha256=tr.sha(Path(__file__).read_bytes()))
        tr.write(self.root/'CREDIT_TRANSPORT_BINDING.json', self.binding)

    def __call__(self, prompt, metadata):
        with self.guard, ev.exclusive(self.root/'.credit-transport.lock'):
            number = self.counter
            self.counter += 1
            folder = self.root/'requests'/f'{number:06d}'
            body = tr.canonical(payload(prompt, self.config)).encode()
            frozen = dict(prompt=prompt, metadata=metadata, request_index=number, run_id=self.run_id)
            identity = dict(payload_sha256=tr.sha(body), frozen_sha256=tr.sha(tr.canonical(frozen)))
            tr.write(folder/'FROZEN_INPUT.json', frozen)
            tr.write_bytes(folder/'payload.json', body)
            logical = self.run_id+f'/{number:06d}'
            # Reconcile each persisted physical attempt through the same circuit
            # even after a crash between terminal persistence and observation.
            physical = []
            for attempt in range(1, 4):
                result = self._attempt(folder/'attempts'/f'{attempt:02d}', body, frozen,
                                       identity, logical, metadata, attempt)
                physical.append(result)
                self.records.append(result)
                if not retryable_failure(result) or attempt == 3:
                    break
                tr.write(folder/'attempts'/f'{attempt:02d}'/'RETRY_ELIGIBLE.json',
                         dict(reason='CONFIRMED_RETRYABLE_SERVICE_REJECTION', payload_sha256=identity['payload_sha256'],
                              next_attempt=attempt+1, no_quality_selection=True))
            final = dict(physical[-1], logical_id=logical, attempts=[x['request_path'] for x in physical],
                         attempt_count=len(physical), unknown_replayed=False)
            tr.write(folder/'terminal.json', final)
            self.logical_records.append(final)
            if final.get('fatal_reason'):
                raise tr.ServiceStopped(final['fatal_reason'])
            return final['content'] if final['service_success'] else None

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
            done = subprocess.run([sys.executable, '-m', 'fuse_credit.runtime', '--submit',
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

    def accounting(self):
        intents = sorted((self.root/'requests').glob('*/attempts/*/intent.json'))
        rows = [tr.read(p.parent/'terminal.json') for p in intents
                if (p.parent/'terminal.json').exists()]
        usage_keys = sorted({key for row in rows for key, value in (row.get('usage') or {}).items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)})
        pending = len(intents)-len(rows)
        return dict(physical_attempts=len(intents), terminal_attempts=len(rows), pending_or_host_failure=pending,
                    confirmed_responses=sum(bool(row['service_success']) for row in rows),
                    unknown=sum(row['status'] == 'UNKNOWN' for row in rows),
                    usage_missing=sum(row.get('usage') is None for row in rows)+pending,
                    # Report provider fields separately; never add total to input/output.
                    usage_known_totals={key: sum(row['usage'][key] for row in rows if
                        isinstance((row.get('usage') or {}).get(key), (int, float))) for key in usage_keys},
                    usage_field_missing={key: pending+sum((row.get('usage') or {}).get(key) is None for row in rows)
                                         for key in usage_keys},
                    boundary_intents=sum((p.parent/'NETWORK_BOUNDARY_INTENT.json').exists() for p in intents),
                    proved_pre_network_host_failures=sum((p.parent/'SENDER_HOST_FAILURE.json').exists()
                                                        for p in intents),
                    llm_wait_seconds_known=sum((tr.read(p.parent/'process.json').get('wall_seconds') or 0)
                                              for p in intents if (p.parent/'process.json').exists()),
                    llm_wait_seconds_missing=sum(not (p.parent/'process.json').exists() or
                                                tr.read(p.parent/'process.json').get('wall_seconds') is None
                                                for p in intents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', required=True)
    parser.add_argument('--credentials', required=True)
    args = parser.parse_args()
    _require_remote(args.submit)
    tr.sender(args.submit, args.credentials)
