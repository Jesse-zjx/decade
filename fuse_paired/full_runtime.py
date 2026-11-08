"""Fresh V2 development/confirmation I/O; no historical study asset reuse.

The frozen micro runtime remains unchanged. This wrapper adds full-stage
barriers, a group NN scale/reference, response checks and restart accounting.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from fuse_addon import evaluation as ev
from fuse_addon import transport as tr
from . import runtime as paired_runtime
from .runtime import (PairedEvaluator, PairedTransport, STUDY_ID, MODEL_SETTINGS,
                      NN_ANCHOR_SOURCE, CANDIDATE_CPUS, seed_for)
from .runtime import payload, retryable_failure

PHASE_GROUPS = {'dev': 4, 'confirm': 20}
ARMS = ('F', 'N', 'ROOT', 'E', 'H', 'R', 'N+')
TASK = 'TSP-Construct'


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _usage(row):
    value = row.get('usage')
    return value if isinstance(value, dict) else {}


def _messages(messages):
    if not isinstance(messages, (list, tuple)) or not messages or any(
            not isinstance(row, dict) or set(row) != {'role', 'content'} or
            row['role'] not in ('system', 'user', 'assistant') or
            not isinstance(row['content'], str) for row in messages):
        raise ValueError('NATIVE_MESSAGES_REQUIRE_EXACT_ROLE_CONTENT_RECORDS')
    return copy.deepcopy(list(messages))


def flatten_native_messages(messages):
    """Unambiguous PRE text; HTTP retains the original role/content messages."""
    return tr.canonical(_messages(messages))


def _file_within(study, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (Path(study)/path).resolve()
    if not path.is_relative_to(Path(study).resolve()) or not path.is_file():
        raise RuntimeError('FULL_STAGE_RECORD_OUTSIDE_CURRENT_STUDY_OR_MISSING')
    return path


class FullEvaluator(PairedEvaluator):
    """Same sandbox, new full-stage binding and strict Val/Test barriers."""
    def __init__(self, run_dir, cpu, config=None):
        self.root = Path(run_dir).resolve()
        self.cpu = str(cpu)
        if self.cpu not in CANDIDATE_CPUS:
            raise ValueError('CPU_OUTSIDE_V2_FOUR_RESERVED_SLOTS')
        self.config = dict(config or {})
        if any(key.startswith('imported_common') for key in self.config):
            raise ValueError('FULL_STAGE_CANNOT_IMPORT_HISTORICAL_COMMON')
        if any('imported_common' in Path(path).parts for path in self.config.get('common_cache_roots', [])):
            raise ValueError('FULL_STAGE_CANNOT_CACHE_HISTORICAL_COMMON')
        continuation = self.config.get('continuation_cache_root')
        if continuation and (self.root.parent.name != 'N+' or
                Path(continuation).resolve() != self.root.parent.parent/'N/evaluator'):
            raise ValueError('CONTINUATION_CACHE_REQUIRES_SAME_GROUP_N_OWNER')
        self.worker = Path(ev.__file__).with_name('sandbox_worker.py').resolve()
        self.guard = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_binding = dict(version='PAIRED-FULL-EVALUATION-V2', study_id=STUDY_ID,
            cpu_count=1, memory='8g', threads=1, image=ev.IMAGE, runtime=ev.RUNTIME,
            worker_sha256=ev.digest(self.worker.read_bytes()),
            native_evaluator_sha256=ev.digest(Path(ev.__file__).read_bytes()),
            inherited_paired_runtime_sha256=ev.digest(Path(paired_runtime.__file__).read_bytes()),
            full_runtime_sha256=ev.digest(Path(__file__).read_bytes()),
            load_seconds=10, search_panel_seconds=500, validation_probe_test_route_seconds=300,
            isolation='network-none/read-only/nonroot/fresh-route-process',
            search_timer='entire Search32 panel including all instance loads',
            runtime_identity=self.config.get('runtime_identity', {}))
        self.binding = dict(self.execution_binding, cache_scope=str(self.root), cpu=self.cpu)
        ev.freeze(self.root/'CREDIT_EVALUATOR_BINDING.json', self.binding)

    def _historical_result(self, source, panel_hash):
        return None

    def _common_result(self, key):
        # Shared references remain complete-only. N+ is an exact continuation
        # of N: an already terminal failed/UNKNOWN endpoint is an inherited
        # fact too, and must never gain a fresh physical attempt on recovery.
        value = super()._common_result(key)
        if value is not None:
            return value
        inherited = self.config.get('continuation_cache_root')
        if not inherited:
            return None
        root = Path(inherited).resolve()
        path = root/'physical_evaluations'/key/'EVALUATION.json'
        if not path.exists():
            if path.parent.exists():
                raise RuntimeError('INHERITED_ENDPOINT_TERMINAL_MISSING')
            return None
        binding = ev.read(root/'CREDIT_EVALUATOR_BINDING.json')
        if {k: v for k, v in binding.items() if k not in ('cache_scope', 'cpu')} != self.execution_binding:
            raise RuntimeError('CONTINUATION_CACHE_EXECUTION_BINDING_CONFLICT')
        return path, ev.read(path)

    def _allow_imported_panel(self, panel, panel_path):
        raise RuntimeError('FULL_STAGE_PANEL_MUST_BE_FRESH_V2')

    def _validate_panel_access(self, panel, panel_path):
        identity = panel.get('identity', {})
        phase = identity.get('phase')
        if (phase not in PHASE_GROUPS or panel.get('role') == 'Probe64' or
                str(identity.get('group')) not in {str(g) for g in range(PHASE_GROUPS[phase])}):
            raise RuntimeError('FULL_STAGE_REJECTS_MICRO_OR_UNREGISTERED_PANEL')
        super()._validate_panel_access(panel, panel_path)
        role = panel['role']
        if role == 'Val128':
            lock = self.config.get('val_lock_path')
            if not lock or not Path(lock).is_file():
                raise RuntimeError('ALL_DEV_PREFIX_LOCK_MISSING')
            receipt = ev.read(lock)
            if (phase != 'dev' or receipt.get('study_id') != STUDY_ID or receipt.get('phase') != 'dev' or
                    receipt.get('all_dev_prefixes_locked') is not True or
                    ev.digest(Path(panel_path).read_bytes()) not in receipt.get('val_panel_hashes', [])):
                raise RuntimeError('VAL_BEFORE_ALL_DEV_PREFIX_LOCKS_OR_PANEL_CONFLICT')
            self._validate_output_locks(receipt, lock, phase)
        elif role.startswith('Test'):
            receipt = ev.read(self.config['stage_lock_path'])
            if (phase != 'confirm' or
                    receipt.get('all_confirmation_search_and_budget_locked') is not True):
                raise RuntimeError('TEST_BEFORE_ALL_CONFIRMATION_SEARCH_AND_BUDGET_LOCKS')
            self._validate_output_locks(receipt, self.config['stage_lock_path'], phase)

    def _validate_output_locks(self, receipt, lock, phase):
        lock = Path(lock).resolve()
        if (lock.name != 'ALL_OUTPUTS_LOCKED.json' or lock.parent.name != phase or
                lock.parent.parent.name != 'stages'):
            raise RuntimeError('FULL_STAGE_LOCK_PATH_CONFLICT')
        study = lock.parent.parent.parent
        outputs = receipt.get('outputs', {})
        if set(outputs) != {f'g{g:02d}' for g in range(PHASE_GROUPS[phase])}:
            raise RuntimeError('FULL_STAGE_GROUP_LOCKS_INCOMPLETE')
        for group, arms in outputs.items():
            if set(arms) != set(ARMS):
                raise RuntimeError('FULL_STAGE_ARM_LOCKS_INCOMPLETE')
            for arm, points in arms.items():
                expected = {'64', '96', '128'} if arm == 'F' else {'final'} if arm == 'N+' else {'128'}
                if set(points) != expected:
                    raise RuntimeError('FULL_STAGE_PREFIX_LOCKS_INCOMPLETE')
                for prefix, record in points.items():
                    path = _file_within(study, record['path'])
                    if ev.digest(path.read_bytes()) != record['sha256']:
                        raise RuntimeError('FULL_STAGE_OUTPUT_HASH_CONFLICT')
                    output = ev.read(path)
                    if (output.get('study_id') != STUDY_ID or output.get('phase') != phase or
                            str(output.get('group')) != str(int(group[1:])) or output.get('arm') != arm or
                            str(output.get('prefix')) != prefix or output.get('search_only') is not True):
                        raise RuntimeError('FULL_STAGE_OUTPUT_IDENTITY_CONFLICT')
                    source = _file_within(study, output['source_program_path'])
                    if ev.digest(source.read_bytes()) != output.get('source_hash'):
                        raise RuntimeError('FULL_STAGE_SELECTED_SOURCE_CONFLICT')
                    search = output['search_result']
                    search_path = _file_within(study, search['path'])
                    if ev.digest(search_path.read_bytes()) != search['sha256']:
                        raise RuntimeError('FULL_STAGE_SEARCH_RESULT_CHANGED')


class FullTransport(PairedTransport):
    """Validate every confirmed generation or auxiliary response, including replay."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._message_override = None
        ev.freeze(self.root/'FULL_TRANSPORT_BINDING.json', dict(study_id=STUDY_ID,
            full_runtime_sha256=ev.digest(Path(__file__).read_bytes()),
            inherited_transport_binding=self.binding, model_check='EVERY_CONFIRMED_RESPONSE_INCLUDING_AUXILIARY',
            native_message_policy='EXACT_ROLE_CONTENT_LIST_SINGLE_CALL_IMMUTABLE_PAYLOAD'))

    @contextmanager
    def native_messages(self, messages):
        with self.guard:
            if self._message_override is not None:
                raise RuntimeError('NESTED_NATIVE_MESSAGE_OVERRIDE')
            self._message_override = _messages(messages)
            try:
                yield
            finally:
                self._message_override = None

    def request_payload(self,prompt,metadata):
        return payload(prompt,self.config)

    def __call__(self, prompt, metadata):
        # Small exact adaptation of PairedTransport.__call__: only messages and
        # their immutable provenance differ. Its physical sender/reconciliation,
        # UNKNOWN handling and bounded confirmed-service retries are inherited.
        with self.guard, ev.exclusive(self.root/'.paired-transport.lock'):
            messages, self._message_override = self._message_override, None
            number = self.counter
            self.counter += 1
            folder = self.root/'requests'/f'{number:06d}'
            body_object = self.request_payload(prompt, metadata)
            metadata = copy.deepcopy(metadata)
            frozen = dict(prompt=prompt, metadata=metadata, request_index=number, run_id=self.run_id)
            if messages is not None:
                if prompt != flatten_native_messages(messages):
                    raise RuntimeError('NATIVE_MESSAGES_FLATTENED_PROVENANCE_CONFLICT')
                body_object['messages'] = messages
                metadata['native_messages_sha256'] = tr.sha(tr.canonical(messages))
                frozen['native_messages'] = messages
            body = tr.canonical(body_object).encode()
            identity = dict(payload_sha256=tr.sha(body), frozen_sha256=tr.sha(tr.canonical(frozen)))
            tr.write(folder/'FROZEN_INPUT.json', frozen)
            tr.write_bytes(folder/'payload.json', body)
            logical = self.run_id+f'/{number:06d}'
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
            terminal = dict(physical[-1], logical_id=logical, attempts=[x['request_path'] for x in physical],
                            attempt_count=len(physical), unknown_replayed=False)
            tr.write(folder/'terminal.json', terminal)
            self.logical_records.append(terminal)
            if terminal.get('fatal_reason'):
                raise tr.ServiceStopped(terminal['fatal_reason'])
            self._verify_model(terminal, metadata, folder)
            return terminal['content'] if terminal['service_success'] else None

    def _verify_model(self, terminal, metadata, folder):
        if terminal.get('service_success'):
            model = terminal.get('response_model')
            if not model or str(model).lower() != MODEL_SETTINGS['model'].lower():
                raise RuntimeError('ACTUAL_MODEL_IDENTITY_UNVERIFIED_OR_CHANGED')
            usage = _usage(terminal)
            details = usage.get('completion_tokens_details')
            reasoning = details.get('reasoning_tokens') if isinstance(details, dict) else None
            if reasoning is not None and reasoning != 0:
                raise RuntimeError('ACTUAL_THINKING_CAPABILITY_CONFLICT')
            ev.freeze(folder/'MODEL_CHECK.json',
                dict(response_model=model, requested_settings={k:v for k,v in self.request_payload('',metadata).items() if k!='messages'},
                     usage=terminal.get('usage'), reasoning_tokens=reasoning,
                     logical_id=terminal['logical_id'], metadata=metadata,
                     auxiliary=bool(metadata.get('native_auxiliary'))))


def physical_cost_snapshot(root, *, started=None):
    """Count actual work in this root only; unknown usage/time never becomes zero."""
    root = Path(root)
    folders = {p.parent for pattern in ('*/attempts/*/intent.json', '*/attempts/*/terminal.json')
               for p in (root/'transport/requests').glob(pattern)}
    requests = []
    for folder in sorted(folders):
        intent = (folder/'intent.json').is_file()
        terminal = ev.read(folder/'terminal.json') if (folder/'terminal.json').is_file() else {}
        process = ev.read(folder/'process.json') if (folder/'process.json').is_file() else {}
        requests.append((intent, terminal, process, folder))
    totals = [_usage(row).get('total_tokens') for _, row, _, _ in requests]
    missing = sum(not intent or not _number(value) for (intent, _, _, _), value in zip(requests, totals))
    known = sum(value for value in totals if _number(value))
    physical_paths = {p.parent for pattern in ('*/INTENT.json', '*/EVALUATION.json')
                      for p in (root/'evaluator/physical_evaluations').glob(pattern)}
    physical = [ev.read(path/'EVALUATION.json') if (path/'EVALUATION.json').is_file()
                else {} for path in sorted(physical_paths)]
    roles = [ev.read(path) for path in (root/'evaluator/evaluations').glob('**/EVALUATION.json')]
    add = lambda key: sum(row[key] for row in physical if _number(row.get(key)))
    cpu = add('child_cpu_user_seconds_known')+add('child_cpu_system_seconds_known')
    cpu_missing = sum(row.get('child_cpu_missing_rows', 0) for row in physical
                      if _number(row.get('child_cpu_missing_rows')))
    cpu_missing += sum(not _number(row.get('child_cpu_user_seconds_known')) or
                       not _number(row.get('child_cpu_system_seconds_known')) for row in physical)
    active_missing = sum(not _number(row.get('active_seconds')) for row in physical)
    if started is None and (root/'STARTED.json').is_file():
        started = ev.read(root/'STARTED.json')['unix']
    wall = max(0., time.time()-started) if started is not None else None
    return dict(http=sum(intent for intent, _, _, _ in requests),
        response_received=sum(row.get('service_success') is True for _, row, _, _ in requests),
        unknown=sum(row.get('status') == 'UNKNOWN' for _, row, _, _ in requests),
        pending_attempts=sum(not row for _, row, _, _ in requests),
        tokens_known=known, tokens_missing=missing, tokens=known if not missing else None,
        input_tokens_known=sum(value for _, row, _, _ in requests
                               if _number(value := _usage(row).get('prompt_tokens'))),
        input_tokens_missing=sum(not _number(_usage(row).get('prompt_tokens')) for _, row, _, _ in requests),
        output_tokens_known=sum(value for _, row, _, _ in requests
                                if _number(value := _usage(row).get('completion_tokens'))),
        output_tokens_missing=sum(not _number(_usage(row).get('completion_tokens')) for _, row, _, _ in requests),
        llm_wall_seconds_known=sum(process['wall_seconds'] for _, _, process, _ in requests
                                  if _number(process.get('wall_seconds'))),
        llm_wall_seconds_missing=sum(not _number(process.get('wall_seconds')) for _, _, process, _ in requests),
        network_boundary_intents=sum((folder/'NETWORK_BOUNDARY_INTENT.json').is_file() for _, _, _, folder in requests),
        evaluation_roles=len(roles), physical_evaluations=len(physical), physical_instances=add('physical_started'),
        physical_unknown=sum(row.get('physical_started_uncertain') is not False for row in physical),
        evaluator_active_seconds=add('active_seconds') if not active_missing else None,
        evaluator_active_seconds_known=add('active_seconds'), evaluator_active_seconds_missing=active_missing,
        evaluator_queue_seconds=add('cpu_slot_wait_seconds'),
        evaluator_cpu_seconds=cpu if not cpu_missing else None, evaluator_cpu_seconds_known=cpu,
        cpu_seconds=cpu if not cpu_missing else None, cpu_seconds_known=cpu,
        evaluator_cpu_missing_rows=cpu_missing,
        evaluation_failures=sum(row.get('complete') is False for row in physical),
        evaluation_pending=sum(not row for row in physical), wall_seconds=wall)


def strategy_cost(io):
    """Standalone fresh strategy: full group reference plus own new physical work."""
    own = io.own_cost_snapshot()
    reference = copy.deepcopy(io.reference['cost'])
    keys = set(own) | set(reference)
    combined = {}
    for key in keys:
        left, right = own.get(key, 0), reference.get(key, 0)
        if left is None or right is None:
            combined[key] = None
        elif _number(left) and _number(right):
            combined[key] = left+right
    combined.update(reference_cost=reference, own_physical_cost=own,
                    reference_charged_in_full=True, historical_common_cost=None,
                    cost_scope='FULL_GROUP_REFERENCE_PLUS_OWN_FRESH_WORK')
    return combined


def _fresh_state(context, reference):
    # V1-compatible archive bookkeeping; no Ridge/controller is instantiated.
    return dict(schema='paired-full-state-v2', context=context, programs={}, qualities={},
        credits={}, credit_history=[], occurrences={}, occurrence_counter=0, history=[],
        parent_visits={}, target_visits={}, op_attempts={}, op_valid={}, completion_history={},
        cf_last_processed={}, cf_events=[], noninit_completed=0, init_completed=0,
        archive_counter=0, scale=reference['mean'], reference_hash=reference['source_hash'],
        controllers={}, failures=[])


class FreshSearchIO:
    def __init__(self, study, phase, group, arm, cpu, credential_file):
        self.study = Path(study).resolve()
        if phase not in PHASE_GROUPS or type(group) is not int or group not in range(PHASE_GROUPS[phase]) or arm not in ARMS:
            raise ValueError('UNREGISTERED_FRESH_STUDY_IDENTITY')
        self.phase, self.group, self.arm = phase, group, arm
        self.root = self.study/'searches'/phase/f'g{group:02d}'/arm
        self.root.mkdir(parents=True, exist_ok=True)
        self.panel = self.study/'panels'/phase/f'g{group:02d}'/'Search32.json'
        panel = ev.read(self.panel)
        if (panel['identity'].get('study_id') != STUDY_ID or panel['identity'].get('phase') != phase or
                str(panel['identity'].get('group')) != str(group) or panel.get('role') != 'Search32'):
            raise RuntimeError('FRESH_SEARCH_PANEL_IDENTITY_CONFLICT')
        reference_path = self.study/'references'/phase/f'g{group:02d}'/'REFERENCE.json'
        reference = ev.read(reference_path)
        reference_source = reference.get('source', NN_ANCHOR_SOURCE)
        if reference.get('source_hash') != ev.digest(NN_ANCHOR_SOURCE) or reference_source != NN_ANCHOR_SOURCE:
            raise RuntimeError('FRESH_REFERENCE_SOURCE_CONFLICT')
        reference_root = Path(reference['evaluator_root']).resolve()
        if not reference_root.is_relative_to(self.study) or 'imported_common' in reference_root.parts:
            raise RuntimeError('FRESH_REFERENCE_CACHE_OUTSIDE_STUDY')
        config = dict(runtime_identity=ev.read(self.study/'engineering/RUNTIME_IDENTITY.json'),
                      cpu_slot_lock_dir=str(self.study/'cpu-slots'), common_cache_roots=[str(reference_root)])
        if arm == 'N+':
            # N+ continues exactly its own group's N prefix. The inherited
            # strategy responsibility already includes these complete results.
            config['continuation_cache_root'] = str(self.study/'searches'/phase/f'g{group:02d}'/'N/evaluator')
            config['common_cache_roots'].append(config['continuation_cache_root'])
        self.evaluator = FullEvaluator(self.root/'evaluator', cpu, config)
        quality = ev.read(_file_within(self.study, reference['evaluation_path']))
        if (quality.get('complete') is not True or quality.get('source_hash') != reference['source_hash'] or
                quality.get('panel_sha256') != ev.digest(self.panel.read_bytes()) or
                quality.get('execution_binding') != self.evaluator.execution_binding or
                quality.get('mean') != reference.get('mean') or not _number(reference['mean']) or reference['mean'] <= 0):
            raise RuntimeError('FRESH_REFERENCE_QUALITY_OR_EXECUTION_CONFLICT')
        if not isinstance(reference.get('cost'), dict):
            raise RuntimeError('FRESH_REFERENCE_COST_MISSING')
        self.reference = dict(reference, source=reference_source, quality=quality)
        self.context = dict(study_id=STUDY_ID, phase=phase, group=group, arm=arm,
            panel_sha256=ev.digest(self.panel.read_bytes()), execution_binding=self.evaluator.execution_binding)
        self.transport = FullTransport(self.root/'transport', self.study/'service', credential_file,
                                       run_id=f'{STUDY_ID}/{phase}/g{group:02d}/{arm}')
        started = self.root/'STARTED.json'
        if not started.exists():
            ev.freeze(started, dict(unix=time.time(), phase=phase, group=group, arm=arm, cpu=str(cpu)))
        self.started = ev.read(started)['unix']
        self.state = _fresh_state(self.context, self.reference)
        self.controllers = {}  # Compatibility slot only; never a controller object.
        latest = self.root/'states/LATEST.json'
        if latest.exists():
            pointer = ev.read(latest)
            checkpoint = _file_within(self.root, pointer['path'])
            if ev.digest(checkpoint.read_bytes()) != pointer['sha256']:
                raise RuntimeError('LATEST_CHECKPOINT_HASH_CONFLICT')
            restored = ev.read(checkpoint)
            if restored.get('context') != self.context or restored.get('reference_hash') != self.reference['source_hash']:
                raise RuntimeError('RESTORED_FRESH_STATE_IDENTITY_CONFLICT')
            self.state = restored
            self.transport.counter = self.state.get('_io_transport_counter', 0)

    def save_program(self, source):
        identity = ev.digest(source)
        ev.freeze_bytes(self.study/'programs'/f'{identity}.py', source.encode())
        return identity

    def evaluate(self, source, role):
        result = self.evaluator.evaluate(source, self.panel, role)
        if result.get('hard_stop'):
            raise RuntimeError('FULL_EVALUATION_INFRASTRUCTURE_CONFLICT: '+str(result.get('infrastructure_error')))
        if result.get('complete'):
            self.state['qualities'][ev.digest(source)] = result
        return result

    def checkpoint(self, state, tag=None):
        if tag is None:
            tag, state = state, self.state  # Frozen native BaselineIO signature.
        if not isinstance(tag, str) or Path(tag).name != tag or tag in ('.', '..', 'LATEST'):
            raise ValueError('UNSAFE_CHECKPOINT_TAG')
        state['_io_transport_counter'] = self.transport.counter
        path = self.root/'states'/f'{tag}.json'
        ev.freeze(path, state)
        pointer = dict(path=str(path.relative_to(self.root)), sha256=ev.digest(path.read_bytes()), tag=tag)
        target = self.root/'states/LATEST.json'
        descriptor, temporary = tempfile.mkstemp(prefix='.latest-', dir=target.parent)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(ev.encoded(pointer)+b'\n');handle.flush();os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
        self.state = state
        return state

    def rng(self, stream, position):
        return seed_for(self.phase, TASK, self.group, self.arm, f'{position}/{stream}')

    def own_cost_snapshot(self):
        return physical_cost_snapshot(self.root, started=self.started)

    def native_messages(self, messages):
        return self.transport.native_messages(messages)

    def cost_snapshot(self):
        return strategy_cost(self)

    def replay_native_from_start(self):
        self.state = _fresh_state(self.context, self.reference)
        self.transport.counter = 0
        self.transport.records.clear()
        self.transport.logical_records.clear()
