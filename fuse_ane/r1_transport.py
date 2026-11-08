"""ANE-r1 bounded transport amendment, keeping each physical attempt immutable."""
import copy
import json
import time
from pathlib import Path
from fuse_addon import evaluation as ev, transport as tr
from fuse_paired.full_runtime import flatten_native_messages


def retryable(folder, result):
    if result.get('fatal_reason') or result.get('service_success'):
        return False
    if result.get('http_status') in (429, 502, 503, 504):
        return True
    if result.get('http_status') == 400:return False
    p = Path(folder)/'sender.error.json'
    error = ev.read(p) if p.exists() else {}
    return (error.get('exception_type') in ('TimeoutError', 'ConnectionResetError', 'RemoteDisconnected')
            or 'connection reset' in str(error.get('message', '')).lower())


def retry_delay(result):
    if result.get('http_status') != 429:
        return 5.
    value = result.get('retry_after')
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            from email.utils import parsedate_to_datetime
            seconds = parsedate_to_datetime(value).timestamp()-time.time()
        except Exception:
            seconds = 5.
    return min(60., max(5., seconds))


def reconcile(folder, identity, logical):
    """Read-only sender reconciliation. The original terminal is never replaced."""
    folder = Path(folder)
    tr.settle_existing(folder)
    original = tr.read(folder/'terminal.json')
    if original.get('identity') != identity or original.get('generation_id') != logical:
        raise RuntimeError('REPLAY_ATTEMPT_IDENTITY_CONFLICT:'+str(folder))
    if original.get('service_success'):
        return original
    settled = tr.classify(folder)
    if settled.get('service_success'):
        resolved=dict(original, **settled, reconciled_from_original_terminal=True)
        ev.freeze(folder/'RECONCILIATION.json',dict(original_terminal_sha256=ev.digest((folder/'terminal.json').read_bytes()),
            identity=identity,generation_id=logical,effective=resolved,physical_attempt_not_repeated=True))
        return resolved
    return original


def effective_attempt(folder):
    """Resolve one physical receipt, retaining its original error/UNKNOWN bytes."""
    folder=Path(folder);original=ev.read(folder/'terminal.json')
    if not (folder/'RECONCILIATION.json').exists():return original
    receipt=ev.read(folder/'RECONCILIATION.json')
    if (receipt['original_terminal_sha256']!=ev.digest((folder/'terminal.json').read_bytes()) or
            receipt['identity']!=original['identity'] or receipt['generation_id']!=original['generation_id']):
        raise RuntimeError('RECONCILED_RECEIPT_IDENTITY_CONFLICT')
    return dict(receipt['effective'],original_status=original['status'],original_service_success=original['service_success'],
        original_usage=original.get('usage'),reconciliation_receipt=str(folder/'RECONCILIATION.json'))


def cost_frontier(io):
    from fuse_patch import runtime as base
    result=base.physical_cost_frontier(io.root,io.transport.counter,list(io._eval_roles.values()),started=io.started)
    recovered=0
    for index in range(io.transport.counter):
        folder=io.root/'transport/requests'/f'{index:06d}'/'attempts'
        for p in sorted(folder.glob('*/RECONCILIATION.json')):
            original=ev.read(p.parent/'terminal.json');current=effective_attempt(p.parent)
            old=original.get('usage') or {};new=current.get('usage') or {}
            for field,prefix in [('total_tokens','tokens'),('prompt_tokens','input_tokens'),('completion_tokens','output_tokens')]:
                before=old.get(field);after=new.get(field)
                if base._number(before) and base._number(after) and before!=after:raise RuntimeError('RECONCILED_USAGE_CONFLICT')
                if not base._number(before) and base._number(after):
                    result[prefix+'_known']+=after;result[prefix+'_missing']-=1
            if current.get('service_success') and not original.get('service_success'):
                result['response_received']+=1;recovered+=1
    result['tokens']=result['tokens_known'] if not result['tokens_missing'] else None
    result['reconciled_physical_attempts']=recovered
    return result


def attempt_request(transport, folder, body, frozen, identity, logical, metadata, attempt):
    from . import runtime as rt
    folder = Path(folder)
    logical_attempt = logical+f'/attempt-{attempt:02d}'
    tr.write(folder/'FROZEN_INPUT.json', frozen)
    tr.write_bytes(folder/'payload.json', body)
    if (folder/'terminal.json').exists():
        return reconcile(folder, identity, logical)
    if (folder/'intent.json').exists():
        if ev.read(folder/'intent.json')['identity'] != identity:
            raise RuntimeError('REPLAY_INTENT_IDENTITY_CONFLICT:'+str(folder))
        tr.settle_existing(folder)
    else:
        # Atomically bind a scientific intent before the sender. DRAINING blocks
        # further intents but cannot interrupt a request already dispatched.
        with ev.exclusive(rt.STUDY/'service/dispatch.lock'):
            rt.require_open(folder)
            tr.write(folder/'intent.json', dict(identity=identity,submitted_unix=time.time(),
                logical_id=logical_attempt,generation_id=logical,submission=1,
                transport_attempt=attempt,timeout_seconds=300,retries=attempt-1,metadata=metadata))
        transport._send_process(folder, identity)
        tr.settle_existing(folder)
    result = tr.classify(folder)
    intent = tr.read(folder/'intent.json')
    result.update(identity=identity, logical_id=logical_attempt,generation_id=logical,
        transport_attempt=attempt,request_path=str(folder),submitted_unix=intent['submitted_unix'],metadata=metadata,
        network_boundary_entered=(folder/'NETWORK_BOUNDARY_INTENT.json').exists(),known_usage=result.get('usage') is not None)
    tr.write(folder/'terminal.json',result)
    return result


def call(transport, prompt, metadata):
    from . import runtime as rt
    with transport.guard, ev.exclusive(transport.root/'.replay-transport.lock'):
        rt.require_open(transport.root)
        messages, transport._message_override = transport._message_override, None
        if isinstance(prompt,list):
            if messages is not None:
                raise RuntimeError('TWO_MESSAGE_SOURCES')
            messages=copy.deepcopy(prompt);prompt=flatten_native_messages(messages)
        if messages is not None and prompt!=flatten_native_messages(messages):
            raise RuntimeError('NATIVE_MESSAGE_PROVENANCE_CONFLICT')
        number=transport.counter;transport.counter+=1
        folder=transport.root/'requests'/f'{number:06d}'
        metadata=copy.deepcopy(metadata)
        body_object=transport.request_payload(prompt,metadata)
        frozen=dict(prompt=prompt,metadata=metadata,request_index=number,run_id=transport.run_id)
        if messages is not None:
            body_object['messages']=messages;frozen['native_messages']=messages
        body=tr.canonical(body_object).encode()
        identity=dict(payload_sha256=tr.sha(body),frozen_sha256=tr.sha(tr.canonical(frozen)))
        # Imported bytes are compared before consumption. No key replacement,
        # prompt normalization, or restart at a fresh RNG coordinate is allowed.
        for name,value in [('payload.json',body),('FROZEN_INPUT.json',None)]:
            p=folder/name
            if p.exists() and ((name=='payload.json' and p.read_bytes()!=value) or
                              (name=='FROZEN_INPUT.json' and ev.read(p)!=frozen)):
                ev.freeze(transport.root/'REPLAY_CONFLICT.json',dict(coordinate=str(folder),field=name,
                    expected_sha256=ev.digest(p.read_bytes()),actual_sha256=ev.digest(value or ev.encoded(frozen))))
                raise RuntimeError('REPLAY_PAYLOAD_CONFLICT:'+str(folder)+':'+name)
        tr.write(folder/'FROZEN_INPUT.json',frozen);tr.write_bytes(folder/'payload.json',body)
        logical=transport.run_id+f'/{number:06d}'
        imported=(folder/'REUSED_REQUEST.json').exists()
        if imported:raise RuntimeError('NO_HISTORICAL_REQUEST_IMPORT')
        # A selected response or resolution lock is authoritative on replay;
        # late attempt01 output may be archived but can never replace attempt02.
        if (folder/'terminal.json').exists():
            terminal=ev.read(folder/'terminal.json')
            if terminal.get('identity')!=identity or terminal.get('logical_id')!=logical or not terminal.get('selected_before_candidate_parse'):
                raise RuntimeError('SELECTED_RESPONSE_REPLAY_CONFLICT:'+str(folder))
            if (folder/'RESOLUTION_LOCK.json').exists() and terminal.get('transport_attempt')!=2:
                raise RuntimeError('RETRY_SELECTION_LOCK_CONFLICT:'+str(folder))
            transport._verify_model(terminal,metadata,folder)
            for p in terminal['attempts']:
                transport.records.append(effective_attempt(p))
            transport.logical_records.append(terminal)
            return terminal['content']
        attempted=[];selected=None
        locked=ev.read(folder/'RESOLUTION_LOCK.json') if (folder/'RESOLUTION_LOCK.json').exists() else None
        if locked:
            first=folder/'attempts/01'
            if locked['payload_sha256']!=identity['payload_sha256'] or locked['attempt01_terminal_sha256']!=ev.digest((first/'terminal.json').read_bytes()):
                raise RuntimeError('RESOLUTION_LOCK_BYTES_CONFLICT')
            attempted.append(str(first));transport.records.append(effective_attempt(first))
            time.sleep(max(0,locked['retry_not_before_unix']-time.time()))
        for attempt in ((2,) if locked else (1,2)):
            attempt_dir=folder/'attempts'/f'{attempt:02d}'
            result=transport._attempt(attempt_dir,body,frozen,identity,logical,metadata,attempt)
            attempted.append(str(attempt_dir));transport.records.append(result)
            # Success is locked before parsing/executing a candidate. Usage may
            # remain null; lack of usage never licenses an additional request.
            if result.get('service_success'):
                transport._verify_model(dict(result,logical_id=logical),metadata,folder)
                selected=result;break
            # Reconcile saved bytes immediately before deciding on another HTTP.
            if attempt==1 and retryable(attempt_dir,result):
                reconciled=reconcile(attempt_dir,identity,logical)
                if reconciled.get('service_success'):
                    transport._verify_model(dict(reconciled,logical_id=logical),metadata,folder)
                    selected=reconciled;break
            boundary=getattr(transport,'_budget_boundary',None)
            if boundary is not None:
                boundary.settled_failed_attempt(result)
                if attempt==1 and retryable(attempt_dir,result) and not boundary.retry_budget_available():
                    selected=dict(result,budget_stopped_before_retry=True)
                    break
            if attempt==2 or not retryable(attempt_dir,result):
                reason=result.get('fatal_reason') or 'UNRECOVERABLE_MODEL_TRANSPORT'
                rt.stop_dispatch(reason)
                raise tr.ServiceStopped(reason+':'+str(attempt_dir))
            delay=retry_delay(result)
            ev.freeze(folder/'RESOLUTION_LOCK.json',dict(logical_id=logical,attempt01_status=result['status'],
                attempt01_terminal_sha256=ev.digest((attempt_dir/'terminal.json').read_bytes()),
                reconciled_complete_success=False,selected_next_attempt=2,payload_sha256=identity['payload_sha256'],
                retry_not_before_unix=time.time()+delay,late_attempt01_response_excluded=True,unknown_cost_not_zero=True))
            time.sleep(delay)
        terminal=dict(selected,logical_id=logical,attempts=attempted,attempt_count=len(attempted),
            unknown_replayed=len(attempted)>1,policy_revision='FUSE_SIBLING1',selected_before_candidate_parse=True)
        tr.write(folder/'terminal.json',terminal)
        ev.freeze(folder/'REPLAY_RECEIPT.json',dict(imported=imported,payload_sha256=identity['payload_sha256'],
            frozen_sha256=identity['frozen_sha256'],new_attempts=sum(not (Path(p)/'REUSED_ATTEMPT.json').exists() for p in attempted),
            selected_attempt=selected['transport_attempt'],same_original_rng_and_logical_coordinate=True))
        transport.logical_records.append(terminal)
        return terminal['content']
