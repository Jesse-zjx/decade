"""Fixed shared-parent proposals; one atomic settled step per generator yield.

Provider and evaluator calls are only the existing guarded SearchIO seams. Pure
protocol/assembly modules do not execute candidates. A physical bill belongs to
one request; every slot references it, never copies it into an additive cost.
"""
import copy
from fuse_addon.evaluation import freeze, digest, encoded
from fuse_patch.budget import DispatchStop
from .budget import Boundary, token_count
from .regions import assemble
from . import edit_evidence as evidence
from . import sibling_protocol as codec, sibling_view as view
from .policy import real_error, technical


def _reservation(io, boundary, width, repairs):
    boundary.before_dispatch(generation=True)
    if boundary.position + width > 256:
        raise DispatchStop('NORMAL_SLOT_CAP')
    # Reserve every initial attempt plus the one permitted repair before sending.
    if io.state['feedback_attempts'] + width + 1 > 384:
        raise DispatchStop('FEEDBACK_RESERVATION_CAP')
    if repairs > 128:
        raise RuntimeError('REPAIR_CAP_VIOLATION')


def _send(io, pre, folder, *, repair_slot=None, failed=None):
    messages = view.messages(io, pre, failed_slot=repair_slot,
        error=failed.get('error') if failed else None,
        failed_output=failed.get('raw_output') if failed else None)
    index = io.transport.counter
    kind = 'repair' if repair_slot else 'sibling_batch' if len(pre['slot_order']) == 2 else 'single'
    freeze(folder/'REQUEST_VIEW.json', dict(pre=pre, request_index=index,
        messages_sha256=digest(encoded(messages)), kind=kind))
    response = io.transport(messages, dict(kind=kind, arm=io.arm,
        normal_slots=pre['normal_slot_numbers'], slot_order=pre['slot_order'],
        actual_parent_sha=pre['parent_sha'], pre_sha256=digest(encoded(pre)),
        max_tokens=16384 if kind=='sibling_batch' else 8192,
        effect_visible=io.arm!='N2', selector_extra_http=0))
    terminal = io.transport.logical_records[-1]
    request_id = terminal['logical_id']
    decoded = codec.decode_response(response, pre, request_id)
    freeze(folder/'PARSE.json', decoded)  # Immutable before any candidate work.
    plans = codec.assemble_plans(decoded, pre['parent_source'], assemble)
    freeze(folder/'ASSEMBLY.json', plans)
    return plans, index, request_id


def _evaluate_slots(io, pre, plans, index, request_id, folder, *, repair=False):
    """Evaluate independently in frozen order; delay all archive admission.

    The evaluator may cache the exact observation, including failure, of a prior
    identical sibling. It never labels that reuse as a new paired measurement.
    """
    pending, observations, records = [], {}, []
    parent_quality = copy.deepcopy(io.state['qualities'].get(pre['parent_sha']))
    terminal=io.transport.logical_records[-1]
    attempt_number=terminal['transport_attempt']
    physical_id=request_id+f'/attempt-{attempt_number:02d}'
    physical_path=terminal.get('request_path') or terminal['attempts'][attempt_number-1]
    for slot in plans['slots']:
        sid = slot['slot_id']; number = slot['normal_index']
        local = pre['slots'][sid]; child = slot.get('child_source')
        h = slot.get('child_sha'); q = None
        duplicate = slot['status'].startswith('DUPLICATE_')
        if repair and h is not None and h in io.state['programs']:
            duplicate=True
            slot.update(status='DUPLICATE_BATCH_SETTLED_ARCHIVE',duplicate_of=h)
        if child is not None and getattr(io, 'evaluation_exhausted', lambda: False)():
            io.retain_unevaluated(child, f'slot-{number:03d}-attempt-{int(repair)}', slot)
            slot.update(status='GENERATED_NOT_EVALUATED_BUDGET_LIMIT', quality=None,
                repairable=False, valid=False, evidence_type=None, target_credit=None,
                request_index=index, batch_id=pre['batch_id'], repair=repair,
                selected_physical_attempt_id=physical_id, selected_physical_attempt_path=physical_path)
            freeze(folder/f'{sid}_ATTEMPT.json',slot)
            continue
        if child is not None:
            io.save_program(child)
            if h in observations:
                q = copy.deepcopy(observations[h])
                slot['observation_reuse'] = 'EARLIER_SIBLING_SAME_SOURCE'
            elif duplicate and h in io.state['qualities']:
                q = copy.deepcopy(io.state['qualities'][h])
                slot['observation_reuse'] = 'EXACT_PRIOR_SOURCE_SAME_FEEDBACK'
            else:
                q = io.evaluate(child, f'slot-{number:03d}-attempt-{int(repair)}')
                slot['observation_reuse'] = None
            observations[h] = copy.deepcopy(q)
            slot['quality'] = q
            if q.get('complete'):
                pending.append((slot, q))
                slot['error'] = None; slot['repairable'] = False
            else:
                slot['error'] = real_error(q) or {'type':q.get('status','EVALUATION_FAILURE')}
                slot['repairable'] = not duplicate and technical(slot['error'])
        error = slot.get('error')
        if isinstance(error,str):error=dict(type='FORMAT_OR_ASSEMBLY_ERROR',message=error)
        slot['error'] = error
        unit = slot.get('unit') or local['menu']['GLOBAL']
        support = slot.get('support') or {}
        record = evidence.record(parent=pre['parent_source'], child=child, region=unit,
            parent_quality=parent_quality, child_quality=q, context=io.context,
            episode=number, attempt=int(repair), mode=support.get('mode'),
            novel=bool(child is not None and not duplicate), error=error,
            scale=io.reference['mean'], elite_before=pre['quality_before']['elite'],
            cost=dict(shared_request_id=request_id, slot_allocation=None,
                      physical_charge_owner='REQUEST_LEDGER', repair=repair),
            request_index=index)
        record.update(batch_id=pre['batch_id'], slot_id=sid,
            selected_physical_attempt_id=physical_id,selected_physical_attempt_path=physical_path,
            proposal_id=slot['proposal_id'], request_id=request_id,
            actual_parent_sha=pre['parent_sha'], request_owner=io.arm,
            normal_index=number, repair=repair,
            independent_random_sample=False,
            changed_byte_ratio=support.get('changed_byte_ratio'),
            public_entry_function='select_next_node',
            support_actual_scope=support.get('realized_scope'))
        slot.update(evidence_id=record['evidence_id'], evidence_type=record['type'],
            new_source=bool(child is not None and not duplicate), repair=repair,
            valid=bool(q and q.get('complete')), request_index=index,
            batch_id=pre['batch_id'],selected_physical_attempt_id=physical_id,
            selected_physical_attempt_path=physical_path)
        if not slot['valid']:
            io.state['failures'].append(dict(proposal_id=slot['proposal_id'], error=error,
                quality=None, repair=repair, normal_index=number))
        freeze(folder/f'{sid}_ATTEMPT.json',slot)
        evidence.persist(folder/f'{sid}_EVIDENCE.json',record)
        records.append(record)
    # Both initial siblings have settled before either is admitted. The exact
    # crossing-request snapshot is made immediately after this admission, before
    # any later repair. Records enter future proposal memory only after the step.
    for slot, q in pending:
        sid=slot['slot_id']; local=pre['slots'][sid]
        parents=[pre['parent_sha']]+([local['donor_sha']] if local.get('donor_sha') else [])
        h=io.register_program(slot['child_source'],q,parents,'FUSE_SIBLING_'+io.arm,
            slot['normal_index'],slot.get('proposal',{}).get('intent',''))
        io.state.setdefault('acquisition_request',{}).setdefault(h,request_id)
    return plans['slots'], records


def run_steps(io, shared):
    io.attach_shared(shared)
    # Original INIT cost/state remain fully attached; the new arm's finite
    # candidate-attempt allowance explicitly excludes the reused INIT attempts.
    io.state['initialization_feedback_attempts']=io.state['feedback_attempts']
    io.state['feedback_attempts']=0
    boundary=Boundary(io);boundary.ensure_reference();boundary.inherit_prefixes(shared)
    memory=[]; repairs=0; pair_used={}; status='NORMAL_SLOT_CAP'; width=1 if io.arm=='F1' else 2
    try:
        while boundary.position<256:
            _reservation(io,boundary,width,repairs)
            numbers=list(range(boundary.position+1,boundary.position+width+1))
            pair=(numbers[0]-1)//2
            pre=view.prepare(io,memory,numbers)
            step=io.root/'batches'/f'{numbers[0]:03d}'
            freeze(step/'PRE.json',pre)
            plans,index,rid=_send(io,pre,step/'initial')
            slots, records=_evaluate_slots(io,pre,plans,index,rid,step/'initial')
            boundary.position=numbers[-1]
            boundary.observe('settled_request',request_id=rid,request_index=index,
                slot_ids=[s['proposal_id'] for s in slots],repair=False)
            eligible=next((s for s in slots if s.get('repairable') and not s.get('status','').startswith('DUPLICATE_')),None)
            repaired=None
            if eligible and not pair_used.get(pair) and repairs<128:
                try:boundary.before_dispatch(generation=True,repair=True)
                except DispatchStop:pass
                else:
                    if io.state['feedback_attempts']<384:
                        rpre=view.prepare_repair(pre,eligible['slot_id'])
                        rplans,rix,rrid=_send(io,rpre,step/'repair',repair_slot=eligible['slot_id'],failed=eligible)
                        pair_used[pair]=True;repairs+=1
                        rslots,rrecords=_evaluate_slots(io,rpre,rplans,rix,rrid,step/'repair',repair=True)
                        records.extend(rrecords);repaired=rslots[0]['proposal_id']
                        boundary.observe('settled_request',request_id=rrid,request_index=rix,
                            slot_ids=[s['proposal_id'] for s in rslots],repair=True)
            memory.extend(records)
            result=dict(normal_slots=numbers,pair_index=pair,initial_request_id=rid,
                shared_parent=pre['parent_sha'],slot_ids=[s['proposal_id'] for s in slots],
                repaired_proposal_id=repaired,repair_allowance_used=bool(pair_used.get(pair)),
                memory_evidence_ids=[r['evidence_id'] for r in memory],
                all_children_retained=True,fully_settled=True)
            freeze(step/'RESULT.json',result)
            io.checkpoint(f'SETTLED-{numbers[-1]:03d}')
            yield result
    except DispatchStop as exc:status=exc.reason
    return boundary.finish([],dict(edit_evidence_ids=[r['evidence_id'] for r in memory],
        repair_count=repairs,empty_initial_memory=True),status=status)


def run(io, shared):
    steps=run_steps(io,shared)
    while True:
        try:next(steps)
        except StopIteration as stop:return stop.value
