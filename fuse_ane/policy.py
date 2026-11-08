"""RNE reuses the proposal / one repair / complete evaluation episode pipeline."""
import json
from dataclasses import asdict
from fuse_addon.evaluation import freeze,freeze_bytes,digest,encoded
from fuse_patch.budget import DispatchStop
from .budget import token_count,Boundary
from .edit_envelope import decode,canonical_for_legacy
from .global_envelope import decode_global
from .core import progress_reward
from .regions import assemble,EditError
from . import edit_evidence as evidence
from .memory_view import prepare
from .joint_edit import decode as joint_decode,prompt as joint_prompt
from .reactive_policy import FocusChain
from .reactive_core import Observation

def real_error(quality):
    if not quality:return None
    row=next((x for x in quality['rows'] if x['status'] not in ('EVALUATED','NOT_EXECUTED')),None)
    if row is None:return None
    error={k:row.get(k) for k in ('instance_id','status','error_type','error','candidate_error','diagnostic','stderr_tail') if row.get(k) is not None}
    for key in ('diagnostics','stderr'):
        if row.get(key):error[key]=str(row[key])[:800]
    return error

def technical(error):
    if not error:return False
    text=json.dumps(error).upper()
    if any(x in text for x in ('TIMEOUT','DEADLINE','UNKNOWN','HARD_STOP')):return False
    return any(x in text for x in ('FORMAT_','SYNTAX','ASSEMBLY_','NAMEERROR','TYPE','EXCEPTION','PROTOCOL_FAILURE','CANDIDATE_EXITED','INVALID','RETURN_','ERROR'))

def run(io,shared):
    io.attach_shared(shared);boundary=Boundary(io);boundary.ensure_reference();boundary.inherit_prefixes(shared)
    records=[];recent=[];repair_tokens=0;status='NORMAL_EPISODE_CAP'
    try:
        for index in range(1,129):
            boundary.before_dispatch(generation=True)
            pre=prepare(io,records,index);h=pre['parent_sha'];folder=io.root/'episodes'/f'{index:03d}'
            parent_loss=io.state['programs'][h]['mean'];incumbent=min(x['mean'] for x in io.state['programs'].values())
            before=io.cost_snapshot();own_before=io.own_cost_snapshot();messages=joint_prompt(io,pre,recent)
            pre.update(payload_messages_sha256=digest(encoded(messages)),programs_before={h:dict(mean=v['mean'],first_index=v['first_index'],parents=v.get('parents',[])) for h,v in io.state['programs'].items()})
            freeze(folder/'PRE.json',pre)
            attempts=[];source=quality=support=decoded=None;new_source=False;error=None;selected=None
            for attempt in (0,1):
                if attempt:
                    if not technical(error) or repair_tokens>=16000 or (source is not None and not new_source):break
                    try:boundary.before_dispatch(generation=True,repair=True)
                    except DispatchStop:break
                    import copy
                    repair_pre=copy.deepcopy(pre);repair_pre['host_full']=True
                    messages=joint_prompt(io,repair_pre,recent,error=error,failed_source=source if source is not None else response)
                else:repair_pre=pre
                request_archive=set(io.state['programs']);request_index=io.transport.counter;start_tokens=token_count(io.cost_snapshot())
                freeze(folder/f'REQUEST_VIEW_{attempt}.json',dict(pre=repair_pre,payload_messages_sha256=digest(encoded(messages)),request_index=request_index))
                response=io.transport(messages,dict(kind='repair' if attempt else 'generation',episode=index,attempt=attempt,
                    pre_sha256=digest(encoded(pre)),requested_scope=pre['effective_scope'],operator=pre['operator'],host_full=repair_pre['host_full'],
                    effect_bound=io.arm=='F' and bool(pre['memory']['cases']),memory_cases=[c['evidence_id'] for c in pre['memory']['cases']],max_tokens=8192))
                if attempt:repair_tokens+=token_count(io.cost_snapshot())-start_tokens
                source=quality=support=decoded=None;selected=None;new_source=False;error=None
                try:
                    decoded=joint_decode(response,repair_pre);selected=pre['menu'][decoded['target']]
                    # This response and the exact chosen target are immutable BEFORE
                    # any compilation/evaluation. stream=false: no separate host
                    # receipt before the code was generated is claimed.
                    freeze(folder/f'SELECTION_{attempt}.json',dict(decoded=decoded,selected_record=selected,request_index=request_index,
                        response_frozen_before_evaluation=True,selector_extra_http=0))
                    source,support=assemble(io.source(h),selected,'# FUSE_'+decoded['mode']+'\n'+decoded['code'])
                    support.update(codec_revision='FUSE_CEE1',host_full=repair_pre['host_full'],actual_parent_sha=h,
                        target_id=decoded['target'],cited_evidence_ids=decoded['evidence'],request_index=request_index)
                except (EditError,SyntaxError,ValueError) as exc:error=dict(type=type(exc).__name__,message=str(exc))
                if error is None:
                    source_hash=io.save_program(source);new_source=source_hash not in request_archive
                    quality=io.evaluate(source,f'episode-{index:03d}-attempt-{attempt}')
                    if quality['complete']:io.register_program(source,quality,[h]+([pre['donor_sha']] if pre.get('donor_sha') else []),'FUSE_CEE_'+io.arm,index)
                    else:error=real_error(quality)
                valid=bool(quality and quality['complete'])
                record=evidence.record(parent=io.source(h),child=source,region=selected or pre['menu']['GLOBAL'],
                    parent_quality=io.state['qualities'].get(h),child_quality=quality,context=io.context,episode=index,attempt=attempt,
                    mode=(support or {}).get('mode'),novel=new_source,error=error,scale=io.reference['mean'],elite_before=incumbent,
                    cost=dict(request_tokens_known=token_count(io.cost_snapshot())-start_tokens,frontier=io.cost_snapshot()),request_index=request_index)
                evidence.persist(folder/f'EDIT_EVIDENCE_{attempt}.json',record);records.append(record)
                ar=dict(attempt=attempt,request_index=request_index,valid=valid,source_hash=digest(source) if source is not None else None,
                    codec_revision='FUSE_CEE1',host_full=repair_pre['host_full'],decoding=decoded,support=support,error=error,
                    quality_mean=quality.get('mean') if quality else None,quality=quality,repair=bool(attempt),new_source=new_source,
                    evidence_type=record['type'],evidence_id=record['evidence_id'],request_tokens=token_count(io.cost_snapshot())-start_tokens)
                freeze(folder/f'ATTEMPT_{attempt}.json',ar);attempts.append(ar)
                boundary.position=index;boundary.observe('settled_episode_attempt',episode=index,attempt=attempt,candidate_valid=valid)
                if valid or not technical(error):break
            cost=io.cost_snapshot();own_after=io.own_cost_snapshot();valid=bool(quality and quality['complete'])
            _,gp,ge=progress_reward(parent_loss,incumbent,quality['mean'] if valid else None,io.reference['mean'],complete_valid=valid,new_source=new_source)
            tokens=token_count(cost)-token_count(before);missing=cost.get('tokens_missing',0)-before.get('tokens_missing',0)
            action=dict(parent_sha=h,scope=(selected or pre['menu']['GLOBAL'])['scope'],region_sha=(selected or pre['menu']['GLOBAL'])['region_sha'])
            row=dict(episode=index,action=action,operator=pre['operator'],attempts=len(attempts),repaired=len(attempts)>1,valid=valid,
                first_attempt_valid=attempts[0]['valid'],source_hash=digest(source) if source is not None else None,quality_mean=quality['mean'] if valid else None,
                new_source=new_source,duplicate=bool(source is not None and not new_source),g_parent=gp,g_elite=ge,
                elite_update=bool(valid and quality['mean']<incumbent),material_progress=bool(valid and new_source and ge>=.01),
                support=support,error=error,tokens=tokens if not missing else None,tokens_known=tokens,tokens_missing=missing,
                started_feedback_instances=own_after['physical_instances']-own_before['physical_instances'],cost=cost,repair_tokens_total=repair_tokens)
            freeze(folder/'RESULT.json',row)
            if not valid:io.state['failures'].append(dict(episode=index,error=error,quality=None))
            recent.append(json.dumps(dict(requested=pre['effective_scope'],realized=(support or {}).get('realized_scope'),valid=valid,duplicate=row['duplicate'],
                g_parent=gp,g_elite=ge,error_type=(error or {}).get('type',(error or {}).get('error_type'))),ensure_ascii=False,separators=(',',':'))[:250])
    except DispatchStop as exc:status=exc.reason
    return boundary.finish([],dict(edit_evidence_ids=[e['evidence_id'] for e in records],memory_from_this_arm_only=True,repair_tokens=repair_tokens),status=status)
