"""One sequential search; all candidate execution and requests use remote adapters."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from fuse_addon.evaluation import freeze, freeze_bytes, read, digest, encoded
from .runtime import CreditEvaluator, CreditTransport, seed_for, STUDY_ID, payload_utf8_bytes
from .controller import (RidgeController, build_slate, make_features, policy_credit_maps,
                         credit_key)
from .generation import init_prompt, render_prompt, consume_response
from .state import (fresh_state, register_program, active_pool, cf_candidate,
                    matched_credit, admit_natural_credit, REFERENCE_SOURCE)


def cost_snapshot(root):
    root = Path(root)
    requests = [read(p) for p in sorted((root/'transport'/'requests').glob('*/attempts/*/terminal.json'))]
    physical = [read(p) for p in sorted((root/'evaluator'/'physical_evaluations').glob('*/EVALUATION.json'))]
    roles = [read(p) for p in sorted((root/'evaluator'/'evaluations').glob('**/EVALUATION.json'))]
    tokens = 0
    missing = 0
    inputs = outputs = 0
    for x in requests:
        u = x.get('usage')
        if not isinstance(u, dict) or not isinstance(u.get('total_tokens'), (int,float)):
            missing += 1
        else:
            tokens += u['total_tokens']
        if isinstance(u,dict):
            inputs += u.get('prompt_tokens') or 0
            outputs += u.get('completion_tokens') or 0
    return dict(http=len(requests), response_received=sum(x.get('status')=='RESPONSE_RECEIVED' for x in requests),
                unknown=sum(x.get('status')=='UNKNOWN' for x in requests),
                tokens_known=tokens, tokens_missing=missing, tokens=tokens if not missing else None,
                input_tokens_known=inputs, output_tokens_known=outputs,
                llm_wall_seconds_known=sum(read(Path(x['request_path'])/'process.json').get('wall_seconds') or 0 for x in requests if (Path(x['request_path'])/'process.json').exists()),
                evaluation_roles=len(roles), physical_evaluations=len(physical),
                physical_instances=sum(x.get('physical_started',0) for x in physical),
                physical_unknown=sum(bool(x.get('physical_started_uncertain')) for x in physical),
                evaluator_active_seconds=sum(x.get('active_seconds') or 0 for x in physical),
                evaluator_queue_seconds=sum(x.get('cpu_slot_wait_seconds') or 0 for x in physical),
                evaluator_cpu_seconds=sum(x.get('child_cpu_user_seconds_known',0)+x.get('child_cpu_system_seconds_known',0) for x in physical),
                evaluation_failures=sum(not x['complete'] for x in physical))


def add_costs(a,b):
    out={}
    for k in set(a)|set(b):
        av,bv=a.get(k,0),b.get(k,0)
        out[k]=None if av is None or bv is None else av+bv
    return out


class Search:
    def __init__(self, study_root, phase, group, arm, cpu, credential_file,
                 *, common_state=None, common_result=None):
        self.study = Path(study_root).resolve()
        self.phase, self.group, self.arm = phase, int(group), arm
        self.root = self.study/'searches'/phase/f'g{self.group:02d}'/arm
        self.root.mkdir(parents=True, exist_ok=True)
        self.panel = self.study/'panels'/phase/f'g{self.group:02d}'/'Search32.json'
        if not self.panel.is_file():
            raise RuntimeError('SEARCH_PANEL_NOT_FROZEN')
        self.evaluator = CreditEvaluator(self.root/'evaluator', cpu,
            dict(cpu_slot_lock_dir=str(self.study/'cpu-slots'), runtime_identity=read(self.study/'engineering'/'RUNTIME_IDENTITY.json')))
        self.transport = CreditTransport(self.root/'transport', self.study/'service',
            credential_file, run_id=f'{STUDY_ID}/{phase}/g{group:02d}/{arm}')
        context=dict(panel_sha256=digest(self.panel.read_bytes()),
                     execution_binding=self.evaluator.execution_binding)
        self.state=copy.deepcopy(common_state) if common_state is not None else fresh_state(context)
        if self.state['context'] != context:
            raise RuntimeError('COMMON_BRANCH_EXECUTION_CONTEXT_CONFLICT')
        self.common_result=common_result
        self.controllers={a: RidgeController.from_dict(s) for a,s in self.state['controllers'].items()}
        if not self.controllers:
            self.controllers={a:RidgeController(a) for a in ('C','U','S')}
        if arm in ('C','U','S'):
            self.controllers={arm:self.controllers[arm]}
        elif arm=='R':
            self.controllers={}
        self.random_controller=RidgeController('R')
        started=self.root/'STARTED.json'
        if not started.exists():freeze(started,dict(unix=time.time(),cpu=str(cpu),arm=arm))
        self.started=read(started)['unix']

    def rng(self, stream, position):
        # Shared numbered streams across actual branch strategies; never a model
        # token pairing claim. Permutation is a separate stream.
        return seed_for(self.phase,'TSP-Construct',self.group,'controller',f'{position}/{stream}')

    def evaluate(self,source,role):
        h=digest(source)
        if h in self.state['qualities']:
            old=self.state['qualities'][h]
            freeze(self.root/'logical_cache'/f'{role}.json',dict(source_hash=h,
                   physical_record=old.get('physical_record'),panel_sha256=old['panel_sha256'],
                   role=role,complete=old['complete'],source='this_strategy_completed_exact_endpoints',
                   physical_cost=0,logical_instance_responsibility=len(old['rows'])))
            return old
        result=self.evaluator.evaluate(source,self.panel,role)
        if result.get('hard_stop'):
            raise RuntimeError('EVALUATION_INFRASTRUCTURE_CONFLICT: '+str(result.get('infrastructure_error')))
        if result['complete']:
            self.state['qualities'][h]=result
        return result

    def save_program(self,source):
        h=digest(source)
        freeze_bytes(self.study/'programs'/f'{h}.py',source.encode())
        return h

    def initialize(self,count):
        # The required real naive scale and fallback, separate from i1 budget.
        self.save_program(REFERENCE_SOURCE)
        quality=self.evaluate(REFERENCE_SOURCE,'reference')
        if not quality['complete']:raise RuntimeError('NAIVE_REFERENCE_NOT_FEASIBLE')
        self.state['scale']=quality['mean']
        for i in range(count):
            prompt=init_prompt()
            slot=f'init-{i:03d}'
            freeze(self.root/'positions'/slot/'PRE.json',dict(kind='i1',position=i,
                   prompt_sha256=digest(prompt),parents=[],candidate_unknown=True))
            response=self.transport(prompt,dict(kind='initialization',position=i,operator='i1'))
            source=None;error=None;result=None;parsed=None
            if response is not None:
                try:
                    parsed=consume_response(response,dict(op='I1',parents=[],target=None),{})
                    freeze(self.root/'positions'/slot/'PARSE.json',parsed)
                    source=parsed['source'] if parsed['valid'] else None
                    error=parsed.get('error')
                except (ValueError,SyntaxError,TypeError) as exc:
                    error=type(exc).__name__+': '+str(exc)
            if source is not None:
                self.save_program(source)
                result=self.evaluate(source,slot)
                if result['complete']:
                    register_program(self.state,source,result,[],'i1',i+1,parsed['algorithm'])
            valid=bool(result and result['complete'])
            receipt=dict(position=i+1,operator='i1',valid=valid,source_hash=digest(source) if source else None,
                         quality=result['mean'] if valid else None,error=error,
                         response_available=response is not None)
            freeze(self.root/'positions'/slot/'RESULT.json',receipt)
            if not valid:self.state['failures'].append(receipt)
            self.state['init_completed']+=1
        if not self.state['programs']:
            register_program(self.state,REFERENCE_SOURCE,quality,[],'all_initialization_failed_reference',0)
        self.checkpoint('INITIALIZED')

    def checkpoint(self,name):
        self.state['controllers']={a:c.to_dict() for a,c in self.controllers.items()}
        freeze(self.root/'states'/f'{name}.json',self.state)

    def measure_cf(self,pool,t):
        selected=cf_candidate(self.state,pool,self.rng('measurement',t))
        folder=self.root/'cf'/f'before-{t+1:03d}'
        freeze(folder/'SELECTION.json',dict(completed_noninit=t,selected=selected,
               measurement_uses_credit_values=False))
        if selected is None:return
        h=selected['program_hash'];f=selected['factor'];ref=selected['reference']
        source=ref['replacement_source'];rh=self.save_program(source)
        self.state['cf_last_processed'][h]=t
        uncached=rh not in self.state['qualities']
        result=self.evaluate(source,f'cf-before-{t+1:03d}')
        c=None
        if result['complete']:
            register_program(self.state,source,result,[h],'additional_cf',self.state['init_completed']+t)
            c=matched_credit(self.state,h,f,rh,'additional_cf',evidence=ref)
        event=dict(completed_noninit=t,parent_hash=h,reference_hash=rh,complete=result['complete'],
                   credit=c,physical_evaluation_requested=uncached,
                   quality=result['mean'] if result['complete'] else None)
        self.state['cf_events'].append(event)
        freeze(folder/'RESULT.json',event)

    def metadata(self,pool,action,prompt,total_positions):
        p=self.state['programs']
        return dict(scale=self.state['scale'],elite_mean=min(x['mean'] for x in p.values()),
            active_count=len(pool),ranks={x['hash']:i for i,x in enumerate(pool)},
            parent_visits=self.state['parent_visits'],target_visits=self.state['target_visits'],
            payload_utf8_bytes=payload_utf8_bytes(prompt),progress=(self.state['init_completed']+self.state['noninit_completed'])/total_positions,
            op_history={op:dict(attempts=n,valid=self.state['op_valid'].get(op,0)) for op,n in self.state['op_attempts'].items()},
            completion_history=self.state['completion_history'])

    def step(self,total_positions,*,uniform=False):
        t=self.state['noninit_completed'];pos=self.state['init_completed']+t+1
        pool=active_pool(self.state,self.rng('reservoir',t))
        if t and t%4==0:
            self.measure_cf(pool,t)
            pool=active_pool(self.state,self.rng('reservoir-after-cf',t))
        byhash={x['hash']:x for x in pool}
        slate=build_slate(pool,self.rng('partner',t))
        maps=policy_credit_maps(pool,self.rng('permutation',t),self.state['credits'])
        features={a:[] for a in ('C','U','S')}
        prompts=[]
        for action in slate:
            prompt=render_prompt(action,byhash)
            prompts.append(prompt)
            metadata=self.metadata(pool,action,prompt,total_positions)
            for a in features:
                row=make_features(action,byhash,metadata=metadata,credits=maps[a] if a in maps else maps['C'],arm=a)
                if a=='U':row['u']=[0.]*30;row['x']=row['z']+[0.]*30
                features[a].append(row)
        chooser=self.random_controller if self.arm in ('COMMON','R') else self.controllers[self.arm]
        actual_features=features['C' if self.arm in ('COMMON','R') else self.arm]
        decision=chooser.choose(actual_features,main_position=t+1,random_seed=self.rng('decision',t),force_uniform=uniform)
        index=decision['selected_input_index'];action=slate[index];prompt=prompts[index]
        shadow=None
        if self.arm=='C':
            u=RidgeController('U')
            for record in self.state['history']:
                row=copy.deepcopy(record['features']);row['u']=[0.]*30;row['x']=row['z']+[0.]*30
                u.observe(row,record['reward_pp'],record['valid'],record.get('actual_cost'))
            shadow=u.choose(features['U'],main_position=t+1,random_seed=self.rng('decision',t),force_uniform=uniform)
            shadow.update(executed=False,credit_changed_selected_action=shadow['selected_input_index']!=index)
        folder=self.root/'positions'/f'action-{t+1:03d}'
        pre=dict(study_id=STUDY_ID,phase=self.phase,group=self.group,arm=self.arm,position=pos,
            noninit_position=t+1,state_sha256=digest(encoded(self.state)),active_hashes=list(byhash),
            slate=slate,policy_features=features if self.arm=='COMMON' else {self.arm:actual_features},
            permutation=maps.get('audit'),decision=decision,shadow_U=shadow,action=action,
            selected_index=index,prompt_sha256=digest(prompt),candidate_unknown=True,
            rng_seeds={s:self.rng(s,t) for s in ('reservoir','partner','permutation','decision','measurement')})
        freeze(folder/'PRE.json',pre)
        response=self.transport(prompt,dict(kind='generation',position=pos,noninit_position=t+1,
            operator=action['op'],parents=action['parents'],target=action.get('target'),
            predecision_sha256=digest(encoded(pre))))
        # The method labels/credit never enter the generator prompt; metadata is
        # local durable provenance, not part of the API messages.
        op=action['op']
        self.state['op_attempts'][op]=self.state['op_attempts'].get(op,0)+1
        for h in set(action['parents']):
            self.state['parent_visits'][h]=self.state['parent_visits'].get(h,0)+1
        if action.get('target'):
            key=credit_key(action['parents'][0],action['target'])
            self.state['target_visits'][key]=self.state['target_visits'].get(key,0)+1
        source=None;error=None;result=None;parsed=None
        if response is not None:
            try:
                parsed=consume_response(response,action,byhash)
                freeze(folder/'PARSE.json',parsed)
                source=parsed['source'] if parsed['valid'] else None
                error=parsed.get('error')
            except (ValueError,SyntaxError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
        if source is not None:
            self.save_program(source)
            result=self.evaluate(source,f'action-{t+1:03d}')
        valid=bool(result and result['complete'])
        ref=min(self.state['programs'][h]['mean'] for h in action['parents'])
        elite=min(x['mean'] for x in self.state['programs'].values())
        reward=100*(ref-result['mean'])/self.state['scale'] if valid else 0.
        h=digest(source) if source is not None else None
        seen=h in self.state['programs'] if h else False
        credit=None
        if valid:
            register_program(self.state,source,result,action['parents'],op,pos,parsed['algorithm'])
            if True:
                for parent_hash in sorted(set(action['parents'])):
                    credit=admit_natural_credit(self.state,parent_hash,h)
                    if credit is not None:break
            self.state['op_valid'][op]=self.state['op_valid'].get(op,0)+1
        usage=(self.transport.records[-1].get('usage') or {}) if self.transport.records else {}
        logical=self.transport.logical_records[-1] if self.transport.logical_records else {}
        attempt_records=[read(Path(path)/'terminal.json') for path in logical.get('attempts',[])]
        actual_cost=dict(final_response_usage=usage,physical_attempts=[dict(request_path=r['request_path'],status=r['status'],usage=r.get('usage')) for r in attempt_records],
                         tokens_known=sum((r.get('usage') or {}).get('total_tokens',0) for r in attempt_records),
                         usage_missing=sum(r.get('usage') is None for r in attempt_records))
        if isinstance(usage.get('completion_tokens'),(int,float)):
            self.state['completion_history'].setdefault(op,[]).append(usage['completion_tokens'])
            self.state['completion_history'][op]=self.state['completion_history'][op][-32:]
        for a,c in self.controllers.items():
            c.observe(features[a][index],reward,valid,actual_cost)
        history_row=actual_features[index]
        self.state['history'].append(dict(features=history_row,reward_pp=reward,valid=valid,
            actual_cost=actual_cost,position=pos,source_hash=h,action=action,
            arm_features={a:features[a][index] for a in ('C','U','S')} if self.arm=='COMMON' else None))
        receipt=dict(position=pos,noninit_position=t+1,op=op,parents=action['parents'],source_hash=h,
            valid=valid,candidate_quality=result['mean'] if valid else None,
            operational_reward_pp=reward,candidate_reward_pp=reward if valid else None,
            retained_parent_on_failure=not valid,elite_gain_pp=100*(elite-result['mean'])/self.state['scale'] if valid else None,
            restores_seen_program=seen,new_source=bool(h and not seen),error=error,
            reused_archived_source_evaluation=seen, new_occurrence_lineage_used_for_decisions=valid,
            occurrence_id=self.state['programs'][h]['occurrence_id'] if valid else None,
            response_available=response is not None,natural_credit=credit,
            credit_changed_actual_action=bool(shadow and shadow['credit_changed_selected_action']),
            actual_cost=actual_cost)
        if not valid:self.state['failures'].append(receipt)
        self.state['noninit_completed']+=1
        freeze(folder/'RESULT.json',receipt)
        self.checkpoint(f'after-{t+1:03d}')
        print(json.dumps(dict(event='ACTION_COMPLETE',phase=self.phase,group=self.group,arm=self.arm,
            position=pos,valid=valid,best=min(x['mean'] for x in self.state['programs'].values()),
            credit_count=len(self.state['credits']),credit_changed=receipt['credit_changed_actual_action'])),flush=True)

    def run(self,init_count,noninit_count,*,uniform_all=False,total_positions=None):
        if self.common_result is None:
            self.initialize(init_count)
        target=self.state['noninit_completed']+noninit_count
        total_positions=total_positions or init_count+target
        while self.state['noninit_completed']<target:
            self.step(total_positions,uniform=uniform_all or self.state['noninit_completed']<16)
        self.checkpoint('FINAL')
        best=min(self.state['programs'].values(),key=lambda p:(p['mean'],p['first_position'],p['hash']))
        own=cost_snapshot(self.root)
        finished=time.time()
        own['wall_seconds']=finished-self.started
        cost=add_costs(self.common_result['cost'],own) if self.common_result else own
        common_started=self.common_result['started_unix'] if self.common_result else self.started
        composed_wall=cost['wall_seconds']
        # Prespecified wall is actual initialization-to-completion elapsed time;
        # disclose scheduler barriers separately, never substitute active time.
        cost['wall_seconds']=finished-common_started
        cost['composed_work_wall_seconds']=composed_wall
        cost['between_common_and_fork_queue_seconds']=max(0.,self.started-self.common_result['finished_unix']) if self.common_result else 0.
        result=dict(study_id=STUDY_ID,phase=self.phase,group=self.group,arm=self.arm,
            search_best_hash=best['hash'],search_best_mean=best['mean'],
            init_completed=self.state['init_completed'],noninit_completed=self.state['noninit_completed'],
            cost=cost,own_physical_cost=own,common_cost_charged_in_full=bool(self.common_result),
            state_path=str(self.root/'states/FINAL.json'),program_count=len(self.state['programs']),
            credit_count=len(self.state['credits']),failure_count=len(self.state['failures']),
            search_locked=True,started_unix=common_started,own_started_unix=self.started,finished_unix=finished)
        # Reconcile original completed wall time when replaying, never rewrite.
        path=self.root/'SEARCH_RESULT.json'
        if path.exists():return read(path)
        freeze(path,result)
        return result
