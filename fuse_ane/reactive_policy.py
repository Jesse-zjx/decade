"""R/S exact focus-chain bindings and prompts; no model/evaluator calls here."""
import copy,json,math,random
from dataclasses import asdict
from pathlib import Path
from fuse_addon.evaluation import digest,encoded
from .core import Action
from .regions import catalog
from .reactive_core import ReactiveController,FixedController,Observation

PROMPTS=Path(__file__).with_name('prompts')

def template(name,**values):return (PROMPTS/(name+'.txt')).read_text().format(**values)

def byte_ranges(source,spans):
    lines=source.splitlines(keepends=True);offset=[0]
    for line in lines:offset.append(offset[-1]+len(line.encode()))
    ranges=sorted((offset[a-1],offset[b]) for a,b in spans);merged=[]
    for a,b in ranges:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
        else:merged.append([a,b])
    return merged

def byte_size(source,record):return sum(b-a for a,b in byte_ranges(source,record['spans']))

def contains(source,outer,inner):
    big=byte_ranges(source,outer['spans']);small=byte_ranges(source,inner['spans'])
    return all(any(c<=a and b<=d for c,d in big) for a,b in small)

class FocusChain:
    def __init__(self,seed):
        self.rng=random.Random(seed);self.parent_sha=None;self.regions={};self.binding=None
    def bind(self,source,parent_sha,context,episode):
        if self.parent_sha==parent_sha:return
        if digest(source)!=parent_sha:raise ValueError('PARENT_VERSION_CONFLICT')
        before=copy.deepcopy(self.rng.getstate())
        all_regions=catalog(source,task=context['task'],group=context['group'],arm=context['arm'],episode=episode,parent_sha=parent_sha,bounded=False)
        blocks=[r for r in all_regions if r['scope']=='B']
        cs=[r for r in all_regions if r['scope']=='C' and byte_size(source,r)<len(source.encode())]
        block=self.rng.choice(blocks) if blocks else None
        if block:
            enclosing=[r for r in cs if contains(source,r,block) and byte_size(source,r)>byte_size(source,block)]
            unit=min(enclosing,key=lambda r:(byte_size(source,r),r['spans'])) if enclosing else None
        else:unit=self.rng.choice(cs) if cs else None
        self.regions={'G':next(r for r in all_regions if r['scope']=='G')}
        if block:self.regions['B']=block
        if unit:self.regions['C']=unit
        old=self.parent_sha;self.parent_sha=parent_sha
        self.binding=dict(parent_sha=parent_sha,invalidated_parent_sha=old,created_episode=episode,
            rng_state_before=before,rng_state_after=copy.deepcopy(self.rng.getstate()),
            full_catalog_counts=dict(B=len(blocks),C=len(cs)),
            chain={k:dict(region_sha=v['region_sha'],spans=v['spans'],byte_ranges=byte_ranges(source,v['spans']),owner=v['owner']) for k,v in self.regions.items()},
            containment='byte-union only; no semantic independence claim')
    def donor(self,programs,elite):
        eligible=[h for h in programs if h!=elite]
        best=sorted(eligible,key=lambda h:(programs[h]['mean'],h))[:4]
        recent=sorted((h for h in eligible if h not in best),key=lambda h:(-programs[h]['first_index'],h))[:2]
        domain=sorted(best+recent,key=lambda h:(programs[h]['mean'],h))
        ranks={};rank=-1;last=None
        for h in domain:
            loss=programs[h]['mean']
            if last is None or abs(loss-last)>1e-12:rank+=1;last=loss
            ranks[h]=rank
        weights=[2.**(-ranks[h]) for h in domain]
        selected=self.rng.choices(domain,weights=weights,k=1)[0] if domain else None
        return selected,dict(domain=domain,dense_ranks=ranks,probabilities={h:w/sum(weights) for h,w in zip(domain,weights)})

def controller(arm):
    if arm not in ('R','S'):raise ValueError('RNE_ARM')
    return ReactiveController() if arm=='R' else FixedController()

def choose(io,control,focus,index):
    programs=io.state['programs'];elite=min(programs,key=lambda h:(programs[h]['mean'],h))
    before=asdict(control);rng_before=copy.deepcopy(focus.rng.getstate())
    focus.bind(io.source(elite),elite,io.context,index)
    available=tuple(k for k in ('B','C','G') if k in focus.regions)
    planned=control.next(available);rec=copy.deepcopy(focus.regions[planned['scope']])
    donor=None;donor_info=None
    if planned['operator']=='MIX':donor,donor_info=focus.donor(programs,elite)
    op=planned['operator'] if planned['scope']=='G' else 'REFINE'
    operation=op+(':'+donor if donor else ':BACKBONE_SINGLE' if op=='MIX' else '')
    action=Action(f'episode-{index:03d}',elite,rec['region_sha'],rec['scope'],operation)
    rec.update(parent_sha=elite,donor_sha=donor,operation=operation,operator=op,action_id=action.action_id,local_key=action.key)
    shadow=FixedController(completed=control.completed,global_calls=control.global_calls).next(available)
    extra=dict(controller_before=before,controller_dispatched=asdict(control),planned=planned,
        available_scopes=list(available),focus_binding=copy.deepcopy(focus.binding),
        focus_rng_before=rng_before,focus_rng_after=copy.deepcopy(focus.rng.getstate()),donor_sampling=donor_info,
        shadow_fixed=dict(shadow,executed=False),shadow_choice_difference={'S':planned['scope']!=shadow['scope'] or planned['operator']!=shadow['operator']},
        exact_binding=dict(parent_sha=elite,region_sha=rec['region_sha'],operation=operation,donor_sha=donor,
            feedback_panels=io.context['feedback_panels'],execution_binding=io.context['execution_binding']),
        target_quality_before=None,target_credit_measured_before=False)
    return action,rec,extra

def prompt(io,record,recent,*,error=None,failed_source=None):
    parent=record['parent_sha'];programs=io.state['programs'];donor=record.get('donor_sha')
    values=dict(task_description=io.task_description,parent_source=io.source(parent),parent_mean=programs[parent]['mean'],
        incumbent_mean=min(v['mean'] for v in programs.values()),recent_outcomes='\n'.join(recent[-2:]) or 'None',
        scope=record['scope'],region_source=record['source'],boundary_summary=json.dumps(record['boundary'],ensure_ascii=False),
        donor_source=io.source(donor) if donor else '',donor_mean=programs[donor]['mean'] if donor else None,
        actual_error=json.dumps(error,ensure_ascii=False)[:1500],failed_source=failed_source)
    name='REPAIR' if error is not None else 'REFINE' if record['operator']=='REFINE' else 'G_ALT' if record['operator']=='ALT' else 'G_MIX' if donor else 'G_BACKBONE_SINGLE'
    return [dict(role='system',content=template('SYSTEM')),dict(role='user',content=template(name,**values))]

def observe_once(control,observation,receipt=None):
    """A frozen receipt is authoritative on recovery; never settle an episode twice."""
    value=asdict(observation);key=digest(encoded(value))
    if receipt is not None:
        if receipt['observation_sha256']!=key:raise ValueError('CONTROLLER_RECEIPT_CONFLICT')
        for k,v in receipt['after'].items():setattr(control,k,copy.deepcopy(v))
        return receipt
    before=asdict(control);control.observe(observation)
    material=bool(observation.valid and observation.novel and observation.elite_gain_pp>=.01)
    if isinstance(control,FixedController):reason='FIXED_CYCLE_ADVANCE'
    elif material:reason='MATERIAL_PROGRESS_RESET'
    elif observation.technical_failure:reason='FINAL_TECHNICAL_FAILURE_EXPAND'
    elif before['no_progress']+1>=control.failure_patience:reason='PATIENCE_EXPAND_OR_STAY_G'
    else:reason='NO_PROGRESS_CONTINUE'
    return dict(observation=value,observation_sha256=key,before=before,after=asdict(control),material_progress=material,reason=reason)
