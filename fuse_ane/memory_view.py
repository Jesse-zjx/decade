"""Finite version-bound cards and outcome-independent menus; no LLM calls."""
import copy,difflib,json,random
from .regions import catalog
from .reactive_policy import byte_ranges,FocusChain
from .edit_evidence import match_case
from fuse_addon.evaluation import digest

def full_catalog(io,h,index):
    rows=catalog(io.source(h),task=io.task,group=io.group,arm=io.arm,episode=index,parent_sha=h,bounded=False)
    for r in rows:
        r['unit_path']=r['owner'] if r['scope']=='B' else '|'.join(r.get('function_names',[])) if r['scope']=='C' else '<module>'
        r['byte_ranges']=byte_ranges(io.source(h),r['spans'])
    return rows

def prepare(io,evidence,index):
    programs=io.state['programs'];h=min(programs,key=lambda k:(programs[k]['mean'],k));source=io.source(h)
    catalog_rows=full_catalog(io,h,index);planned=('B','B','C','C','G_ALT','G_MIX')[(index-1)%6]
    scope=planned[0];pool=[r for r in catalog_rows if r['scope']==scope]
    if not pool:
        scope='C' if scope=='B' and any(r['scope']=='C' for r in catalog_rows) else 'G'
        pool=[r for r in catalog_rows if r['scope']==scope]
    operator='MIX' if planned=='G_MIX' else 'ALT' if scope=='G' else 'REFINE'
    rng=random.Random(io.rng('edit-menu',index));rng_before=copy.deepcopy(rng.getstate())
    matches=[];omitted=[]
    for e in evidence:
        m,reason=match_case(e,source,catalog_rows,io.context,index,programs)
        if m:matches.append(m)
        else:omitted.append(dict(evidence_id=e['evidence_id'],reason=reason))
    matches.sort(key=lambda m:(m['status']=='HISTORICAL_LINEAGE',-m['record']['episode'],-m['record']['attempt'],m['record']['evidence_id']))
    # G accepts only FULL; expose bounded local units as case references, never
    # as permitted REPLACE choices for a structural round.
    domain=pool if scope!='G' else [r for r in catalog_rows if r['scope'] in ('B','C')]
    key=lambda r:(r['scope'],tuple(map(tuple,r['spans'])))
    selected=[]
    for m in matches:
        r=m['region']
        if key(r) in {key(p) for p in domain} and key(r) not in {key(p) for p in selected}:
            selected.append(r)
            if len(selected)==2:break
    rest=[r for r in domain if key(r) not in {key(p) for p in selected}]
    selected+=rng.sample(rest,min(6-len(selected),len(rest)))
    menu={f'u{i:02d}':copy.deepcopy(r) for i,r in enumerate(selected)}
    menu['GLOBAL']=copy.deepcopy(next(r for r in catalog_rows if r['scope']=='G'))
    ids={key(r):k for k,r in menu.items()};cards=[];used=set();code_chars=0
    # First cover different current units, then allow a second recent contrast.
    ordered=[]
    for m in matches:
        k=key(m['region'])
        if k in ids and k not in used:ordered.append(m);used.add(k)
    ordered += [m for m in matches if key(m['region']) in ids and m not in ordered]
    for m in ordered:
        if len(cards)==4:break
        e=m['record'];before='\n'.join(p['before'] for p in e['parts']);after='\n'.join(p['after'] for p in e['parts'])
        text='BEFORE:\n'+before+'\nAFTER:\n'+after;truncated=False
        if len(text)>2000:
            diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before',tofile='after',n=2))
            text='EXACT_DIFF_EXCERPT (full source retained in evidence):\n'+diff[:1880]+'\n[TRUNCATED]';truncated=True
            if not any(s.startswith(('+','-')) and not s.startswith(('+++','---')) for s in diff[:1880].splitlines()):
                omitted.append(dict(evidence_id=e['evidence_id'],reason='TRUNCATION_REMOVED_EDIT'));continue
        if code_chars+len(text)>6000:
            omitted.append(dict(evidence_id=e['evidence_id'],reason='CODE_CHARACTER_CAP'));continue
        code_chars+=len(text)
        cards.append(dict(id=f'e{len(cards):02d}',evidence_id=e['evidence_id'],target=ids[key(m['region'])],
            context_status=m['status'],matched_side=m['matched_side'],parent_sha=e['parent_sha'],child_sha=e['child_sha'],
            current_effect_unknown=m['current_effect_unknown'],target_credit=m['current_target_credit'] if io.arm=='F' else None,
            code=text,truncated=truncated,source_part_hashes=[{k:v for k,v in p.items() if k.endswith('_sha')} for p in e['parts']],
            effect=copy.deepcopy(e['effect']) if io.arm=='F' else None,
            note='Historical experience; current effect UNKNOWN.' if m['current_effect_unknown'] else 'Observed before-to-after contrast; sign is NOT reversed for current AFTER.'))
    donor=None;donor_info=None
    if operator=='MIX':
        focus=FocusChain(io.rng('edit-donor',index));donor,donor_info=focus.donor(programs,h)
    for uid,r in menu.items():r.update(parent_sha=h,donor_sha=donor,operator=operator,operation=operator,action_id=uid,local_key=digest(h+'|'+r['region_sha']+'|'+operator))
    return dict(episode=index,parent_sha=h,menu=menu,planned_scope=planned,effective_scope=scope,operator=operator,
        host_full=scope=='G',donor_sha=donor,donor_sampling=donor_info,menu_rng_before=rng_before,menu_rng_after=copy.deepcopy(rng.getstate()),
        memory=dict(cases=cards,status='AVAILABLE' if cards else 'EMPTY_MEMORY',code_chars=code_chars,omitted=omitted),
        quality_before=dict(parent=programs[h]['mean'],elite=min(x['mean'] for x in programs.values()),scale=io.reference['mean']),
        target_selected_pre_evaluation=False,selector_extra_http=0)
