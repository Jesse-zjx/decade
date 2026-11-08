"""Measured edits, exact bytes and full paired Feedback; no candidate execution."""
import ast,copy,hashlib,json
from .edit_reference import Replacement,apply_exact,paired_effect
from .reactive_policy import byte_ranges
from fuse_addon.evaluation import digest,encoded,freeze

def execution_contract(q):
    # These two fields identify the recorder, not the executable contract. Only
    # exact pre-fork INIT reuse can differ here; all other fields stay bound.
    return {k:v for k,v in q.get('execution_binding',{}).items() if k not in ('study_id','runtime_identity')}

def measured_pair(p,q):
    if not p or not q or not p.get('complete') or not q.get('complete'):raise ValueError('INCOMPLETE_FEEDBACK')
    if p.get('split')!='feedback' or q.get('split')!='feedback':raise ValueError('NOT_FEEDBACK')
    a,b=p['rows'],q['rows']
    if len(a)!=64 or len(b)!=64 or p.get('denominator')!=64 or q.get('denominator')!=64:raise ValueError('NOT_FULL_FEEDBACK64')
    ids=[r['instance_id'] for r in a]
    if ids!=[r['instance_id'] for r in b] or len(set(ids))!=64:raise ValueError('INSTANCE_PAIR_MISMATCH')
    if any(r['status']!='EVALUATED' for r in a+b):raise ValueError('INVALID_MEASUREMENT')
    if not p.get('panel_hashes') or p['panel_hashes']!=q.get('panel_hashes'):raise ValueError('PANEL_MISMATCH')
    if not execution_contract(p) or execution_contract(p)!=execution_contract(q):raise ValueError('EXECUTION_MISMATCH')
    return [r['length'] for r in a],[r['length'] for r in b],ids

def pieces(parent,child,region):
    """Reconstruct child ONLY from one registered unit and preserved backgrounds."""
    pb,qb=parent.encode(),child.encode();rs=byte_ranges(parent,region['spans'])
    if region['scope']=='B':
        if len(rs)!=1:raise ValueError('BLOCK_RANGE_UNION')
        a,b=rs[0];end=len(qb)-(len(pb)-b)
        qr=[(a,end)]
    else:
        tree=ast.parse(child);funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        spans=[]
        for name in region['function_names']:
            if name not in funcs:raise ValueError('FUNCTION_MISSING')
            n=funcs[name];spans.append([min([n.lineno]+[d.lineno for d in n.decorator_list]),n.end_lineno])
        qr=byte_ranges(child,spans)
    if len(rs)!=len(qr):raise ValueError('RANGE_UNION_CHANGED')
    replacements=[];parts=[]
    for (a,b),(c,d) in zip(rs,qr):
        if not (0<=c<d<=len(qb)):raise ValueError('CHILD_RANGE_INVALID')
        before,after=pb[a:b],qb[c:d]
        replacements.append(Replacement(a,b,before,after))
        parts.append(dict(before_start=a,before_end=b,after_start=c,after_end=d,before=before.decode(),after=after.decode(),before_sha=digest(before),after_sha=digest(after)))
    if apply_exact(pb,replacements)!=qb:raise ValueError('OUTSIDE_CHANGE_OR_UNRECORDED_EDIT')
    # Length-framed background pieces prevent concatenation ambiguity.
    cursor=0;background=[]
    for a,b in rs:background.append(pb[cursor:a].hex());cursor=b
    background.append(pb[cursor:].hex())
    return parts,digest(encoded(background))

def record(*,parent,child,region,parent_quality,child_quality,context,episode,attempt,mode,novel,error,scale,elite_before,cost,request_index):
    event=f"{context['study_id']}/{context['phase']}/{context['task']}/{context['group']}/{context['arm']}/{episode}/{attempt}"
    base=dict(event_id=event,evidence_id=digest(event)[:20],context=copy.deepcopy(context),episode=episode,attempt=attempt,request_index=request_index,
        parent_sha=digest(parent),child_sha=digest(child) if child is not None else None,scope=region['scope'],actual_mode=mode,
        unit={k:copy.deepcopy(region.get(k)) for k in ('spans','region_sha','owner','function_names','boundary','unit_path')},
        error=error,cost=copy.deepcopy(cost),effect=None,target_credit=None,causal_generalization_claim=False)
    if not child_quality or not child_quality.get('complete'):
        return dict(base,type='TECHNICAL_FAILURE',quality=None)
    if not novel or parent==child:return dict(base,type='DUPLICATE',quality=child_quality['mean'])
    try:
        a,b,ids=measured_pair(parent_quality,child_quality)
        effect=paired_effect(a,b,scale)
        base.update(effect=effect,fixed_scale=scale,instance_ids=ids,panel_sha=digest(encoded(parent_quality['panel_hashes'])),
            runtime_sha=digest(encoded(execution_contract(parent_quality))),
            elite_gain=100*(elite_before-child_quality['mean'])/scale,quality=child_quality['mean'])
    except (ValueError,KeyError) as exc:return dict(base,type='UNIDENTIFIED_LOCAL_EFFECT',reason=str(exc),effect=None)
    if mode!='REPLACE' or region['scope'] not in ('B','C'):return dict(base,type='GLOBAL_OR_COMPOSITE')
    try:
        raw=''.join(''.join(parent.splitlines(keepends=True)[a-1:b]) for a,b in region['spans'])
        if digest(raw)!=region['region_sha']:raise ValueError('REGISTERED_REGION_BYTES_MISMATCH')
        parts,bg=pieces(parent,child,region)
    except (ValueError,KeyError,SyntaxError) as exc:return dict(base,type='GLOBAL_OR_COMPOSITE',reason=str(exc))
    return dict(base,type='LOCAL_PAIRED',parts=parts,outside_background_sha=bg,quality=child_quality['mean'])

def persist(path,value):
    # freeze is equality-checked and immutable; recovery cannot settle twice.
    return freeze(path,value)

def lineage_distance(programs,x,targets,max_distance=2):
    edges={h:set(r.get('parents',[])) for h,r in programs.items()}
    for h,ps in list(edges.items()):
        for p in ps:edges.setdefault(p,set()).add(h)
    frontier={x};seen=set()
    for distance in range(max_distance+1):
        if frontier.intersection(targets):return distance
        seen.update(frontier);frontier={y for h in frontier for y in edges.get(h,()) if y not in seen}
    return None

def match_case(e,current,regions,context,episode,programs):
    if e['type']!='LOCAL_PAIRED':return None,'NOT_LOCAL_PAIRED'
    if any(e['context'].get(k)!=context.get(k) for k in ('study_id','phase','task','group','arm','feedback_panels','execution_binding')):return None,'CONTEXT_MISMATCH'
    if not 0<episode-e['episode']<=32:return None,'OUTSIDE_LOOKBACK'
    h=digest(current);side='before' if h==e['parent_sha'] else 'after' if h==e['child_sha'] else None
    historical=side is None
    if historical and lineage_distance(programs,h,{e['parent_sha'],e['child_sha']}) is None:return None,'LINEAGE_TOO_FAR'
    matches=[]
    for which in ([side] if side else ['before','after']):
        for r in regions:
            if r['scope']!=e['scope'] or r.get('unit_path')!=e['unit'].get('unit_path'):continue
            raw=current.encode();rs=byte_ranges(current,r['spans'])
            if [raw[a:b].decode() for a,b in rs]!=[p[which] for p in e['parts']]:continue
            if not historical and rs!=[[p[which+'_start'],p[which+'_end']] for p in e['parts']]:continue
            matches.append((r,which))
    unique={(json.dumps(r['spans']),which):(r,which) for r,which in matches}
    if len(unique)!=1:return None,'NO_OR_AMBIGUOUS_EXACT_MATCH'
    r,which=next(iter(unique.values()))
    return dict(record=e,region=r,matched_side=which,status='HISTORICAL_LINEAGE' if historical else 'CURRENT_EXACT_'+which.upper(),
        current_target_credit=None if historical else e['effect'],current_effect_unknown=historical),None
