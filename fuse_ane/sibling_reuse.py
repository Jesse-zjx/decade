"""Bounded read-only historical INIT/H recovery; never imports H into online search."""
from __future__ import annotations
import copy, json, math, shutil
from pathlib import Path
from fuse_addon import evaluation as ev
from .reuse import initialize, register_reused_split

OLD_ID='fuse-reactive-neighborhood-evolution-20261006'
TASKS=('tsp','cvrp')
GROUPS=tuple(f'p{i:02d}' for i in range(4))
H_ROLES=('H40','H60','H80','H90','H120','H160')

def _read(path):
    return json.loads(Path(path).read_text())

def _copy(source,target):
    ev.freeze_bytes(Path(target),Path(source).read_bytes())

def _request_cost(rows):
    def finite(v):return type(v) in (int,float) and math.isfinite(v)
    known=sum(r['tokens'] for r in rows if finite(r.get('tokens')))
    missing=sum(not finite(r.get('tokens')) for r in rows)
    out=dict(tokens=known if not missing else None,tokens_known=known,tokens_missing=missing,http=len(rows),
             cpu_seconds=None,evaluator_cpu_seconds=None,wall_seconds=None,
             evaluation_cost_status='PREFIX_TIMING_NOT_RECONSTRUCTED_NO_CPU_BACKFILL')
    for key,u in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens')]:
        vs=[(r.get('usage') or {}).get(u) for r in rows]
        out[key+'_known']=sum(v for v in vs if finite(v));out[key+'_missing']=sum(not finite(v) for v in vs)
    out['llm_wall_seconds_known']=sum(r['llm_wall_seconds'] for r in rows if finite(r.get('llm_wall_seconds')))
    out['llm_wall_seconds_missing']=sum(not finite(r.get('llm_wall_seconds')) for r in rows)
    return out

def reconstruct_prefix(endpoint,shared,request_rows,native_positions,budget):
    """Recover only from exact owner/request set and its native result positions.

    Auxiliary insight requests have no candidate. INIT native position numbers
    never stand in for H requests; every consumed generation has one result.
    """
    try:
        requests=endpoint['result']['request_set']
        if endpoint['result'].get('request_set_complete') is not True:raise ValueError('INCOMPLETE_REQUEST_SET')
        bypath={r['path']:r for r in request_rows}
        if len(bypath)!=len(request_rows):raise ValueError('DUPLICATE_PHYSICAL_RECEIPT')
        positions={}
        for row in native_positions:
            key=(row['arm'],row['position'])
            if key in positions:raise ValueError('AMBIGUOUS_NATIVE_POSITION')
            positions[key]=row
        reference_hash=shared['reference']['source_hash']
        reference_record=shared['archive_state']['programs'][reference_hash]
        if reference_record.get('first_position')!=0 or reference_record.get('origin')!='public_nearest_neighbor':
            raise ValueError('PUBLIC_REFERENCE_IDENTITY')
        # The declared public reference precedes requests; no INIT winner lookup.
        archive={reference_hash:reference_record}
        selected=[];physical=[];seen=set();tokens=0;crossing=None;hpositions=set()
        for req in requests:
            owner=req['owner']
            if owner not in ('INIT','H'):raise ValueError('ILLEGAL_REQUEST_OWNER')
            rows=[]
            for path in req['physical_attempts']:
                if path in seen:raise ValueError('PHYSICAL_RECEIPT_COUNTED_TWICE')
                row=bypath[path]
                if row['arm']!=owner:raise ValueError('PHYSICAL_OWNER_MISMATCH')
                if type(row.get('tokens')) is not int:raise ValueError('MISSING_USAGE_BOUNDARY')
                seen.add(path);rows.append(row);tokens+=row['tokens']
            if not rows:raise ValueError('EMPTY_LOGICAL_REQUEST')
            selected.append(req);physical+=rows
            meta=req['metadata'];kind=meta.get('kind')
            if kind in ('generation','initialization'):
                key=(owner,meta['position']);p=positions[key]
                if p.get('record_unreadable'):raise ValueError('UNREADABLE_NATIVE_RESULT')
                if owner=='H':hpositions.add(meta['position'])
                if p.get('valid'):
                    h=p['source_hash'];record=endpoint['programs'].get(h) or shared['archive_state']['programs'].get(h)
                    if record is None or record.get('mean')!=p['candidate_quality']:raise ValueError('NATIVE_QUALITY_BINDING')
                    archive[h]=record
            elif kind!='insight_extraction':raise ValueError('UNKNOWN_NATIVE_REQUEST_KIND:'+str(kind))
            if tokens>=budget:
                crossing=req['logical_id'];break
        if crossing is None:raise ValueError('BUDGET_NOT_REACHED')
        if not archive:raise ValueError('NO_ARCHIVE_AT_BOUNDARY')
        h0=shared['selection']['source_hash'];best=sorted((h for h in archive if h!=h0),key=lambda h:(archive[h]['mean'],h))[:2]
        shortlist=([h0] if h0 in archive else [])+best
        cost=_request_cost(physical)
        cost.update(initialization_charged_in_full=True,shared_initialization_cost=shared['cost'],
                    accounting_scope='Exact original physical request set through first crossing; prefixes not additive')
        return dict(status='RECONSTRUCTED',reached=True,threshold_tokens=budget,tokens=tokens,cost=cost,
            shortlist=shortlist,endpoint_archive=sorted(archive),feedback_by_source={h:v['mean'] for h,v in archive.items()},
            feedback_mean=min(v['mean'] for v in archive.values()),initialization_winner_eligible=h0 in archive,
            request_set=selected,request_set_sha256=ev.digest(ev.encoded(selected)),request_set_complete=True,
            crossing_request=crossing,report_owner='H',own_logical_requests=sum(r['owner']=='H' for r in selected),
            completed_positions=len(hpositions),development_reference_seen=True,contemporaneous=False,
            recovered_without_generation_or_candidate_calls=True,physical_request_paths=[r['path'] for r in physical])
    except (KeyError,TypeError,ValueError) as exc:
        return dict(status='UNRECONSTRUCTABLE',reason=str(exc),reached=False,threshold_tokens=budget,shortlist=[],
                    tokens=None,cost=None,feedback_by_source={},request_set_complete=False,
                    development_reference_seen=True,contemporaneous=False)

def prepare_inputs(cee_reuse_root,rne_compact_root,destination):
    """Offline packaging of bounded compact files. No model/evaluator invocation.

    The existing complete INIT whitelist is copied unchanged. Only H endpoint,
    request/native summaries and actually shortlisted sources/caches are added.
    """
    cee,rne,dst=map(Path,(cee_reuse_root,rne_compact_root,destination))
    if dst.resolve() in (cee.resolve(),rne.resolve()):raise ValueError('NEW_DIRECTORY_REQUIRED')
    # Existing compact REUSE is already a validated whitelist, not a RAW scan.
    for p in cee.rglob('*'):
        if p.is_file():_copy(p,dst/p.relative_to(cee))
    requests=_read(rne/'REQUESTS.json');positions=_read(rne/'NATIVE_POSITIONS.json')
    oldshort=_read(rne/'phases/pilot/SHORTLISTS.json')['groups']
    oldlock=_read(rne/'phases/pilot/SELECTION_LOCK.json')['groups']
    out={};sources=set();cache_files=[];endpoint_sources={}
    for task in TASKS:
        for group in GROUPS:
            key=f'{task}/{group}';shared=_read(cee/'SEARCH'/task/group/'COMMON_INITIALIZATION.json')
            ep_path=rne/'phases/pilot/groups'/task/group/'H_ENDPOINT.json';ep=_read(ep_path)
            endpoint_sources[key]=dict(path=str(ep_path),sha256=ev.digest(ep_path.read_bytes()))
            req=[r for r in requests if r['task']==task and r['group']==group and r['arm'] in ('INIT','H')]
            pos=[r for r in positions if r['task']==task and r['group']==group and r['arm'] in ('INIT','H')]
            out[key]={}
            for role in H_ROLES:
                if role in ('H60','H120'):
                    row=reconstruct_prefix(ep,shared,req,pos,int(role[1:])*1000)
                else:
                    row=copy.deepcopy(oldshort[key][role]);row.update(status='ORIGINAL_LOCKED_ENDPOINT',
                        original_selection=oldlock[key][role],development_reference_seen=True,contemporaneous=False)
                out[key][role]=row;sources.update(row.get('shortlist',[]))
                for h in row.get('shortlist',[]):
                    for split in ('screen','test'):
                        p=rne/'phases/pilot/validation'/split/task/group/'results'/f'{h}.json'
                        if p.exists():
                            rel=Path('REPORTING/H_validation')/split/task/group/f'{h}.json'
                            _copy(p,dst/rel);cache_files.append(dict(path=str(rel),sha256=ev.digest(p.read_bytes())))
    for h in sorted(sources):
        source=rne/'programs'/f'{h}.py'
        if not source.exists() or ev.digest(source.read_bytes())!=h:raise RuntimeError('H_SHORTLIST_SOURCE_UNAVAILABLE:'+h)
        _copy(source,dst/'REPORTING/H_programs'/f'{h}.py')
    physical=_read(rne/'PHYSICAL_EVALUATIONS.json')
    cached_paths=set()
    for row in cache_files:
        q=_read(dst/row['path']);cached_paths.update(f['physical_record'] for f in q.get('fragments',[]))
    summaries=[r for r in physical if str(Path(r['path'])/'EVALUATION.json') in cached_paths or r['path'] in cached_paths]
    ev.freeze(dst/'REPORTING/H_VALIDATION_PHYSICAL.json',summaries)
    ev.freeze(dst/'REPORTING/H_PREFIXES.json',dict(groups=out,source_endpoints=endpoint_sources,
        original_study=OLD_ID,max_added_endpoints=16,source_scope='Reporting only; H programs never enter online archive',
        cache_files=cache_files,old_sources_rehashed_only_when_needed_for_shortlists=True))
    return dict(groups=len(out),reconstructed=sum(r['status']=='RECONSTRUCTED' for v in out.values() for r in v.values()),
        unreconstructable=sum(r['status']=='UNRECONSTRUCTABLE' for v in out.values() for r in v.values()),
        shortlisted_source_count=len(sources),cached_validation_count=len(cache_files))

def historical_prefixes(study,task,group):
    return copy.deepcopy(ev.read(Path(study)/'REUSE/REPORTING/H_PREFIXES.json')['groups'][f'{task}/{group}'])

def expose_h_shortlists(study,task,group,roles):
    """Copy report-only sources after search is complete; never attach to SearchIO."""
    study=Path(study)
    if not (study/'phases/dev_reuse/groups'/task/group/'SEARCH_COMPLETE.json').exists():raise RuntimeError('H_REPORT_BEFORE_SEARCH_COMPLETE')
    for row in roles.values():
        for h in row.get('shortlist',[]):
            p=study/'REUSE/REPORTING/H_programs'/f'{h}.py'
            if ev.digest(p.read_bytes())!=h:raise RuntimeError('H_SOURCE_BYTES')
            _copy(p,study/'programs'/f'{h}.py')

def cached_historical(study,task,group,h,split):
    from .reuse import cached_init
    study=Path(study)
    if split=='feedback':return cached_init(study,task,group,h,split)
    required='SHORTLISTS.json' if split=='screen' else 'SELECTION_LOCK.json'
    path=study/'phases/dev_reuse'/required
    if not path.exists():raise RuntimeError('REUSED_VALIDATION_BEFORE_LOCK')
    all_lock=ev.read(path)
    marker='all_searches_stopped' if split=='screen' else 'locked_before_test'
    expected={f'{t}/{g}' for t in TASKS for g in GROUPS}
    if all_lock.get(marker) is not True or set(all_lock.get('groups',{}))!=expected:
        raise RuntimeError('REUSED_VALIDATION_REQUIRES_COMPLETE_GLOBAL_LOCK')
    if split=='test':
        short=study/'phases/dev_reuse/SHORTLISTS.json'
        if not short.exists() or all_lock.get('shortlists_sha256')!=ev.digest(short.read_bytes()):
            raise RuntimeError('REUSED_TEST_SHORTLIST_LOCK_MISMATCH')
    initial=cached_init(study,task,group,h,split)
    if initial is not None:return initial
    lock=all_lock['groups'][f'{task}/{group}']
    allowed={x for r in lock.values() for x in r.get('shortlist',[])} if split=='screen' else {r.get('selected_source') for r in lock.values()}
    if h not in allowed:raise RuntimeError('H_VALIDATION_CACHE_NOT_LOCKED')
    p=study/'REUSE/REPORTING/H_validation'/split/task/group/f'{h}.json'
    return dict(ev.read(p),reused_paid_historical_H=True) if p.exists() else None

def baseline_rows(study):
    from .reuse import baseline_rows as original
    return original(study)
