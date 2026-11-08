"""Explicit historical INIT and panel whitelist; validation requires new locks."""
import copy
from pathlib import Path
from fuse_addon import evaluation as ev

def root(study):return Path(study)/'REUSE'

def initialize(study,task,group):
    shared=ev.read(root(study)/'SEARCH'/task/group/'COMMON_INITIALIZATION.json')
    if shared['archive_state']['noninit_completed']!=0:raise RuntimeError('POST_INIT_IMPORT')
    for h in set(shared['archive_state']['qualities'])|set(shared['archive_state']['programs']):
        b=(root(study)/'SEARCH/programs'/f'{h}.py').read_bytes()
        if ev.digest(b)!=h:raise RuntimeError('INIT_SOURCE_BYTES')
        ev.freeze_bytes(Path(study)/'programs'/f'{h}.py',b)
    return shared

def register_reused_split(study,split):
    study=Path(study);phase=study/'phases/dev_reuse'
    original=ev.read(root(study)/'PANELS'/f'PANEL_MANIFEST_{split}.json');entries={}
    for key,rows in original['groups'].items():
        entries[key]=[]
        for r in rows:
            b=(root(study)/'PANELS'/r['path']).read_bytes()
            if ev.digest(b)!=r['sha256']:raise RuntimeError('REUSED_PANEL_HASH')
            rel=r['path'].replace('phases/pilot/','phases/dev_reuse/')
            ev.freeze_bytes(study/rel,b);entries[key].append(dict(r,path=rel))
    return ev.freeze(phase/f'PANEL_MANIFEST_{split}.json',dict(study_id=study.name,phase='dev_reuse',split=split,
        mode='DEVELOPMENT_REUSE_ONLY',original_study=original['study_id'],groups=entries))

def cached_init(study,task,group,h,split):
    study=Path(study)
    if split=='feedback':
        return copy.deepcopy(ev.read(root(study)/'SEARCH'/task/group/'COMMON_INITIALIZATION.json')['archive_state']['qualities'].get(h))
    required='SHORTLISTS.json' if split=='screen' else 'SELECTION_LOCK.json'
    if not (study/'phases/dev_reuse'/required).exists():raise RuntimeError('REUSED_VALIDATION_BEFORE_LOCK')
    p=root(study)/'REPORTING/init_validation'/split/task/group/f'{h}.json'
    return ev.read(p) if p.exists() else None

def baseline_rows(study):
    if not (Path(study)/'phases/dev_reuse/SELECTION_LOCK.json').exists():raise RuntimeError('OLD_BASELINE_REPORT_BEFORE_LOCK')
    return ev.read(root(study)/'REPORTING/BASELINE_ROWS.json')
