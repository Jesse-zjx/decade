"""A single generation response binds target/mode/evidence before evaluation."""
import re,json,copy
from pathlib import Path
from .regions import EditError
from .reactive_policy import prompt as structural_prompt
from fuse_addon.evaluation import digest

def decode(response,pre):
    if not isinstance(response,str):raise EditError('UNKNOWN_NO_MODEL_OUTPUT')
    m=re.fullmatch(r'\s*TARGET:\s*([A-Za-z0-9_]+)\s*\nMODE:\s*(REPLACE|FULL)\s*\nEVIDENCE:\s*([^\n]*)\n\s*```python\s*\n([\s\S]*?)\n```\s*',response)
    if not m or '```' in m.group(4):raise EditError('FORMAT_JOINT_ENVELOPE_REQUIRED_SINGLE_FENCE')
    target,mode,raw,code=m.groups();refs=[] if raw.strip()=='NONE' else [s.strip() for s in raw.split(',')]
    if target not in pre['menu']:raise EditError('FORMAT_INVALID_TARGET')
    valid_ids={c['id'] for c in pre['memory']['cases']}
    if len(set(refs))!=len(refs) or any(e not in valid_ids for e in refs):raise EditError('FORMAT_INVALID_EVIDENCE')
    if (mode=='FULL')!=(target=='GLOBAL') or (pre['host_full'] and mode!='FULL'):raise EditError('FORMAT_TARGET_MODE_CONFLICT')
    return dict(target=target,mode=mode,evidence=refs,code=code+'\n',raw_sha256=digest(response),body_sha256=digest(code+'\n'),route='ONE_CALL_TARGET_AND_CODE')

def prompt(io,pre,recent,error=None,failed_source=None):
    menu={uid:{k:v for k,v in r.items() if k in ('scope','spans','byte_ranges','region_sha','owner','function_names','unit_path','boundary','source')} for uid,r in pre['menu'].items()}
    base=(Path(__file__).with_name('prompts')/'EDIT_JOINT.txt').read_text()
    values={'TASK_AND_API':io.task_description,'ACTUAL_PARENT_PROGRAM':io.source(pre['parent_sha']),
        'CURRENT_UNIT_MENU':json.dumps(menu,ensure_ascii=False,sort_keys=True),'EDIT_CASES':json.dumps(pre['memory'],ensure_ascii=False,sort_keys=True),
        'ORDINARY_FEEDBACK':json.dumps(dict(current_parent=pre['quality_before']['parent'],current_elite=pre['quality_before']['elite'],recent=recent[-2:]),ensure_ascii=False,sort_keys=True)}
    # Rendering never rescans inserted source text as a placeholder template.
    base=re.sub(r'\{\{([A-Z_]+)\}\}',lambda m:values[m[1]],base)
    messages=[dict(role='system',content='Generate one executable algorithm under the task API and host edit contract.'),dict(role='user',content=base)]
    if pre['host_full'] or error is not None:
        rec=json.loads(json.dumps(pre['menu']['GLOBAL'],sort_keys=True));canonical_error=json.loads(json.dumps(error,sort_keys=True)) if error is not None else None
        context=structural_prompt(io,rec,recent,error=canonical_error,failed_source=failed_source)
        # Keep structural/repair instruction bytes; final joint envelope explicitly
        # overrides only their legacy output wrapper, and host fixes FULL.
        messages.append(dict(role='user',content=context[-1]['content']+'\nHOST OUTPUT CONTRACT: TARGET: GLOBAL; MODE: FULL. Return the three-line TARGET/MODE/EVIDENCE header and exactly one Python fence required above. The structural/repair task above is unchanged; legacy output wrappers are superseded.'))
    return messages
