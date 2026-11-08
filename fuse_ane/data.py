"""New phase/task/group instance streams; no historical instance loading."""
import hashlib
from pathlib import Path
import numpy as np
from fuse_addon.evaluation import freeze,read,digest
from fuse_paired.runtime import panel_limits
from .core import stable_seed

STUDY_ID='fuse-shared-edit-siblings-20261007'
TASKS={'tsp':'TSP-Construct','cvrp':'CVRP-Construct'}
COUNTS={'feedback':64,'screen':64,'test':256}

PHASE_ARMS={'dev_reuse':('F2','F1','N2'),'confirm':('F2','N2','H')}
PREFIXES={a:(60000,90000,120000,160000) for a in ('F2','F1','N2')}
PREFIXES.update({a:(40000,60000,80000,90000,120000,160000) for a in ('H','INIT')})

def groups(phase):
    counts={'dev_reuse':(('tsp',4),('cvrp',4)),'confirm':(('tsp',12),('cvrp',8))}
    if phase not in counts:raise ValueError('UNKNOWN_PHASE')
    return [(task,f"{'p' if phase=='dev_reuse' else 'c'}{i:02d}") for task,n in counts[phase] for i in range(n)]


def make_instances(phase,task,group,split,count,cities=50):
    out=[]
    for i in range(count):
        seed=stable_seed(STUDY_ID,phase,task,group,split,i)
        rng=np.random.Generator(np.random.PCG64(seed))
        xy=rng.random((cities+(task=='cvrp'),2))
        row=dict(instance_id=f'{STUDY_ID}/{phase}/{task}/{group}/{split}/{i:03d}',coordinates=xy.tolist(),
            distance_matrix=np.linalg.norm(xy[:,None]-xy,axis=2).tolist(),
            execution_seed=int(np.random.Generator(np.random.PCG64(stable_seed(STUDY_ID,phase,task,group,split,i,'execution'))).integers(0,2**32,dtype=np.uint32)))
        if task=='cvrp':row.update(demands=[0]+rng.integers(1,10,size=cities).tolist(),depot=0,capacity=40)
        out.append(row)
    return out

def register_split(study,phase,split):
    study=Path(study);base=study/'phases'/phase
    if phase=='confirm' and not read(study/'PILOT_GATE.json')['passed']:raise RuntimeError('PILOT_GATE_NOT_PASSED')
    if not (study/'EXECUTION_FREEZE.json').exists():raise RuntimeError('FREEZE_BEFORE_PANELS')
    if split in ('screen','test') and not (base/('SHORTLISTS.json' if split=='screen' else 'SELECTION_LOCK.json')).exists():raise RuntimeError('SELECTION_ACCESS_LOCK')
    path=base/f'PANEL_MANIFEST_{split}.json'
    if path.exists():return read(path)
    if phase=='dev_reuse':
        from .reuse import register_reused_split
        return register_reused_split(study,split)
    entries={}
    for task,g in groups(phase):
        items=make_instances(phase,task,g,split,COUNTS[split])
        slices=[('Search32',items[:32]),('Search32',items[32:])] if split=='feedback' else [('Probe64' if split=='screen' else 'Test256',items)]
        entries[f'{task}/{g}']=[]
        for j,(role,rows) in enumerate(slices):
            p=base/'panels'/task/g/f'{split}-{j}.json'
            panel=dict(identity=dict(study_id=STUDY_ID,phase=phase,task=task,group=g,split=split,role=role,count=len(rows),cities=50),
                task=TASKS[task],role=role,instances=rows,limits=panel_limits(role),
                seed_recipe='SHA256(study|phase|task|group|split|instance)[:16] big endian -> PCG64; coordinates then demands',
                optimal_tour_reference=None)
            freeze(p,panel)
            entries[f'{task}/{g}'].append(dict(path=str(p.relative_to(study)),sha256=digest(p.read_bytes()),role=role,count=len(rows)))
    return freeze(path,dict(study_id=STUDY_ID,phase=phase,split=split,groups=entries))

TSP_NN='''import numpy as np
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
'''
CVRP_NN='''def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    feasible = [int(n) for n in unvisited_nodes if demands[int(n)] <= rest_capacity]
    if not feasible:
        return int(depot)
    return min(feasible, key=lambda n: (distance_matrix[current_node, n], n))
'''
