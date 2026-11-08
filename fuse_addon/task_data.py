"""Frozen, arm-independent data for the native attribution study.

Dataset construction never evaluates a generated candidate. The TSP sampler
loads the frozen official HiFo module, replacing only its hidden seed reset.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import threading

STUDY_ID = 'fuse-native-queue-slot-20260923-01'
_SAMPLER_LOCK = threading.Lock()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def sha256(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, (STUDY_ID,) + parts)).encode()).digest()[:4], 'big')


def _freeze(path, value):
    from .evaluation import freeze
    return freeze(path, value)


def official_tsp_sampler_path(repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[1])
    return repo / 'vendor/upstream_hifo/hifo/src/hifo/problems/optimization/tsp_greedy/get_instance.py'


def sample_tsp(count, cities, seed, sampler_path=None):
    """Execute actual upstream sampler with only its reset seed substituted.

    Upstream uses global NumPy RNG and may hard-code seed=2024. The private
    shim forwards all operations to NumPy except random.seed. Calls run under
    a lock and restore the host RNG state; generation cannot perturb search
    RNG. Sampling output is checked against the specified uniform algorithm.
    """
    import numpy as np
    path = Path(sampler_path or official_tsp_sampler_path()).resolve()
    spec = importlib.util.spec_from_file_location('_native_hifo_tsp_sampler', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with _SAMPLER_LOCK:
        before = np.random.get_state()
        original_seed = np.random.seed
        calls = []
        try:
            def own_seed(_seed=None):
                calls.append(_seed)
                original_seed(seed)
            # Only the module's NumPy view gets patched, never global numpy.
            class RandomView:
                seed = staticmethod(own_seed)
                def __getattr__(self, name):
                    return getattr(np.random, name)
            class NumpyView:
                random = RandomView()
                def __getattr__(self, name):
                    return getattr(np, name)
            module.np = NumpyView()
            original_seed(seed)
            data = module.GetData(count, cities).generate_instances()
        finally:
            np.random.set_state(before)
    if len(data) != count:
        raise RuntimeError('OFFICIAL_SAMPLER_INSTANCE_COUNT_CHANGED')
    # A source witness, not a substitute implementation or quality check.
    expected = np.random.RandomState(seed).rand(count, cities, 2)
    for index, (coordinates, distances) in enumerate(data):
        if not np.array_equal(coordinates, expected[index]):
            raise RuntimeError('OFFICIAL_UNIFORM_SAMPLER_SEMANTICS_CHANGED')
        if not np.array_equal(distances, np.linalg.norm(coordinates[:, None] - coordinates, axis=2)):
            raise RuntimeError('OFFICIAL_DISTANCE_MATRIX_SEMANTICS_CHANGED')
    return data, dict(sampler_path=str(path), sampler_sha256=sha256(path.read_bytes()),
                     hidden_seed_calls=calls, replacement_seed=seed,
                     numpy_version=np.__version__, generator='official GetData.generate_instances')


def register_tsp_panel(path, *, stage, group, role, count=None, cities=None,
                       study_id=STUDY_ID, sampler_path=None):
    """Create exact Search/Val/Test bytes once, without exposing quality.

    Same stage/group/role is shared across arms. A stage distinguishes D0,
    possible D1, C, ablation, transfer; no old Probe/quality influences seeds.
    """
    path = Path(path).resolve()
    if role == 'Search8':
        expected_count, expected_cities = 8, 50
    elif role in ('Val128','Audit128'):
        expected_count, expected_cities = 128, 50
    elif role in ('Test50', 'Test100', 'Test200'):
        expected_count, expected_cities = (256 if stage=='C3' and role=='Test50' else 100), int(role[4:])
    else:
        raise ValueError('UNKNOWN_TSP_PANEL_ROLE: ' + str(role))
    count = expected_count if count is None else count
    cities = expected_cities if cities is None else cities
    if (count, cities) != (expected_count, expected_cities):
        raise ValueError('FROZEN_TSP_PANEL_SHAPE_CHANGED')
    identity = dict(study_id=study_id, stage=stage, group=group, role=role,
                    count=count, cities=cities, task='TSP-Construct')
    if path.exists():
        value = json.loads(path.read_text())
        if value['identity'] != identity:
            raise RuntimeError('PANEL_IDENTITY_CONFLICT')
        return panel_manifest(path, value)
    derivation = '/'.join(map(str, (study_id, stage, group, role)))
    seed = int.from_bytes(hashlib.sha256(derivation.encode()).digest()[:4], 'big')
    data, witness = sample_tsp(count, cities, seed, sampler_path)
    namespace = int.from_bytes(hashlib.sha256((study_id + '/execution').encode()).digest()[:4], 'big')
    import numpy as np
    streams = np.random.SeedSequence([namespace, seed]).spawn(count)
    instances = []
    for index, ((coords, dist), stream) in enumerate(zip(data, streams)):
        instances.append(dict(instance_id=f'{stage}/{group}/{role}/{index:03}',
                              coordinates=coords.tolist(), distance_matrix=dist.tolist(),
                              execution_seed=int(stream.generate_state(1)[0])))
    value = dict(identity=identity, task='TSP-Construct', role=role,
                 seed=seed, seed_derivation=derivation, execution_namespace=namespace,
                 sampler=witness, instances=instances, independent_of_old_quality=True,
                 limits=dict(panel_seconds=500 if role == 'Search8' else None,
                             route_seconds=None if role == 'Search8' else 300,
                             load_seconds=10),
                 quality_access='SEARCH' if role == 'Search8' else 'POST_SEARCH_VAL' if role == 'Val128'
                                else 'AFTER_LOCKED_SEARCH_ONLY_SELECTION' if role=='Audit128'
                                else 'AFTER_ALL_STAGE_OUTPUTS_AND_VAL_SELECTIONS_LOCKED',
                 optimal_tour_reference=None,
                 gap_status='MISSING_NO_FROZEN_OPTIMAL_REFERENCE_FOR_NEW_INSTANCES')
    _freeze(path, value)
    return panel_manifest(path, value)


def panel_manifest(path, value=None):
    path = Path(path).resolve()
    value = value if value is not None else json.loads(path.read_text())
    return dict(path=str(path), panel_path=str(path), sha256=sha256(path.read_bytes()),
                panel_sha256=sha256(path.read_bytes()), identity=value['identity'],
                role=value['role'], task=value['task'], count=len(value['instances']),
                seed=value['seed'], ordered_coordinates_sha256=sha256(canonical(
                    [x['coordinates'] for x in value['instances']])))


# Exact plain NN from official EoH TSPCONST.template_program; a read-only cost
# anchor evaluated separately in every arm/run, never FUSE-only initialization.
NN_ANCHOR_SOURCE = '''import numpy as np\n\ndef select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:\n    return unvisited_nodes[np.argmin(distance_matrix[current_node][unvisited_nodes])]\n'''


def register_tsp_group(root, stage, group, *, include_test=False, sampler_path=None):
    root = Path(root)
    roles = ['Search8', 'Val128'] + (['Test50', 'Test100', 'Test200'] if include_test else [])
    return {role: register_tsp_panel(root / stage / str(group) / (role + '.json'),
                stage=stage, group=group, role=role, sampler_path=sampler_path) for role in roles}
