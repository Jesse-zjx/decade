"""Frozen V3 CVRP data and route facts; no scientific execution on import.

The standalone cvrp_route is worker code only. Local tests inspect its source
and validate recorded synthetic routes; they never call a candidate function.
"""
from __future__ import annotations
import math

TASK = 'CVRP-Construct'
DEPOT = 0
CAPACITY = 40
PANEL_SHAPES = {'Search32': (32, 50), 'Test50': (256, 50), 'Test100': (128, 100)}
CVRP_SIGNATURE = 'select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix) -> int'
NN_ANCHOR_SOURCE = '''import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    feasible = [int(node) for node in unvisited_nodes if demands[int(node)] <= rest_capacity]
    if not feasible:
        return int(depot)
    return min(feasible, key=lambda node: (float(distance_matrix[current_node, node]), node))
'''
CVRP_DESCRIPTION = (
    'Construct capacitated vehicle routes by choosing the next unserved customer. '
    'Depot is node 0; there are 50 customers (100 in fixed generalization), each with demand 1 through 9. '
    'Vehicle capacity is 40. Select an unserved customer whose demand does not exceed remaining capacity. '
    'Away from depot you may close the current nonempty route by returning to depot, which resets capacity to 40. '
    'If no customer is feasible, return to depot. At depot with unserved customers, a depot self-loop is invalid. '
    'The host ends construction after every customer has been served exactly once and closes the final route to depot. '
    'Return an integer node, never bool. The host independently sums Euclidean distances over all route edges; minimize it. '
    'The main-loop limit is (number_of_customers + 1) ** 2 calls; inputs must not modify the host state.')
CVRP_FUNC_DESCRIPTION = (
    'Inputs: current_node and depot are integer node IDs; unvisited_nodes is an ndarray of all unserved customer IDs; '
    'rest_capacity is the remaining integer capacity; demands is an array including depot demand zero; '
    'distance_matrix is the full Euclidean matrix. Output one legal integer customer ID or depot. '
    'Customer coordinates are independent uniform [0,1)^2, customer demands are independent uniform integers 1..9.')


def _real(value):
    return type(value) in (int, float) and math.isfinite(value)


def _instance_facts(item):
    if not isinstance(item, dict):raise ValueError('CVRP_INSTANCE_NOT_OBJECT')
    if type(item.get('depot')) is not int or item['depot'] != DEPOT:
        raise ValueError('CVRP_INSTANCE_DEPOT_CONFLICT')
    if type(item.get('capacity')) is not int or item['capacity'] != CAPACITY:
        raise ValueError('CVRP_INSTANCE_CAPACITY_CONFLICT')
    coords=item.get('coordinates');demands=item.get('demands')
    if (not isinstance(coords, list) or len(coords)<2 or
            any(not isinstance(p, list) or len(p)!=2 or any(not _real(v) or not 0<=v<1 for v in p) for p in coords)):
        raise ValueError('CVRP_INSTANCE_COORDINATES_INVALID')
    if (not isinstance(demands, list) or len(demands)!=len(coords) or
            any(type(v) is not int for v in demands) or demands[0]!=0 or any(not 1<=v<=9 for v in demands[1:])):
        raise ValueError('CVRP_INSTANCE_DEMANDS_INVALID')
    matrix=item.get('distance_matrix');n=len(coords)
    if (not isinstance(matrix, list) or len(matrix)!=n or
            any(not isinstance(row, list) or len(row)!=n for row in matrix)):
        raise ValueError('CVRP_INSTANCE_DISTANCE_SHAPE')
    for i in range(n):
        for j in range(n):
            value=matrix[i][j]
            if not _real(value) or value<0 or not math.isclose(value,math.dist(coords[i],coords[j]),rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError('CVRP_INSTANCE_DISTANCE_CONFLICT')
    return coords,demands


def make_instances(count, customers, seed, execution_seed_fn):
    """Return (JSON-ready items, sampler witness); never execute a candidate.

    The caller supplies the study/phase/group/role identity and index-based
    execution seed derivation. Production panel counts are fixed separately.
    """
    if type(count) is not int or count<=0:raise ValueError('CVRP_INVALID_INSTANCE_COUNT')
    if type(customers) is not int or customers not in (50,100):raise ValueError('CVRP_FIXED_CUSTOMER_COUNT')
    if type(seed) is not int or not 0<=seed<2**32:raise ValueError('CVRP_INVALID_DATA_SEED')
    if not callable(execution_seed_fn):raise ValueError('CVRP_EXECUTION_SEED_FUNCTION_REQUIRED')
    import numpy as np
    rng=np.random.default_rng(seed);items=[]
    for index in range(count):
        coords=rng.random((customers+1,2));demands=[0]+rng.integers(1,10,size=customers).tolist()
        distances=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=2)
        execution_seed=execution_seed_fn(index)
        if type(execution_seed) is not int or not 0<=execution_seed<2**32:raise ValueError('CVRP_INVALID_EXECUTION_SEED')
        items.append(dict(instance_id=f'cvrp/{index:03d}',coordinates=coords.tolist(),distance_matrix=distances.tolist(),
                          demands=demands,depot=DEPOT,capacity=CAPACITY,execution_seed=execution_seed))
    return items,dict(schema='V3_CVRP_UNIFORM_CUSTOMERS_V1',generator='numpy.random.default_rng/PCG64',
        numpy_version=np.__version__,seed=seed,count=count,customers=customers,depot=DEPOT,capacity=CAPACITY,
        coordinates='independent uniform [0,1)^2 including depot',customer_demands='independent uniform integers 1..9',
        draw_order='for each instance: all coordinates, then customer demands',
        distances='Euclidean numpy.linalg.norm',execution_seeds='caller-supplied execution_seed_fn(index)')


def validate_route(row, item):
    """Verify a recorded route and worker length; return independently computed facts.

    Raises ValueError on malformed panel facts, route, metadata or objective.
    This function never mutates inputs, loads source, or invokes a candidate.
    Small public synthetic instances are supported; make_instances and the
    stage panel guard enforce the frozen 50/100-customer production sizes.
    """
    coords,demands=_instance_facts(item);customers=len(coords)-1
    if not isinstance(row,dict):raise ValueError('HOST_CVRP_ROW_NOT_OBJECT')
    route=row.get('route')
    if (not isinstance(route,list) or len(route)<3 or any(type(node) is not int for node in route)):
        raise ValueError('HOST_CVRP_ROUTE_NOT_INTEGER_LIST')
    if route[0]!=DEPOT or route[-1]!=DEPOT:raise ValueError('HOST_CVRP_ROUTE_NOT_CLOSED_AT_DEPOT')
    if len(route)-2>(customers+1)**2:raise ValueError('HOST_CVRP_ITERATION_LIMIT')
    if any(node<0 or node>customers for node in route):raise ValueError('HOST_CVRP_NODE_OUT_OF_RANGE')
    if any(a==DEPOT and b==DEPOT for a,b in zip(route,route[1:])):raise ValueError('HOST_CVRP_DEPOT_SELF_LOOP')
    if sorted(node for node in route if node!=DEPOT)!=list(range(1,customers+1)):
        raise ValueError('HOST_CVRP_CUSTOMERS_NOT_EXACTLY_ONCE')
    load=0;vehicle_loads=[]
    for node in route[1:]:
        if node==DEPOT:
            vehicle_loads.append(load);load=0
        else:
            load+=demands[node]
            if load>CAPACITY:raise ValueError('HOST_CVRP_CAPACITY_EXCEEDED')
    calls=len(route)-2
    if 'call_count' in row and (type(row['call_count']) is not int or row['call_count']!=calls):
        raise ValueError('HOST_CVRP_CALL_COUNT_CONFLICT')
    edges=[math.dist(coords[a],coords[b]) for a,b in zip(route,route[1:])]
    length=math.fsum(edges);reported=row.get('length')
    if not _real(reported) or reported<0 or not math.isclose(reported,length,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('HOST_CVRP_ROUTE_LENGTH_MISMATCH')
    return dict(route=list(route),length=length,independent_length=length,host_independent_length=length,
                edges=edges,call_count=calls,vehicle_loads=vehicle_loads,legal_capacity_routes=True,
                customer_count=customers,depot=DEPOT,capacity=CAPACITY)


def cvrp_route(function, item):
    """Standalone worker function: inspect/freeze locally; execute only remotely.

    No imports from this contract or any controller are required by its source.
    Candidate arrays are copies; legality and objective use private panel facts.
    """
    import math
    import numbers
    import numpy as np
    coords=item['coordinates'];demands=item['demands'];matrix=item['distance_matrix']
    real=lambda v:type(v) in (int,float) and math.isfinite(v)
    if type(item.get('depot')) is not int or item['depot']!=0 or type(item.get('capacity')) is not int or item['capacity']!=40:
        raise ValueError('CVRP_INSTANCE_DEPOT_OR_CAPACITY_CONFLICT')
    if (not isinstance(coords,list) or len(coords)<2 or
            any(not isinstance(p,list) or len(p)!=2 or any(not real(v) or not 0<=v<1 for v in p) for p in coords)):
        raise ValueError('CVRP_INSTANCE_COORDINATES_INVALID')
    n=len(coords)-1
    if (not isinstance(demands,list) or len(demands)!=n+1 or any(type(v) is not int for v in demands)
            or demands[0]!=0 or any(not 1<=v<=9 for v in demands[1:])):
        raise ValueError('CVRP_INSTANCE_DEMANDS_INVALID')
    coords=tuple(tuple(p) for p in coords);demands=tuple(demands)
    distances=np.asarray([[math.dist(a,b) for b in coords] for a in coords],dtype=float)
    if (not isinstance(matrix,list) or len(matrix)!=n+1 or
            any(not isinstance(row,list) or len(row)!=n+1 or any(not real(v) or v<0 for v in row) for row in matrix) or
            not np.allclose(np.asarray(matrix,dtype=float),distances,rtol=1e-12,atol=1e-12)):
        raise ValueError('CVRP_INSTANCE_DISTANCE_CONFLICT')
    route=[0];unserved=set(range(1,n+1));current=0;remaining=40;calls=0;loads=[];load=0
    while unserved:
        if calls>=(n+1)**2:raise ValueError('CVRP_ITERATION_LIMIT')
        allowed={node for node in unserved if demands[node]<=remaining}
        if current!=0:allowed.add(0)
        unvisited=np.asarray(sorted(unserved),dtype=int)
        demand_view=np.asarray(demands,dtype=int).copy();distance_view=distances.copy()
        unvisited.setflags(write=False);demand_view.setflags(write=False);distance_view.setflags(write=False)
        answer=function(current,0,unvisited,remaining,demand_view,distance_view);calls+=1
        if isinstance(answer,(bool,np.bool_)) or not isinstance(answer,numbers.Integral):
            raise ValueError('RETURN_TYPE_NOT_INTEGER_NODE')
        node=int(answer)
        if node not in allowed:raise ValueError('RETURN_OUTSIDE_ORIGINAL_CVRP_ALLOWED_SET')
        route.append(node);current=node
        if node==0:
            loads.append(load);load=0;remaining=40
        else:
            unserved.remove(node);load+=demands[node];remaining-=demands[node]
    route.append(0);loads.append(load)
    edges=[math.dist(coords[a],coords[b]) for a,b in zip(route,route[1:])]
    length=math.fsum(edges)
    if not math.isfinite(length):raise ValueError('NONFINITE_ROUTE_LENGTH')
    return dict(route=route,length=length,independent_length=length,edges=edges,
                call_count=calls,vehicle_loads=loads,legal_capacity_routes=True)
