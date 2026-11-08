"""Networkless Docker worker: fresh isolated OS process for every TSP route.

This file is executed as a script. It imports no host controller and mounts no
credentials. The parent kills the entire route process group on timeout, so a
candidate cannot survive a ThreadPool timeout into the next evaluation.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import random
import resource
import selectors
import signal
import subprocess
import sys
import time

PREFIX = 'FUSE_NATIVE_JSON '
_EVENT_STREAM = None


def emit(record):
    print(PREFIX + json.dumps(record, allow_nan=False, separators=(',', ':')), file=_EVENT_STREAM or sys.stdout, flush=True)


def validate_city(value, allowed):
    import numpy as np
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError('RETURN_TYPE_NOT_INTEGER_CITY')
    if int(value) not in allowed:
        raise ValueError('RETURN_OUTSIDE_ORIGINAL_ALLOWED_SET')
    return int(value)


def validate_route(route, count):
    if len(route) != count or sorted(route) != list(range(count)):
        raise ValueError('INCOMPLETE_CITY_PERMUTATION')


def tsp_route(function, item):
    """Official greedy construction, with shared necessary legality guards.

    Preserves start=0, distance-ordered allowed nodes, min(50,n) neighbor cap,
    last-node completion, and native sum of Euclidean cycle-edge norms.
    """
    import numpy as np
    coordinates = np.asarray(item['coordinates'], dtype=float)
    distances = np.asarray(item['distance_matrix'], dtype=float)
    n = len(coordinates)
    neighbors = np.zeros((n, n), dtype=int)
    for i in range(n):
        neighbors[i] = np.argsort(np.linalg.norm(coordinates[i] - coordinates, axis=1))
    route = np.zeros(n, dtype=int)
    current = destination = 0
    call_count = 0
    for i in range(1, n - 1):
        near = neighbors[current][1:]
        allowed = near[~np.isin(near, route[:i])]
        allowed = allowed[:min(50, n, allowed.size)]
        before = tuple(map(int, allowed))
        answer = function(current, destination, allowed, distances)
        call_count += 1
        current = validate_city(answer, before)
        route[i] = current
    remaining = np.arange(n)[~np.isin(np.arange(n), route[:n-1])]
    if len(remaining) != 1:
        raise ValueError('FINAL_CITY_NOT_UNIQUE')
    route[n-1] = remaining[0]
    nodes = [int(x) for x in route]
    validate_route(nodes, n)
    cost = sum(np.linalg.norm(coordinates[nodes[j]] - coordinates[nodes[j+1]]) for j in range(n-1))
    cost += np.linalg.norm(coordinates[nodes[-1]] - coordinates[nodes[0]])
    edges = [math.hypot(float(coordinates[a, 0] - coordinates[b, 0]),
                        float(coordinates[a, 1] - coordinates[b, 1]))
             for a, b in zip(nodes, nodes[1:] + nodes[:1])]
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError('NONFINITE_OR_NONPOSITIVE_LENGTH')
    return dict(route=nodes, length=float(cost), independent_length=math.fsum(edges),
                edges=edges, call_count=call_count, legal_permutation=True)


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

class LimitedText(io.TextIOBase):
    def __init__(self, limit=8192):
        self.limit, self.parts, self.count = limit, [], 0
    def write(self, value):
        value = str(value)
        if self.count < self.limit:
            self.parts.append(value[:self.limit-self.count])
        self.count += len(value)
        return len(value)
    def flush(self):
        pass
    def summary(self):
        return dict(text=''.join(self.parts), total_chars=self.count, truncated=self.count > self.limit)


def route_child():
    global _EVENT_STREAM
    # Candidate print(), sys.__stdout__, os.write(1/2), and subprocess output
    # cannot enter the trusted event stream. Keep a separate control descriptor.
    _EVENT_STREAM = os.fdopen(os.dup(1), 'w', buffering=1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    os.dup2(null, 2)
    os.close(null)
    payload = json.load(sys.stdin)
    item, source = payload['instance'], payload['source']
    capture = LimitedText()
    began = time.monotonic()
    load_seconds = None
    try:
        import numpy as np
        seed = item['execution_seed']
        random.seed(seed)
        np.random.seed(seed)
        namespace = {'np': np, '__name__': 'native_candidate', '__file__': '/candidate/source.py'}
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(compile(source, '/candidate/source.py', 'exec'), namespace, namespace)
        function = namespace.get('select_next_node')
        if not callable(function):
            raise ValueError('MISSING_SELECT_NEXT_NODE_ABI')
        load_seconds = time.monotonic() - began
        emit(dict(type='LOADED', instance_id=item['instance_id'], load_seconds=load_seconds))
        route_began = time.monotonic()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            result = cvrp_route(function,item) if item.get('task')=='CVRP-Construct' else tsp_route(function,item)
        emit(dict(type='ROUTE_RESULT', instance_id=item['instance_id'], status='EVALUATED',
                  load_seconds=load_seconds, route_seconds=time.monotonic()-route_began,
                  captured=capture.summary(), **result))
    except BaseException as exc:
        emit(dict(type='ROUTE_RESULT', instance_id=item['instance_id'], status='ALGORITHM_FAILURE',
                  length=None, load_seconds=load_seconds,
                  route_seconds=None if load_seconds is None else time.monotonic()-began-load_seconds,
                  error_type=type(exc).__name__, error=str(exc)[:8192], captured=capture.summary()))


def _terminate(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        raise RuntimeError('ROUTE_PROCESS_COULD_NOT_BE_KILLED')


def run_instance(source, item, *, load_seconds, route_seconds, panel_deadline):
    began = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    proc = subprocess.Popen([sys.executable, __file__, '--route-child'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    proc.stdin.write(json.dumps(dict(source=source, instance=item), allow_nan=False).encode())
    proc.stdin.close()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(proc.stderr, selectors.EVENT_READ, 'stderr')
    deadline = min(began + load_seconds, panel_deadline)
    buffer = b''
    diagnostics = bytearray()
    loaded = None
    result = None
    timeout_phase = 'LOAD_TIMEOUT'
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_phase = 'PANEL_TIMEOUT' if time.monotonic() >= panel_deadline else timeout_phase
                break
            events = selector.select(min(remaining, .2))
            for key, _ in events:
                data = os.read(key.fileobj.fileno(), 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == 'stderr':
                    diagnostics.extend(data[:max(0, 8192-len(diagnostics))])
                    continue
                buffer += data
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not line.startswith(PREFIX.encode()):
                        diagnostics.extend(line[:max(0, 8192-len(diagnostics))])
                        continue
                    try:
                        event = json.loads(line[len(PREFIX):])
                    except (UnicodeError, ValueError):
                        continue
                    if event.get('type') == 'LOADED' and loaded is None:
                        loaded = event
                        deadline = min(time.monotonic()+route_seconds, panel_deadline)
                        timeout_phase = 'ROUTE_TIMEOUT'
                    if event.get('type') == 'ROUTE_RESULT':
                        result = event
                if len(buffer) > 1024 * 1024:
                    raise RuntimeError('CANDIDATE_UNBOUNDED_OUTPUT')
            if result is not None:
                break
        if result is None:
            timed_out = time.monotonic() >= deadline
            result = dict(type='ROUTE_RESULT', instance_id=item['instance_id'],
                          status=timeout_phase if timed_out else 'CHILD_PROCESS_FAILURE', length=None,
                          load_seconds=loaded.get('load_seconds') if loaded else None,
                          route_seconds=None,
                          error_type=timeout_phase if timed_out else 'NO_TERMINAL_RECORD')
    finally:
        selector.close()
        _terminate(proc)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(child_returncode=proc.returncode, child_wall_seconds=time.monotonic()-began,
                  child_cpu_user_seconds=usage_after.ru_utime-usage_before.ru_utime,
                  child_cpu_system_seconds=usage_after.ru_stime-usage_before.ru_stime,
                  diagnostic_output=diagnostics.decode(errors='replace'))
    return result


def worker_main():
    source = Path('/candidate/source.py').read_bytes().decode('utf-8')
    panel = json.loads(Path('/panel/registered.json').read_text())
    limits = panel['limits']
    began = time.monotonic()
    deadline = began + limits['panel_seconds'] if limits['panel_seconds'] is not None else math.inf
    rows = []
    emit(dict(type='WORKER_STARTED', pid=os.getpid(), time_unix=time.time(),
              instance_count=len(panel['instances']), role=panel['role'], limits=limits,
              source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              panel_sha256=hashlib.sha256(Path('/panel/registered.json').read_bytes()).hexdigest()))
    for item in panel['instances']:
        if time.monotonic() >= deadline:
            break
        emit(dict(type='INSTANCE_STARTED', instance_id=item['instance_id'], time_unix=time.time()))
        row = run_instance(source, item, load_seconds=limits['load_seconds'],
                           route_seconds=limits['route_seconds'] or 500,
                           panel_deadline=deadline)
        rows.append(row)
        emit(row)
        if row['status'] != 'EVALUATED':
            break
    complete = len(rows) == len(panel['instances']) and all(x['status'] == 'EVALUATED' for x in rows)
    import numpy as np
    mean = float(np.average([x['length'] for x in rows])) if complete else None
    emit(dict(type='WORKER_RESULT', complete=complete, mean=mean, rows=rows,
              physical_started=len(rows), physical_completed=sum(x['status']=='EVALUATED' for x in rows),
              elapsed_seconds=time.monotonic()-began, time_unix=time.time(),
              status='EVALUATED' if complete else 'ALGORITHM_FAILURE'))


if __name__ == '__main__':
    route_child() if '--route-child' in sys.argv else worker_main()
