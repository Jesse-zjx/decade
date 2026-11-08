"""Bounded AST neighborhoods and deterministic exact-region assembly."""
import ast,copy,difflib,hashlib,random,re,textwrap
from .core import Action,stable_seed
from .data import STUDY_ID

class EditError(ValueError):pass

def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def _names(nodes):
    nodes=list(nodes);items=[x for n in nodes for x in ast.walk(n)]
    return dict(reads=sorted({x.id for x in items if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}),
        writes=sorted({x.id for x in items if isinstance(x,ast.Name) and isinstance(x.ctx,(ast.Store,ast.Del))}),
        calls=sorted({x.func.id for x in items if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)}),
        dynamic_risk=any(isinstance(x,ast.Call) and (not isinstance(x.func,ast.Name) or x.func.id in ('eval','exec','getattr','globals','locals','__import__')) for x in items))

def _scc(graph):
    order={};low={};stack=[];active=set();out=[]
    def visit(v):
        order[v]=low[v]=len(order);stack.append(v);active.add(v)
        for w in sorted(graph[v]):
            if w not in order:visit(w);low[v]=min(low[v],low[w])
            elif w in active:low[v]=min(low[v],order[w])
        if low[v]==order[v]:
            part=[]
            while True:
                w=stack.pop();active.remove(w);part.append(w)
                if w==v:break
            out.append(tuple(sorted(part)))
    for v in sorted(graph):
        if v not in order:visit(v)
    return out

def catalog(source,*,task,group,arm,episode,parent_sha,bounded=True):
    tree=ast.parse(source);lines=source.splitlines(keepends=True)
    funcs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    graph={n:set(_names([f])['calls']) & funcs.keys() for n,f in funcs.items()}
    components=_scc(graph);regions=[]
    def span(node):return [min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])]),node.end_lineno]
    def record(scope,nodes,owner=None):
        spans=[span(n) for n in nodes];raw=''.join(''.join(lines[a-1:b]) for a,b in spans)
        boundary=_names(nodes);boundary['outputs']=boundary['writes']
        boundary['parameters']=ast.unparse(funcs[owner].args) if owner in funcs else {n.name:ast.unparse(n.args) for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        return dict(scope=scope,spans=spans,region_sha=sha(raw),source=raw,ast_kinds=[type(n).__name__ for n in nodes],owner=owner,
            function_names=[n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],boundary=boundary,
            semantic_impact='UNIDENTIFIED_DYNAMIC' if boundary['dynamic_risk'] else 'STATIC_BOUNDARY_ONLY_NOT_SEMANTIC_PROOF')
    for name,f in funcs.items():
        for n in ast.walk(f):
            if isinstance(n,(ast.For,ast.AsyncFor,ast.While,ast.If,ast.With,ast.AsyncWith,ast.Try,ast.Match)):
                # Uncontained function exits and dynamic namespace mutation use larger C/G neighborhoods.
                if not any(isinstance(x,(ast.Return,ast.Yield,ast.YieldFrom,ast.Global,ast.Nonlocal)) for x in ast.walk(n)):
                    regions.append(record('B',[n],name))
            for _,body in ast.iter_fields(n):
                if not isinstance(body,list) or not body or not all(isinstance(x,ast.stmt) for x in body):continue
                def simple(x):
                    return isinstance(x,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Assert,ast.Delete)) and not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant))
                for width in (2,3):
                    for i in range(len(body)-width+1):
                        nodes=body[i:i+width]
                        if all(simple(x) for x in nodes):regions.append(record('B',nodes,name))
    sets=set(components)
    for comp in components:
        calls=set().union(*(graph[n] for n in comp))
        for helper in components:
            if helper!=comp and calls.intersection(helper):sets.add(tuple(sorted(set(comp)|set(helper))))
    for comp in sorted(sets):
        nodes=sorted((funcs[n] for n in comp),key=lambda n:n.lineno)
        if len(nodes)==len(tree.body):continue
        regions.append(record('C',nodes))
    unique={}
    for r in regions:
        if r['scope']=='B':
            a=min(s[0] for s in r['spans']);b=max(s[1] for s in r['spans'])
            r['spans']=[[a,b]];r['source']=''.join(lines[a-1:b]);r['region_sha']=sha(r['source'])
        unique[(r['scope'],tuple(map(tuple,r['spans'])))]=r
    result=[]
    for scope,cap in [('B',4),('C',3)]:
        pool=sorted((r for r in unique.values() if r['scope']==scope),key=lambda r:r['spans'])
        rng=random.Random(stable_seed(STUDY_ID,task,group,arm,episode,parent_sha,scope))
        result.extend(sorted(rng.sample(pool,min(cap,len(pool))),key=lambda r:r['spans']) if bounded else pool)
    result.append(dict(scope='G',spans=[[1,len(lines)]],region_sha=sha(source),source=source,function_names=[],boundary=_names(tree.body),ast_kinds=['Module'],owner=None,semantic_impact='FULL_PROGRAM'))
    return result

def menu(io,episode):
    programs=io.state['programs'];best=sorted(programs,key=lambda h:(programs[h]['mean'],h))[:4]
    recent=sorted((h for h in programs if h not in best),key=lambda h:(-programs[h]['first_index'],h))[:2]
    parents=best+recent;actions=[];records={}
    for h in parents:
        for region in catalog(io.source(h),task=io.task,group=io.group,arm=io.arm,episode=episode,parent_sha=h):
            operations=[('rewrite',None)]
            if region['scope']=='G' and len(parents)>1:
                donor=min((x for x in parents if x!=h),key=lambda x:stable_seed(STUDY_ID,io.phase,io.task,io.group,episode,h,x,'donor'))
                operations.append(('recombine:'+donor,donor))
            for op,donor in operations:
                aid=f'a{len(actions):03d}';a=Action(aid,h,region['region_sha'],region['scope'],op)
                actions.append(a);records[aid]=dict(region,parent_sha=h,operation=op,donor_sha=donor,action_id=aid,local_key=a.key)
    return actions,records

def extract(response):
    lines=response.splitlines(keepends=True);fences=[];opened=None;marker=None
    for i,line in enumerate(lines):
        match=re.match(r'^\s*(`{3,}|~{3,})([^\r\n]*)',line)
        if not match:continue
        if opened is None:
            opened=i;marker=match.group(1)
        elif match.group(1)[0]==marker[0] and len(match.group(1))>=len(marker) and not match.group(2).strip():
            fences.append(''.join(lines[opened+1:i]));opened=None
    if opened is not None or len(fences)>1:raise EditError('FORMAT_MULTIPLE_OR_UNCLOSED_FENCE')
    value=fences[0] if fences else response
    rows=value.splitlines(keepends=True)
    if not rows or rows[0].strip() not in ('# FUSE_REPLACE','# FUSE_FULL'):raise EditError('FORMAT_MODE_LINE_REQUIRED')
    return rows[0].strip()[7:],''.join(rows[1:])

def assemble(parent,record,response):
    mode,body=extract(response);original=ast.parse(parent);lines=parent.splitlines(keepends=True)
    if mode=='FULL':source=body;realized='G'
    else:
        if record['scope']=='G':raise EditError('ASSEMBLY_GLOBAL_REQUIRES_FULL')
        value=textwrap.dedent(body);spans=record['spans']
        if record['scope']=='B':
            a,b=spans[0];indent=re.match(r'^[ \t]*',lines[a-1]).group()
            replacement=''.join(indent+s if s.strip() else s for s in value.splitlines(keepends=True))
            if replacement and not replacement.endswith(('\n','\r')):replacement+='\n'
            source=''.join(lines[:a-1])+replacement+''.join(lines[b:])
        else:
            new=ast.parse(value);functions={n.name:n for n in new.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            if len(functions)!=len(new.body) or set(functions)!=set(record['function_names']):raise EditError('ASSEMBLY_FUNCTION_SET_REQUIRES_FULL')
            old={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};vlines=value.splitlines(keepends=True)
            edits=[]
            for name,(a,b) in zip(record['function_names'],spans):
                node=functions[name]
                if ast.dump(old[name].args)!=ast.dump(node.args) or type(old[name])!=type(node):raise EditError('ASSEMBLY_INTERFACE_CHANGE_REQUIRES_FULL')
                start=min([node.lineno]+[n.lineno for n in node.decorator_list]);replacement=''.join(vlines[start-1:node.end_lineno])
                if not replacement.endswith('\n'):replacement+='\n'
                edits.append((a,b,replacement))
            for a,b,replacement in sorted(edits,reverse=True):lines[a-1:b]=[replacement]
            source=''.join(lines)
        realized=record['scope']
    try:compile(source,'<candidate>','exec')
    except (SyntaxError,ValueError) as e:raise EditError('SYNTAX:'+str(e)) from e
    changed=[dict(parent_lines=[a+1,b],child_lines=[c+1,d],operation=tag) for tag,a,b,c,d in difflib.SequenceMatcher(a=parent.splitlines(),b=source.splitlines(),autojunk=False).get_opcodes() if tag!='equal']
    new_tree=ast.parse(source)
    def units(tree):
        return {f'{type(n).__name__}:{getattr(n,"name",i)}':ast.dump(n,include_attributes=False) for i,n in enumerate(tree.body)}
    old_units,new_units=units(original),units(new_tree)
    ast_support=[k for k in sorted(set(old_units)|set(new_units)) if old_units.get(k)!=new_units.get(k)]
    return source,dict(requested_scope=record['scope'],realized_scope=realized,mode=mode,escape=record['scope']!='G' and realized=='G',changed_spans=changed,
        ast_changed_top_level_units=ast_support,ast_structure_changed=bool(ast_support),region_sha=record['region_sha'],outside_region_bytes_preserved=mode=='REPLACE',semantic_impact='FULL_PROGRAM_UNIDENTIFIED_NONLOCAL' if mode=='FULL' else record['semantic_impact'])
