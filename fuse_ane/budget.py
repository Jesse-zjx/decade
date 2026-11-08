"""Settled-response prefix accounting, reusing the existing cost boundaries."""
from fuse_patch.budget import TokenBoundary as Base,DispatchStop
from fuse_addon.evaluation import freeze,read,digest,encoded
import copy

def token_count(cost):
    """Operational known-usage budget only; exact cost is never imputed."""
    return cost.get("tokens_known",0)

from .runtime import stop_dispatch

class Boundary(Base):
    def __init__(self,search):
        from .data import PREFIXES
        target=160000 if search.arm=='H' or search.arm in ('F2','F1','N2') else 80000
        if search.phase=='confirm' and search.arm in ('F2','N2'):
            target=read(search.study/'CONFIRMATION_FREEZE.json')['budget_tokens']
        super().__init__(search,target=target,position_cap=256 if search.arm in ('F2','F1','N2') else 100000)
        self.prefixes=PREFIXES[search.arm]
        search.transport._budget_boundary=self

    def before_dispatch(self,*,generation=False,repair=False):
        from .runtime import STUDY
        if (STUDY/'STOP_DISPATCH.json').exists():raise DispatchStop('DRAINING_NO_NEW_INTENTS')
        tokens=token_count(self.search.cost_snapshot())
        if tokens is None:
            stop_dispatch('UNKNOWN_USAGE_NO_NEW_DISPATCH');raise DispatchStop('UNKNOWN_USAGE')
        if tokens>=self.target:raise DispatchStop('TOKEN_TARGET_REACHED')
        # Repairs belong to the current normal episode, including episode 128.
        if self.position>=self.budget and not repair:raise DispatchStop('POSITION_CAP_REACHED')
        cap=384 if self.search.arm in ('F2','F1','N2') else 256 if self.search.arm in ('H','E') else 160
        if self.search.state['feedback_attempts']>=cap:raise DispatchStop('FEEDBACK_ATTEMPT_CAP')
    def observe(self,kind,**detail):
        if kind=='response' and detail.get('candidate_response'):return None
        self.event_counter += 1
        from fuse_patch.budget import selection
        cost=self.search.cost_snapshot();tokens=token_count(cost)
        row=dict(event=self.event_counter,boundary=kind,position=self.position,pending_position=self._pending_position,
            selection=selection(self.search),archive_hashes=sorted(self.search.state.get('programs',{})),cost=cost,**detail)
        freeze(self.search.root/'budget_boundaries'/f'{self.event_counter:06d}.json',row)
        for threshold in self.prefixes:
            if threshold not in self.prefix_results and not cost.get('tokens_missing') and tokens>=threshold:
                prefix=dict(threshold_tokens=threshold,status='REACHED',reached=True,first_boundary_event=self.event_counter,
                    boundary=kind,completed_positions=self.position,pending_position=self._pending_position,selection=row['selection'],
                    search_best_hash=row['selection']['source_hash'],search_best_mean=row['selection']['mean'],cost=cost,
                    archive_hashes=row['archive_hashes'],search_locked=True,actual_tokens=cost.get('tokens'),known_tokens=tokens,
                    overshoot_tokens=None if cost.get('tokens_missing') else tokens-threshold,
                    known_overshoot_tokens=tokens-threshold,budget_semantics=cost.get('budget_semantics'),
                    **self.request_set())
                freeze(self.search.root/'prefixes'/str(threshold)/'PREFIX.json',prefix);self.prefix_results[threshold]=prefix
        return row

    def settled_failed_attempt(self, result):
        """Freeze physical failed-request crossings before any transient resend."""
        self.observe('settled_failed_physical_request',physical_request_id=result.get('logical_id'),
            request_path=result.get('request_path'),candidate_generated=False)

    def retry_budget_available(self):
        return token_count(self.search.cost_snapshot()) < self.target

    def request_set(self):
        shared=self.search._shared or {}
        requests=copy.deepcopy(shared.get('request_set', [])); covered=set()
        for row in self.search.transport.logical_records:
            attempts=list(row['attempts']); covered.update(attempts)
            requests.append(dict(owner=self.search.arm,logical_id=row['logical_id'],metadata=copy.deepcopy(row.get('metadata',{})),
                physical_attempts=attempts,selected_attempt=row['transport_attempt'],selected_before_candidate_parse=row['selected_before_candidate_parse']))
        # A known paid failure can cross a prefix before the logical operation's
        # retry decision. Preserve only the physical attempts already settled.
        pending={}
        for row in self.search.transport.records:
            path=row.get('request_path')
            if not path or path in covered:continue
            rid=row.get('generation_id') or row['logical_id'].rsplit('/attempt-',1)[0]
            item=pending.setdefault(rid,dict(owner=self.search.arm,logical_id=rid,
                metadata=copy.deepcopy(row.get('metadata',{})),physical_attempts=[],
                selected_attempt=None,selected_before_candidate_parse=True,physical_frontier_only=True))
            if path not in item['physical_attempts']:item['physical_attempts'].append(path)
        requests.extend(pending.values())
        return dict(request_set=requests,request_set_sha256=digest(encoded(requests)),request_set_complete=True,
            report_owner=self.search.arm,own_logical_requests=sum(r['owner']==self.search.arm for r in requests))

    def inherit_prefixes(self,shared):
        for raw,prefix in shared.get('prefixes',{}).items():
            point=int(raw)
            if point not in self.prefixes or prefix.get('reached') is not True:continue
            inherited=dict(copy.deepcopy(prefix),inherited_shared_initialization=True,
                shared_initialization_sha256=digest(encoded(shared)),report_owner=self.search.arm,own_logical_requests=0)
            freeze(self.search.root/'prefixes'/str(point)/'PREFIX.json',inherited)
            self.prefix_results[point]=inherited

    def finish(self,*args,**kwargs):
        cost=self.search.cost_snapshot()
        if cost.get('tokens_missing'):
            for point in self.prefixes:
                if point not in self.prefix_results:
                    p=dict(threshold_tokens=point,status='UNKNOWN_CROSSING',reached=None,
                        reason='MISSING_USAGE_FIRST_CROSSING_UNIDENTIFIED',cost=None,
                        selection=None,archive_hashes=[],quality_work_preserved_in_final_archive=True)
                    freeze(self.search.root/'prefixes'/str(point)/'PREFIX.json',p)
                    self.prefix_results[point]=p
        result=super().finish(*args,**kwargs)
        provenance=self.request_set()
        freeze(self.search.root/'ENDPOINT_REQUEST_SET.json',provenance)
        return dict(result,**provenance)
