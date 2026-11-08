"""R2 narrowly scoped transport classification; no model or candidate invocation."""
import json,time
from pathlib import Path
from .r1_transport import retryable as original_retryable
SCOPES={f'confirm/tsp/{g}/{a}' for g in ('c08','c09') for a in ('F2','N2','H')}
MESSAGE='The model is overloaded. Please try again later.'
def normalize(s):return ' '.join(s.split()).casefold() if isinstance(s,str) else None
def require_scope(logical,study='fuse-shared-edit-siblings-20261007'):
 scope=logical.removeprefix(study+'/').rsplit('/',1)[0]
 if scope not in SCOPES:raise RuntimeError('R2_SEARCH_WRITE_SCOPE')
 return scope
def retryable(folder,result,logical):
 require_scope(logical)
 if result.get('http_status')!=400:return original_retryable(folder,result)
 if result.get('fatal_reason') or result.get('service_success') or not result.get('body_complete'):return False
 try:
  raw=(Path(folder)/'response.body').read_text()
  try:
   obj=json.loads(raw)
   if not isinstance(obj,dict):return False
   error=obj.get('error',obj)
   message=error.get('message') if isinstance(error,dict) else None
  except json.JSONDecodeError:message=raw
 except (OSError,UnicodeError):return False
 return normalize(message)==normalize(MESSAGE)
def retry_wait(result):
 value=result.get('retry_after');seconds=20.
 try:
  x=float(value)
  if x>=0 and x!=float('inf'):seconds=max(seconds,x)
 except (ValueError,TypeError):
  try:
   from email.utils import parsedate_to_datetime
   seconds=max(seconds,parsedate_to_datetime(value).timestamp()-time.time())
  except (ValueError,TypeError,OverflowError):pass
 return seconds
