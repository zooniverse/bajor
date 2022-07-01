import uvicorn
import os
import secrets
import uuid
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from honeybadger import contrib
from logging.config import dictConfig
from bajor.batch import schedule_job
from bajor.log_config import log_config


if os.getenv('DEBUG'):
  import pdb

dictConfig(log_config)

app = FastAPI()
# configure HB through the env vars https://docs.honeybadger.io/lib/python/#configuration
app.add_middleware(contrib.ASGIHoneybadger)
# add the http basic authorization mode
# https://fastapi.tiangolo.com/advanced/security/http-basic-auth/#simple-http-basic-auth
security = HTTPBasic()

class Job(BaseModel):
    manifest_path: str
    id: str | None
    status: str | None

def validate_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username, os.environ.get('BASIC_AUTH_USERNAME', 'bajor'))
    correct_password = secrets.compare_digest(
        credentials.password, os.environ.get('BASIC_AUTH_PASSWORD', 'bajor'))
    if not (correct_username and correct_password):
        return False
    else:
      return True

@app.post("/jobs/", status_code=status.HTTP_201_CREATED)
async def create_job(job: Job, authorized: bool = Depends(validate_basic_auth)):
  if not authorized:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Incorrect username or password",
      headers={"WWW-Authenticate": "Basic"},
    )
  # TODO: this is where we schedule ze job!
  # embedd the scheduling information into the Job model
  job_id = str(uuid.uuid4())

  results = schedule_job(job_id, job.manifest_path)
  job.id = results['submitted_job_id']
  job.status = results['job_task_status']

  return job

@app.get("/")
def root():
    return { "revision": os.environ.get('REVISION') }

def start_app(reload=False):
    uvicorn.run(
        "bajor.main:app",
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '8000')),
        reload=reload
    )

def start_dev_app():
    start_app(reload=True)
