import uvicorn
import os
import secrets
import uuid
import logging
from fastapi import Depends, FastAPI, HTTPException, status, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from honeybadger import contrib
from logging.config import dictConfig
from bajor.training.batch import schedule_job, active_jobs_running
from bajor.log_config import log_config

if os.getenv('DEBUG'):
  import pdb

# setup the logger
dictConfig(log_config)
log = logging.getLogger('BAJOR')

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
    run_opts: str = ''

def validate_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username, os.environ.get('BASIC_AUTH_USERNAME', 'bajor'))
    correct_password = secrets.compare_digest(
        credentials.password, os.environ.get('BASIC_AUTH_PASSWORD', 'bajor'))
    if not (correct_username and correct_password):
        return False
    else:
      return True


def active_batch_jobs():
  return active_jobs_running()

@app.post("/jobs/", status_code=status.HTTP_201_CREATED)
async def create_job(job: Job, response: Response, authorized: bool = Depends(validate_basic_auth)) -> Job:
    if not authorized:
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
      )

    job_id = str(uuid.uuid4())

    if active_batch_jobs():
      msg = 'Active Jobs are running in the batch system - please wait till they are fininshed processing.'
      log.debug(msg)
      response.status_code = status.HTTP_409_CONFLICT
      return { "state": "error", "message": msg }
    else:
      log.debug('No active jobs running - lets get scheduling!')
      results = schedule_job(job_id, job.manifest_path, job.run_opts)
      job.id = results['submitted_job_id']
      job.status = results['job_task_status']

      return job

@app.get("/")
def root():
    return { "revision": os.environ.get('REVISION') }

def start_app(reload=False):
    uvicorn.run(
        "bajor.api:app",
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '8000')),
        reload=reload
    )

def start_dev_app():
    start_app(reload=True)
