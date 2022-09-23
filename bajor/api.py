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
from bajor.batch.training import schedule_job, active_jobs_running, get_batch_job_status, get_active_batch_job_list, get_non_active_batch_job_list
from bajor.env_helpers import api_basic_username, api_basic_password, revision, host, port
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
        credentials.username, api_basic_username())
    correct_password = secrets.compare_digest(
        credentials.password, api_basic_password())
    if not (correct_username and correct_password):
        return False
    else:
      return True

# Main app end points
@app.get("/")
async def root():
    return {"revision": revision()}

# SubAPI for training jobs on /training path
# https://fastapi.tiangolo.com/advanced/sub-applications/
training_app = FastAPI()

@training_app.post("/jobs/", status_code=status.HTTP_201_CREATED)
async def create_job(job: Job, response: Response, authorized: bool = Depends(validate_basic_auth)) -> Job:
    if not authorized:
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
      )

    job_id = str(uuid.uuid4())

    if active_jobs_running():
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


@training_app.get("/jobs/", status_code=status.HTTP_200_OK)
async def list_jobs(response: Response, active: bool = True, authorized: bool = Depends(validate_basic_auth)):
    if not authorized:
      raise HTTPException(
          status_code=status.HTTP_401_UNAUTHORIZED,
          detail="Incorrect username or password",
          headers={"WWW-Authenticate": "Basic"},
      )

    log.debug('Fetching job list from batch service')
    return get_active_batch_job_list() if active else get_non_active_batch_job_list()


@training_app.get("/job/{job_id}", status_code=status.HTTP_200_OK)
async def get_job_by_id(job_id: str, response: Response, authorized: bool = Depends(validate_basic_auth)):
    if not authorized:
      raise HTTPException(
          status_code=status.HTTP_401_UNAUTHORIZED,
          detail="Incorrect username or password",
          headers={"WWW-Authenticate": "Basic"},
      )

    log.debug(f'Job status for id: {job_id}')
    return get_batch_job_status(job_id)

# mount the subapi at a path
app.mount("/training", training_app)


def start_app(reload=False):
    uvicorn.run(
        "bajor.api:app",
        host=host(),
        port=int(port()),
        reload=reload
    )


def start_dev_app():
    start_app(reload=True)
