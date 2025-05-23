import os

from fastapi import Depends, APIRouter, HTTPException, status, Response

from bajor.models.job import PredictionJob
from bajor.log_config import log
from bajor.apis.basic_auth import validate_basic_auth

import bajor.batch.predictions as predictions
import bajor.batch.jobs as batch_jobs

if os.getenv('DEBUG'):
  import pdb

# router prefix for prediction jobs api
# https://fastapi.tiangolo.com/tutorial/bigger-applications/#import-apirouter
router = APIRouter(
    prefix="/prediction",
)


@router.post("/jobs/", status_code=status.HTTP_201_CREATED)
async def create_job(job: PredictionJob, response: Response, authorized: bool = Depends(validate_basic_auth)) -> PredictionJob:
    if not authorized:
      raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
      )

    job_id = batch_jobs.create_job_id()

    if predictions.active_jobs_running():
      msg = 'Active Jobs are running in the batch system - please wait till they are fininshed processing.'
      log.debug(msg)
      response.status_code = status.HTTP_409_CONFLICT
      return { "state": "error", "message": msg }
    else:
      log.debug('No active jobs running - lets get scheduling!')
      results = predictions.schedule_job(
          job_id, job.manifest_url, job.opts)
      job.id = results['submitted_job_id']
      job.status = results['job_task_status']

      return job


@router.get("/jobs/", status_code=status.HTTP_200_OK)
async def list_jobs(response: Response, active: bool = True, authorized: bool = Depends(validate_basic_auth)):
    if not authorized:
      raise HTTPException(
          status_code=status.HTTP_401_UNAUTHORIZED,
          detail="Incorrect username or password",
          headers={"WWW-Authenticate": "Basic"},
      )

    log.debug('Fetching job list from batch service')
    return predictions.get_active_batch_job_list() if active else predictions.get_non_active_batch_job_list()


@router.get("/job/{job_id}", status_code=status.HTTP_200_OK)
async def get_job_by_id(job_id: str, response: Response, include_tasks: bool = True, authorized: bool = Depends(validate_basic_auth)):
    if not authorized:
      raise HTTPException(
          status_code=status.HTTP_401_UNAUTHORIZED,
          detail="Incorrect username or password",
          headers={"WWW-Authenticate": "Basic"},
      )

    log.debug(f'Job status for id: {job_id}')
    job_status = batch_jobs.get_batch_job_status(job_id)

    if include_tasks:
      log.debug(f'Task stats for job id: {job_id}')
      job_status['tasks'] = batch_jobs.get_batch_job_tasks(job_id)

    return job_status
