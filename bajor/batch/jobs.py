# common batch job functions
from datetime import datetime

import azure.batch.models as batchmodels

from bajor.batch.client import azure_batch_client

def job_submission_prefix(job_id):
    job_submission_timestamp = datetime.now().isoformat(timespec='minutes')
    return f'{job_submission_timestamp}_{job_id}'


def task_submission_status(state, message='Job submitted successfully'):
    return {"status": state, "message": message}


def active_jobs_running(pool_id):
    active_jobs = batchmodels.JobListOptions(
        filter=f"state eq 'active' AND executionInfo/poolId eq '{pool_id}'",
        select='id'
    )
    num_active_jobs = len(get_batch_job_list(active_jobs))
    return num_active_jobs > 0


def get_non_active_batch_job_list(pool_id):
    return get_batch_job_list(batchmodels.JobListOptions(filter="executionInfo/poolId eq '{pool_id}' AND state ne 'active'"))


def get_active_batch_job_list(pool_id):
    return get_batch_job_list(batchmodels.JobListOptions(filter="executionInfo/poolId eq '{pool_id}' AND state eq 'active'"))


def get_batch_job_list(job_list_options):
    jobs_generator = azure_batch_client().job.list(
        job_list_options=job_list_options)
    jobs_list = [j for j in jobs_generator]
    return jobs_list


def get_batch_job_status(job_id):
    # https://learn.microsoft.com/en-us/python/api/azure-batch/azure.batch.operations.joboperations
    job_status = azure_batch_client().job.get(job_id)

    return job_status.as_dict()


def get_batch_job_tasks(job_id):
    tasks = azure_batch_client().task.list(job_id)
    task_list = [t.as_dict() for t in tasks]

    return task_list


# get a summary of what the batch service is up to
# https://learn.microsoft.com/en-us/python/api/azure-batch/azure.batch.operations.joboperations?view=azure-python#azure-batch-operations-joboperations-get-all-lifetime-statistics
def get_all_batch_job_stats():
    job_stats = azure_batch_client().job.get_all_lifetime_statistics
    return job_stats.response.json()


def job_submission_response(submitted_job_id, job_task_status):
    return {"submitted_job_id": submitted_job_id, "job_task_status": job_task_status}
