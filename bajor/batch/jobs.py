# common batch job functions
import uuid
import os

from datetime import datetime, timedelta

import azure.batch.models as batchmodels
from azure.storage.blob import ContainerSasPermissions, generate_container_sas

from bajor.batch.client import azure_batch_client

def create_job_id():
    return str(uuid.uuid4())


def storage_container_sas_url(storage_container_name):
    permissions = ContainerSasPermissions(read=True, write=True, list=True)
    # make sure this is long enough to complete the job
    access_duration_hrs = os.getenv('SAS_ACCESS_DURATION_HOURS', 12)
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME', 'kadeactivelearning')
    container_name = storage_container_name,
    container_sas_token = generate_container_sas(
        account_name=storage_account_name,
        container_name=storage_container_name,
        account_key=os.getenv('STORAGE_ACCOUNT_KEY'),
        permission=permissions,
        expiry=datetime.utcnow() + timedelta(hours=access_duration_hrs))
    # construct the SAS token storate account URL
    return f'https://{storage_account_name}.blob.core.windows.net/{storage_container_name}?{container_sas_token}'


def job_submission_prefix(job_id):
    job_submission_timestamp = datetime.now().isoformat(timespec='minutes')
    return f'{job_submission_timestamp}_{job_id}'


def task_submission_status(state, message='Job submitted successfully'):
    return {"status": state, "message": message}


def active_jobs_running(pool_id):
    active_jobs = batchmodels.JobListOptions(
        filter=f"state eq 'active' and executionInfo/poolId eq '{pool_id}'",
        select='id'
    )
    num_active_jobs = len(get_batch_job_list(active_jobs))
    return num_active_jobs > 0


def get_non_active_batch_job_list(pool_id):
    return get_batch_job_list(batchmodels.JobListOptions(filter=f"executionInfo/poolId eq '{pool_id}' and state ne 'active'"))


def get_active_batch_job_list(pool_id):
    return get_batch_job_list(batchmodels.JobListOptions(filter=f"executionInfo/poolId eq '{pool_id}' and state eq 'active'"))


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
