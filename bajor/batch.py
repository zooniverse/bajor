# launching batch processing functions go here
import logging, uuid, os, sys

if os.getenv('DEBUG'):
  import pdb

from functools import cache
from datetime import datetime, timezone, timedelta

from azure.batch import BatchServiceClient
from azure.batch.models import *
from azure.common.credentials import ServicePrincipalCredentials
# ContainerClient
from azure.storage.blob import ContainerSasPermissions, generate_container_sas

log = logging.getLogger('BAJOR')

@cache
def azure_batch_client():
    credentials = ServicePrincipalCredentials(
        client_id=os.getenv('APP_CLIENT_ID'),
        secret=os.getenv('APP_CLIENT_SECRET'),
        tenant=os.getenv('APP_TENANT_ID'),
        resource='https://batch.core.windows.net/'
    )
    return BatchServiceClient(
        credentials=credentials,
        batch_url=os.getenv('BATCH_ACCOUNT_URL',
                            'https://zoobot.eastus.batch.azure.com')
    )

def get_utc_time():
    # return current UTC time as a string in the ISO 8601 format (so we can query by
    # timestamp in the Cosmos DB job status table.
    # example: '2021-02-08T20:02:05.699689Z'
    return datetime.now(timezone.utc).isoformat(timespec='microseconds') + 'Z'


def create_batch_job(job_id, manifest_container_path, pool_id):
    job_submission_timestamp = get_utc_time()

    log.debug('server_job, create_batch_job, using manifest from path: {}'.format(
        manifest_container_path))
    # TODO: add job completed / error reporting for job tracking data model (pgserver / redis / sqlite?)
    # or possibly use container storage entires or even maybe container table storage
    #   if len(image_paths) == 0:
    #       job_status = get_job_status(
    #           'completed', '0 images found in provided list of images.')
    #       job_status_table.update_job_status(job_id, job_status)
    #       return
    # and then job status like it's good etc
    # job_status = get_job_status('created', f'{num_images} images listed; submitting the job...')
    # job_status_table.update_job_status(job_id, job_status)

    log.debug(f'BatchJobManager, create_job, job_id: {job_id}')
    job = JobAddParameter(
        id=job_id,
        pool_info=PoolInformation(pool_id=pool_id),
        # setup the env variables for all tasks in the job
        common_environment_settings=[
            EnvironmentSetting(
                name='MANIFEST_PATH',
                value=manifest_container_path),
            EnvironmentSetting(
                name='CONTAINER_MOUNT_DIR',
                value='training_storage')
        ],
        # set the on_all_tasks_complete option to 'terminateJob'
        # so the Job's status changes automatically after all submitted tasks are done
        # This is so that we do not take up the quota for active Jobs in the Batch account.
        on_all_tasks_complete=OnAllTasksComplete.terminate_job
    )
    azure_batch_client().job.add(job)


def storage_container_sas_url():
    permissions = ContainerSasPermissions(read=True, write=True, list=True)
    access_duration_hrs = os.getenv('SAS_ACCESS_DURATION_HOURS', 12)
    storage_account_name = os.getenv(
        'STORAGE_ACCOUNT_NAME', 'kadeactivelearning')
    container_name = os.getenv('STORAGE_CONTAINER', 'staging')
    container_sas_token = generate_container_sas(
        account_name=storage_account_name,
        container_name=container_name,
        account_key=os.getenv('STORAGE_ACCOUNT_KEY'),
        permission=permissions,
        expiry=datetime.utcnow() + timedelta(hours=access_duration_hrs))
    # construct the SAS token storate account URL
    return f'https://{storage_account_name}.blob.core.windows.net/{container_name}?{container_sas_token}'

def training_job_results_dir(job_id):
  return f'training_jobs/job_{job_id}/results/'

def training_job_logs_path(job_id, task_id, suffix):
  return f'training_jobs/job_{job_id}/task_logs/job_{job_id}_task_{task_id}_{suffix}.txt'

def create_job_tasks(job_id, task_id=1):
    # for persisting stdout and stderr log files in container storage
    container_sas_url = storage_container_sas_url()
    # persist stdout and stderr (will be removed when node removed)
    # paths are relative to the Task working directory
    stderr_destination = OutputFileDestination(
        container=OutputFileBlobContainerDestination(
            container_url=container_sas_url,
            path=training_job_logs_path(
              job_id=job_id, task_id=task_id, suffix='stderr')
        )
    )
    stdout_destination = OutputFileDestination(
        container=OutputFileBlobContainerDestination(
            container_url=container_sas_url,
            path=training_job_logs_path(
                job_id=job_id, task_id=task_id, suffix='stdout')
        )
    )
    std_err_and_out = [
        OutputFile(
            file_pattern='../stderr.txt',  # stderr.txt is at the same level as wd
            destination=stderr_destination,
            upload_options=OutputFileUploadOptions(
                upload_condition=OutputFileUploadCondition.task_completion)
            # can also just upload on failure
        ),
        OutputFile(
            file_pattern='../stdout.txt',
            destination=stdout_destination,
            upload_options=OutputFileUploadOptions(
                upload_condition=OutputFileUploadCondition.task_completion)
        )
    ]

    tasks = []
    # ZOOBOT command for catalogue based training!
    # Note: Zoobot was baked into the conatiner the Azure batch VM
    # via the batch nodepool - see notes on panoptes-python-notebook
    # OR
    # TODO: add links to the Batch Scheduling system setup
    #       container for zoobot built in etc to show how this works
    # TODO: figure out how to avoid 0 byte file artifacts being created on storage conatiners
    #       from the mkdir cmd - maybe just write a file in the nested conatiner paths where it's needed?
    # TODO: ensure we pass the original decals 5 training catalog as well (staging will use a sample, production the whole shebang)
    command = f'/bin/bash -c \"mkdir -p $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/{training_job_results_dir(job_id)} && python /usr/src/zoobot/train_model_on_catalog.py --experiment-dir $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/{training_job_results_dir(job_id)} --epochs 3 --batch-size 5 --catalog $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/$MANIFEST_PATH\" '

    # test the cuda install (there is a built in script for this - https://github.com/mwalmsley/zoobot/blob/048543f21a82e10e7aa36a44bd90c01acd57422a/zoobot/pytorch/estimators/cuda_check.py)
    # command = '/bin/bash -c \'python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"\' '

    # create a job task to run the Zoobot training system command
    task = TaskAddParameter(
        id=str(task_id),
        command_line=command,
        container_settings=TaskContainerSettings(
            image_name=os.getenv('CONTAINER_IMAGE_NAME'),
            working_directory='taskWorkingDirectory'
        ),
        output_files=std_err_and_out
    )
    tasks.append(task)

    # return type: TaskAddCollectionResult
    collection_results = azure_batch_client().task.add_collection(job_id, tasks)

    for task_result in collection_results.value:
        if task_result.status is not TaskAddStatus.success:
            log.debug(f'task {task_result.task_id} failed to submitted. '
                     f'status: {task_result.status}, error: {task_result.error}')


def active_jobs_running():
  return len(get_batch_job_list()) > 0

def get_batch_job_list(job_list_options=JobListOptions(
    filter='state eq \'active\'',
    select='id'
)):
    jobs_generator = azure_batch_client().job.list(
        job_list_options=job_list_options)
    jobs_list = [j for j in jobs_generator]
    return jobs_list

def list_active_jobs():
  log.debug('Active batch jobs list')
  log.debug(get_batch_job_list())

def schedule_job(job_id):
    # Zoobot Azure Batch pool ID
    pool_id = os.getenv('POOL_ID', 'gz_training_staging_0')

    # TODO: allow this manifest path to be set via an API query / post param
    manifest_path = os.getenv(
        'MANIFEST_PATH', 'training_catalogues/workflow-3598-2022-06-24T14:18:16+00:00.csv')

    if active_jobs_running():
      log.debug(
          'Active Jobs are running in the batch system - please wait till they are fininshed processing.')
    else:
      log.debug('No active jobs running - lets get scheduling!')
      # create_batch_job(
      #     job_id=job_id, manifest_container_path=manifest_path, pool_id=pool_id)
      # create_job_tasks(job_id=job_id)


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.DEBUG,
        format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        stream = sys.stdout
    )

    job_id = str(uuid.uuid4())
    schedule_job(job_id)

