# launching batch processing functions go here
import logging, uuid, os, sys

if os.getenv('DEBUG'):
  import pdb

from functools import cache
from datetime import datetime, timezone, timedelta

from bajor.env_helpers import api_basic_username, api_basic_password, api_job_complete_url

import azure.batch.models as batchmodels
from azure.batch import BatchServiceClient
from azure.common.credentials import ServicePrincipalCredentials
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

def create_batch_job(job_id, manifest_container_path, pool_id):


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
    job = batchmodels.JobAddParameter(
        id=job_id,
        pool_info=batchmodels.PoolInformation(pool_id=pool_id),
        # setup the env variables for all tasks in the job
        common_environment_settings=[
            # specify the place we have setup the code that setups our catalogs and calls zoobot correctly
            # note: this dir contains a custom 'train_model_on_catalog.py' file copied from zoobot to our blob storage system
            # on each batch job run this file will be copied from the blob storage location to an execution location on the batch system
            # this setup allows us to quickly iterate on code changes on how we use zoobot withougt requiring a rebuild to the zoobot image
            # -- can be set by the bajor system CODE_FILE_PATH env var
            batchmodels.EnvironmentSetting(
                name='CODE_FILE_PATH',
                value=os.getenv('CODE_FILE_PATH', 'code/staging/train_model_on_catalog.py')),
            # specify the place we have setup the blob storage container to mount to
            # this is linked to how we built the batch system, see the batch system setup code in
            # https://github.com/zooniverse/panoptes-python-notebook/blob/master/examples/create_batch_pool_zoobot_staging.ipynb
            batchmodels.EnvironmentSetting(
                name='CONTAINER_MOUNT_DIR',
                value='training'),
            # set the manifest file path from the value supplied by the API
            batchmodels.EnvironmentSetting(
                name='MANIFEST_PATH',
                value=manifest_container_path),
            # set the mission catalog file path (defaults to decals 5 at the moment)
            # -- can be set by the bajor system MISSION_MANIFEST_PATH env var
            batchmodels.EnvironmentSetting(
                name='MISSION_MANIFEST_PATH',
                value=os.getenv('MISSION_MANIFEST_PATH', 'catalogues/decals_dr5/decals_dr5_ortho_catalog.parquet'))
        ],
        # set the on_all_tasks_complete option to 'terminateJob'
        # so the Job's status changes automatically after all submitted tasks are done
        # This is so that we do not take up the quota for active Jobs in the Batch account.
        on_all_tasks_complete=batchmodels.OnAllTasksComplete.terminate_job
    )

    # Batch Job lifecycle hooks setup
    # https://github.com/Azure-Samples/azure-batch-samples/blob/079a7d24b129bdd21a12efe81bdd54f0c1211aa3/Python/Batch/sample1_jobprep_and_release.py#L76-L86
    #
    # job preparation task can be used to download data / files used in the jobs tasks etc
    # for prediction workloads this can be used to pull the data down
    # from remote URLs to a local file system to be fed through the ML model
    #
    # job preparation task to
    # 1. create the job checkpoint results output directory on blob storage using a 0 byte file (mkdir -p makes this)
    # 2. copy the training code from blob storage to a shared job directory
    # see https://learn.microsoft.com/en-us/azure/batch/files-and-directories#root-directory-structure
    job.job_preparation_task = batchmodels.JobPreparationTask(
        command_line=f'/bin/bash -c \"set -ex; mkdir -p $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/{training_job_results_dir(job_id)}/checkpoints && cp $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/$CODE_FILE_PATH $AZ_BATCH_NODE_SHARED_DIR/"',
        #
        # A busted preparation task means the main task won't launch...ever!
        # and leave the node in a scaled state costing $$ ££
        #
        # Long term: perhaps add a background worker to check the
        # preparation task status and if it has failed then terminate the main job to avoid
        # leaving the node pool in the scaled state...
        #
        # Short term: avoid waiting for this prep task to complete before starting the main task
        # https://learn.microsoft.com/en-us/python/api/azure-batch/azure.batch.models.JobPreparationTask?view=azure-python#constructor
        # https://learn.microsoft.com/en-us/azure/batch/batch-job-task-error-checking#job-preparation-tasks
        wait_for_success=False)


    # add a callback to bajor to notify the job completed via a
    # Job release task that runs after the job completes
    #
    # TODO: use this to run the post training hooks
    #   e.g. 1.promote trained mode for use in prediction system
    #
    # longer term can be the hook system for a training run where
    # we promote best the zoobot model to a shared blob storage location
    # and do the job lifecycle management webhook to bajor
    job.job_release_task = batchmodels.JobReleaseTask(
        command_line=f'/bin/bash -c \"set -ex; echo "Job {job_id} has completed" > $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/{training_job_results_dir(job_id)}/job_release_task_output.txt\"')

    # use the job manager task to do something with the job information
    # and submit say job tasks to run, i.e. interpret a file and create a set of tasks from that file (think camera traps task batching)
    # job_manager_task = batchmodels.JobManagerTask(
    #     id="JobManagerTask",
    #     command_line=f'/bin/bash -c \"set -e; CMD_TO_DO_JOB_MANAGEMENT"',
    #     resource_files=[batchmodels.ResourceFile(
    #         file_path=_SIMPLE_TASK_NAME,
    #         http_url=sas_url)]))
    azure_batch_client().job.add(job)
    return job_id


def storage_container_sas_url():
    permissions = ContainerSasPermissions(read=True, write=True, list=True)
    access_duration_hrs = os.getenv('SAS_ACCESS_DURATION_HOURS', 12)
    storage_account_name = os.getenv(
        'STORAGE_ACCOUNT_NAME', 'kadeactivelearning')
    container_name = os.getenv('STORAGE_CONTAINER', 'training')
    container_sas_token = generate_container_sas(
        account_name=storage_account_name,
        container_name=container_name,
        account_key=os.getenv('STORAGE_ACCOUNT_KEY'),
        permission=permissions,
        expiry=datetime.utcnow() + timedelta(hours=access_duration_hrs))
    # construct the SAS token storate account URL
    return f'https://{storage_account_name}.blob.core.windows.net/{container_name}?{container_sas_token}'


def training_job_dir(job_id):
    # append a timestamp to the job blob storage dir to help us navigate the job history timeline
    job_submission_timestamp = datetime.now().isoformat(timespec='minutes')
    return f'jobs/{job_submission_timestamp}_{job_id}'


def training_job_results_dir(job_id):
  return f'{training_job_dir(job_id)}/results'


def training_job_logs_path(job_id, task_id, suffix):
  return f'{training_job_dir(job_id)}/task_logs/job_{job_id}_task_{task_id}_{suffix}.txt'


def create_job_tasks(job_id, task_id=1, run_opts=''):
    # for persisting stdout and stderr log files in container storage
    container_sas_url = storage_container_sas_url()
    # persist stdout and stderr (will be removed when node removed)
    # paths are relative to the Task working directory
    stderr_destination = batchmodels.OutputFileDestination(
        container=batchmodels.OutputFileBlobContainerDestination(
            container_url=container_sas_url,
            path=training_job_logs_path(
              job_id=job_id, task_id=task_id, suffix='stderr')
        )
    )
    stdout_destination = batchmodels.OutputFileDestination(
        container=batchmodels.OutputFileBlobContainerDestination(
            container_url=container_sas_url,
            path=training_job_logs_path(
                job_id=job_id, task_id=task_id, suffix='stdout')
        )
    )
    std_err_and_out = [
        batchmodels.OutputFile(
            file_pattern='../stderr.txt',  # stderr.txt is at the same level as wd
            destination=stderr_destination,
            upload_options=batchmodels.OutputFileUploadOptions(
                upload_condition=batchmodels.OutputFileUploadCondition.task_completion)
            # can also just upload on failure
        ),
        batchmodels.OutputFile(
            file_pattern='../stdout.txt',
            destination=stdout_destination,
            upload_options=batchmodels.OutputFileUploadOptions(
                upload_condition=batchmodels.OutputFileUploadCondition.task_completion)
        )
    ]

    tasks = []
    # ZOOBOT command for catalogue based training!
    # Note: Zoobot was baked into the conatiner the Azure batch VM
    # via the batch nodepool - see notes on panoptes-python-notebook
    # OR
    # TODO: add links to the Batch Scheduling system setup
    #       container for zoobot built in etc to show how this works
    #
    command = f'/bin/bash -c \"set -ex; python $AZ_BATCH_NODE_SHARED_DIR/train_model_on_catalog.py {run_opts} --experiment-dir $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/{training_job_results_dir(job_id)}/ --mission-catalog $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/$MISSION_MANIFEST_PATH --catalog $AZ_BATCH_NODE_MOUNTS_DIR/$CONTAINER_MOUNT_DIR/$MANIFEST_PATH\" '

    # test the cuda install (there is a built in script for this - https://github.com/mwalmsley/zoobot/blob/048543f21a82e10e7aa36a44bd90c01acd57422a/zoobot/pytorch/estimators/cuda_check.py)
    # command = '/bin/bash -c \'python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"\' '

    # create a job task to run the Zoobot training system command via the zoobot docker conatiner
    #
    task = batchmodels.TaskAddParameter(
        id=str(task_id),
        command_line=command,
        container_settings=batchmodels.TaskContainerSettings(
            image_name=os.getenv('CONTAINER_IMAGE_NAME'),
            working_directory='taskWorkingDirectory',
            container_run_options='--ipc=host'
        ),
        output_files=std_err_and_out
    )
    tasks.append(task)

    # return type: TaskAddCollectionResult
    collection_results = azure_batch_client().task.add_collection(job_id, tasks)

    failed_task_submission = False
    for task_result in collection_results.value:
        if task_result.status is not batchmodels.TaskAddStatus.success:
            log.debug(f'task {task_result.task_id} failed to submitted. '
                     f'status: {task_result.status}, error: {task_result.error}')
            failed_task_submission = True

    if failed_task_submission:
        return task_submission_status(state='error', message=task_result.error)
    else:
        return task_submission_status(state='submitted')


def task_submission_status(state, message='Job submitted successfully'):
    return {"status": state, "message": message}

def active_jobs_running():
  return len(get_batch_job_list()) > 0

def get_batch_job_list(job_list_options=batchmodels.JobListOptions(
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

def get_batch_job_status(job_id):
    # use the raw response object vs digging into the CloudJob resource for summary data
    # https://learn.microsoft.com/en-us/python/api/azure-batch/azure.batch.operations.joboperations?view=azure-python#azure-batch-operations-joboperations-get
    job_status = azure_batch_client().job.get(job_id, raw=True)
    return job_status.response.json()

    # longer term we can look at the job task lists as well
    # tasks = azure_batch_client().task.list(job_id)
    # task_list = [t for t in tasks]

# get a summary of what the batch service is up to
# https://learn.microsoft.com/en-us/python/api/azure-batch/azure.batch.operations.joboperations?view=azure-python#azure-batch-operations-joboperations-get-all-lifetime-statistics
def get_all_batch_job_stats():
    job_stats = azure_batch_client().job.get_all_lifetime_statistics
    return job_stats.response.json()

def schedule_job(job_id, manifest_path, run_opts=''):
    # Zoobot Azure Batch pool ID
    pool_id = os.getenv('POOL_ID', 'training_0')

    submitted_job_id = create_batch_job(
        job_id=job_id, manifest_container_path=manifest_path, pool_id=pool_id)
    job_task_submission_status = create_job_tasks(job_id=job_id, run_opts=run_opts)

    # return the submitted job_id and task submission status dict
    return { "submitted_job_id": submitted_job_id, "job_task_status": job_task_submission_status }


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.DEBUG,
        format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        stream = sys.stdout
    )

    # pdb.set_trace()
    # job_id = str(uuid.uuid4())
    # manifest_path = os.getenv('MANIFEST_PATH')
    # schedule_job(job_id, manifest_path)

