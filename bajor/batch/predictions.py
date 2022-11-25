# training job specific functions
import logging, os, sys

if os.getenv('DEBUG'):
  import pdb

import azure.batch.models as batchmodels

from bajor.batch.client import azure_batch_client
import bajor.batch.jobs as batch_jobs
from bajor.log_config import log

# Zoobot Azure Batch predictions pool ID
predictions_pool_id = os.getenv('POOL_ID', 'predictions_0')

# wrapper functions to isolate the jobs to the training pool
def active_jobs_running():
    return batch_jobs.active_jobs_running(predictions_pool_id)

def get_active_batch_job_list():
  return batch_jobs.get_active_batch_job_list(predictions_pool_id)

def get_non_active_batch_job_list():
  return batch_jobs.get_non_active_batch_job_list(predictions_pool_id)

# schedule a training job
def schedule_job(job_id, manifest_url, run_opts=''):
    submitted_job_id = create_batch_job(
        job_id=job_id, manifest_url=manifest_url, pool_id=predictions_pool_id)
    job_task_submission_status = create_job_tasks(
        job_id=job_id, run_opts=run_opts)

    # return the submitted job_id and task submission status dict
    return batch_jobs.job_submission_response(submitted_job_id, job_task_submission_status)


def create_batch_job(job_id, manifest_url, pool_id):
    log.debug('server_job, create_batch_job, using manifest at url: {}'.format(manifest_url))

    log.debug(f'BatchJobManager, create_job, job_id: {job_id}')
    job = batchmodels.JobAddParameter(
        id=job_id,
        pool_info=batchmodels.PoolInformation(pool_id=pool_id),
        # setup the env variables for all tasks in the job
        common_environment_settings=[
            batchmodels.EnvironmentSetting(
                name='CODE_DIR_PATH',
                value=os.getenv('CODE_DIR_PATH', 'code')),
            # specify the place we have setup the blob storage container to mount to
            # this is linked to how we built the batch system, see the batch system setup code in
            # https://github.com/zooniverse/bajor/tree/main/azure/batch#create-a-azure-batch-compute-nodepool
            batchmodels.EnvironmentSetting(
                name='PREDICTIONS_CONTAINER_MOUNT_DIR',
                value='predictions'),
            # the models storage container mount dir
            batchmodels.EnvironmentSetting(
                name='MODELS_CONTAINER_MOUNT_DIR',
                value='models'),
            # set the manifest file path from the value supplied by the API
            batchmodels.EnvironmentSetting(
                name='MANIFEST_URL',
                value=manifest_url),
            # set the training results dir path
            batchmodels.EnvironmentSetting(
                name='PREDICTIONS_JOB_RESULTS_DIR',
                value=job_results_dir(job_id))
        ],
        # set the on_all_tasks_complete option to 'terminateJob'
        # so the Job's status changes automatically after all submitted tasks are done
        # This is so that we do not take up the quota for active Jobs in the Batch account.
        on_all_tasks_complete=batchmodels.OnAllTasksComplete.terminate_job
    )

    # job preparation task
    create_results_dir = f'mkdir -p $AZ_BATCH_NODE_MOUNTS_DIR/$PREDICTIONS_CONTAINER_MOUNT_DIR/$PREDICTIONS_JOB_RESULTS_DIR'
    copy_code_to_shared_dir = 'cp -Rf $AZ_BATCH_NODE_MOUNTS_DIR/$PREDICTIONS_CONTAINER_MOUNT_DIR/$CODE_DIR_PATH/* $AZ_BATCH_NODE_SHARED_DIR/'
    job.job_preparation_task = batchmodels.JobPreparationTask(
        command_line=f'/bin/bash -c \"set -ex; {create_results_dir}; {copy_code_to_shared_dir}\"',
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


    # Job release task that runs after the job completes
    # NOTE: job release tasks are hard to debug as you can't easily extract the logs :(
    #       like you can for the main task runs. Moving this code to the main task for now for logging purposes
    # promote trained model to a blob storage location for use in prediction system
    # promote_model_code_path = os.getenv('ZOOBOT_PROMOTE_CMD', 'promote_best_checkpoint_to_model.sh')
    # promote_checkpoint_cmd = f'$AZ_BATCH_NODE_SHARED_DIR/{promote_model_code_path} $AZ_BATCH_NODE_MOUNTS_DIR/$TRAINING_CONTAINER_MOUNT_DIR/$TRAINING_JOB_RESULTS_DIR'
    # job.job_release_task = batchmodels.JobReleaseTask(command_line=f'/bin/bash -c \"set -ex; {promote_checkpoint_cmd}\"'

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


# TODO: extract these as they are all common and isolated to their mounted containers
def job_dir(job_id):
    # append a timestamp to the job blob storage dir to help us navigate the job history timeline
    return f'jobs/{batch_jobs.job_submission_prefix(job_id)}'

def job_results_dir(job_id):
  return f'{job_dir(job_id)}/results'


def job_logs_path(job_id, task_id, suffix):
  return f'{job_dir(job_id)}/task_logs/job_{job_id}_task_{task_id}_{suffix}.txt'


def create_job_tasks(job_id, task_id=1, run_opts=''):
    # for persisting stdout and stderr log files in container storage
    container_sas_url = batch_jobs.storage_container_sas_url(
        os.getenv('PREDICTIONS_STORAGE_CONTAINER', 'predictions'))
    # persist stdout and stderr (will be removed when node removed)
    # paths are relative to the Task working directory
    stderr_destination = batchmodels.OutputFileDestination(
        container=batchmodels.OutputFileBlobContainerDestination(
            container_url=container_sas_url,
            path=job_logs_path(job_id=job_id, task_id=task_id, suffix='stderr')
        )
    )
    stdout_destination = batchmodels.OutputFileDestination(
        container=batchmodels.OutputFileBlobContainerDestination(
            container_url=container_sas_url,
            path=job_logs_path(job_id=job_id, task_id=task_id, suffix='stdout')
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
    # ZOOBOT command for catalogue predictions!
    # see jobPreparation task for code setup
    prediction_code_path = os.getenv('ZOOBOT_PREDICTION_CMD', 'predict_catalog_with_model.py')
    prediction_cmd = f'$AZ_BATCH_NODE_SHARED_DIR/{prediction_code_path} {run_opts} --checkpoint-path $AZ_BATCH_NODE_MOUNTS_DIR/$MODELS_CONTAINER_MOUNT_DIR/zoobot.ckpt --catalog-url $MANIFEST_URL --save-path $AZ_BATCH_NODE_MOUNTS_DIR/$PREDICTIONS_CONTAINER_MOUNT_DIR/$PREDICTIONS_JOB_RESULTS_DIR/predictions.csv'
    # redirect the stdout to stderr for logging
    command = f'/bin/bash -c \"set -ex; python {prediction_cmd}\"'

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
        return batch_jobs.task_submission_status(state='error', message=task_result.error)
    else:
        return batch_jobs.task_submission_status(state='submitted')


if __name__ == '__main__':
    logging.basicConfig(
        level = logging.DEBUG,
        format = '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        stream = sys.stdout
    )

