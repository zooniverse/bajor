import pdb
from bajor.batch import schedule_job, create_batch_job, create_job_tasks
import uuid, os
from unittest import mock

fake_job_id = str(uuid.uuid4())

@mock.patch('bajor.batch.active_jobs_running')
@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_active_jobs_running(mock_create_job_tasks, mock_create_batch_job, mock_active_jobs_running):
    mock_active_jobs_running.return_value = True
    schedule_job(fake_job_id, 'fake-manifest.csv')
    mock_active_jobs_running.assert_called_once()
    mock_create_batch_job.assert_not_called()
    mock_create_job_tasks.assert_not_called()


@mock.patch('bajor.batch.active_jobs_running')
@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_no_active_jobs(mock_create_job_tasks, mock_create_batch_job, mock_active_jobs_running):
    mock_active_jobs_running.return_value = False
    schedule_job(fake_job_id, 'fake-manifest.csv')
    mock_active_jobs_running.assert_called_once()
    mock_create_batch_job.assert_called_once_with(job_id=fake_job_id, manifest_container_path='fake-manifest.csv', pool_id='gz_training_staging_0')
    mock_create_job_tasks.assert_called_once_with(job_id=fake_job_id)


@mock.patch('bajor.batch.active_jobs_running')
@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_schedule_job(mock_create_job_tasks, mock_create_batch_job, mock_active_jobs_running):
    mock_active_jobs_running.return_value = False
    submitted_job_id = 'fake-job-id'
    job_task_status = {"task_submission_status": {"status": 'submitted', "message": 'job has been submitted for processing'}}
    mock_create_batch_job.return_value = submitted_job_id
    mock_create_job_tasks.return_value = job_task_status
    expected_result_dict = {
        "submitted_job_id": submitted_job_id, "job_task_status": job_task_status}
    result_dict = schedule_job(submitted_job_id, 'fake-manifest-path.csv')
    assert(result_dict) == expected_result_dict

