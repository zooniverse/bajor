from bajor.batch import schedule_job, create_batch_job, create_job_tasks, active_jobs_running
import uuid, os
from unittest import mock

fake_job_id = str(uuid.uuid4())

@mock.patch('bajor.batch.get_batch_job_list')
def test_active_jobs_running(mock_get_batch_job_list):
    mock_get_batch_job_list.return_value = ["FakeJob"]
    assert(active_jobs_running() == True)

@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_no_active_jobs(mock_create_job_tasks, mock_create_batch_job):
    schedule_job(fake_job_id, 'fake-manifest.csv')
    mock_create_batch_job.assert_called_once_with(job_id=fake_job_id, manifest_container_path='fake-manifest.csv', pool_id='gz_training_staging_0')
    mock_create_job_tasks.assert_called_once_with(job_id=fake_job_id)

@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_schedule_job(mock_create_job_tasks, mock_create_batch_job):
    submitted_job_id = 'fake-job-id'
    job_task_status = {"status": 'submitted',
        "message": 'Job submitted successfully'}
    mock_create_batch_job.return_value = submitted_job_id
    mock_create_job_tasks.return_value = job_task_status
    expected_result_dict = {
        "submitted_job_id": submitted_job_id, "job_task_status": job_task_status}
    result_dict = schedule_job(submitted_job_id, 'fake-manifest-path.csv')
    assert(result_dict) == expected_result_dict

