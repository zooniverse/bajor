from bajor.batch import schedule_job, create_batch_job, create_job_tasks
import uuid, os
from unittest import mock

fake_job_id = str(uuid.uuid4())

@mock.patch('bajor.batch.active_jobs_running')
@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_active_jobs_running(mock_create_job_tasks, mock_create_batch_job, mock_active_jobs_running):
    mock_active_jobs_running.return_value = True
    schedule_job(fake_job_id)
    mock_active_jobs_running.assert_called_once()
    mock_create_batch_job.assert_not_called()
    mock_create_job_tasks.assert_not_called()


@mock.patch.dict(os.environ, {"MANIFEST_PATH": 'fake-manifest.csv'})
@mock.patch('bajor.batch.active_jobs_running')
@mock.patch('bajor.batch.create_batch_job')
@mock.patch('bajor.batch.create_job_tasks')
def test_no_active_jobs(mock_create_job_tasks, mock_create_batch_job, mock_active_jobs_running):
    mock_active_jobs_running.return_value = False
    schedule_job(fake_job_id)
    mock_active_jobs_running.assert_called_once()
    mock_create_batch_job.assert_called_once_with(job_id=fake_job_id, manifest_container_path='fake-manifest.csv', pool_id='gz_training_staging_0')
    mock_create_job_tasks.assert_called_once_with(job_id=fake_job_id)

# def test_creates_batch_jobs():
#     assert __version__ == '0.1.0'
