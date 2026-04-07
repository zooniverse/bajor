import bajor.batch.train_from_scratch as train_from_scratch
import bajor.batch.train_finetuning as train_finetuning
import bajor.batch.predictions as predictions
from bajor.batch.jobs import active_jobs_running
import uuid, os
from unittest import mock
from bajor.models.job import Options

fake_job_id = str(uuid.uuid4())
test_pool = 'pool'

@mock.patch('bajor.batch.jobs.get_batch_job_list')
def test_active_jobs_running(mock_get_batch_job_list):
    mock_get_batch_job_list.return_value = ["FakeJob"]
    assert(active_jobs_running(test_pool) == True)


@mock.patch('bajor.batch.train_from_scratch.create_batch_job')
@mock.patch('bajor.batch.train_from_scratch.create_job_tasks')
def test_no_active_jobs(mock_create_job_tasks, mock_create_batch_job):
    train_from_scratch.schedule_job(fake_job_id, 'fake-manifest.csv')
    mock_create_batch_job.assert_called_once_with(job_id=fake_job_id, manifest_container_path='fake-manifest.csv', pool_id='training_1')
    mock_create_job_tasks.assert_called_once_with(job_id=fake_job_id, run_opts='')


@mock.patch('bajor.batch.train_from_scratch.create_batch_job')
@mock.patch('bajor.batch.train_from_scratch.create_job_tasks')
def test_schedule_job(mock_create_job_tasks, mock_create_batch_job):
    submitted_job_id = 'fake-job-id'
    job_task_status = {"status": 'submitted',
        "message": 'Job submitted successfully'}
    mock_create_batch_job.return_value = submitted_job_id
    mock_create_job_tasks.return_value = job_task_status
    expected_result_dict = {
        "submitted_job_id": submitted_job_id, "job_task_status": job_task_status}
    result_dict = train_from_scratch.schedule_job(submitted_job_id, 'fake-manifest-path.csv')
    assert(result_dict) == expected_result_dict


@mock.patch('bajor.batch.train_finetuning.create_batch_job')
@mock.patch('bajor.batch.train_finetuning.create_job_tasks')
def test_no_active_jobs(mock_create_job_tasks, mock_create_batch_job):
    train_finetuning.schedule_job(fake_job_id, 'fake-manifest.csv')
    mock_create_batch_job.assert_called_once_with(
        job_id=fake_job_id, manifest_container_path='fake-manifest.csv', pool_id='training_1', options=Options())
    mock_create_job_tasks.assert_called_once_with(
        job_id=fake_job_id, options=Options())


@mock.patch('bajor.batch.train_finetuning.create_batch_job')
@mock.patch('bajor.batch.train_finetuning.create_job_tasks')
def test_schedule_job(mock_create_job_tasks, mock_create_batch_job):
    submitted_job_id = 'fake-job-id'
    job_task_status = {"status": 'submitted',
                       "message": 'Job submitted successfully'}
    mock_create_batch_job.return_value = submitted_job_id
    mock_create_job_tasks.return_value = job_task_status
    expected_result_dict = {
        "submitted_job_id": submitted_job_id, "job_task_status": job_task_status}
    result_dict = train_finetuning.schedule_job(
        submitted_job_id, 'fake-manifest-path.csv')
    assert(result_dict) == expected_result_dict


@mock.patch('bajor.batch.predictions.create_batch_job')
@mock.patch('bajor.batch.predictions.create_job_tasks')
def test_prediction_schedule_job_uses_options(mock_create_job_tasks, mock_create_batch_job):
    options = Options(
        prediction_script_path='predict_catalog_with_model.py',
        pretrained_checkpoint_url='custom.ckpt'
    )

    predictions.schedule_job(fake_job_id, 'https://manifest-host/predictions.json', options)

    mock_create_batch_job.assert_called_once_with(
        job_id=fake_job_id,
        manifest_url='https://manifest-host/predictions.json',
        pool_id='predictions_0',
        options=options
    )
    mock_create_job_tasks.assert_called_once_with(job_id=fake_job_id, options=options)
