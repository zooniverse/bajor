from fastapi.testclient import TestClient
import uuid, os, pytest
from unittest import mock

fake_revision = str(uuid.uuid4())
submitted_job_id = 'fake-job-id'


@pytest.fixture
def mocked_client(autorun=True):
  """ Setup a TestClient API application with the mocked batch.py job scheduling function """
  with mock.patch('bajor.batch.schedule_job') as mock_schedule_job:

    result_set = {"submitted_job_id": submitted_job_id,
                  "job_task_status": {"status": "started", "message": "Job submitted successfully"}}

    mock_schedule_job.return_value = result_set

    # ensure we import this code after patching the batch#schedule_job function
    from bajor.main import app
    client = TestClient(app)

    yield client


@mock.patch.dict(os.environ, {"REVISION": fake_revision})
def test_read_main(mocked_client):
    response = mocked_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"revision": fake_revision}

def test_manifest_job_creation_without_auth_creds(mocked_client):
    response = mocked_client.post(
        "/jobs/",
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 401
    assert response.json() == { "detail": "Not authenticated"}


@mock.patch.dict(os.environ, {"BASIC_AUTH_USERNAME": 'bajor', "BASIC_AUTH_PASSWORD": 'bajor'})
def test_manifest_job_creation_incorrect_auth_creds(mocked_client):
    response = mocked_client.post(
        "/jobs/",
        auth=('test', 'test'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password"}

def test_manifest_job_creation_correct_auth_creds(mocked_client):
    response = mocked_client.post(
        "/jobs/",
        auth=('bajor', 'bajor'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 201
    assert response.json() == {
        'manifest_path': 'test_manifest_file_path.csv', 'id': submitted_job_id, 'status': {"status": "started", "message": "Job submitted successfully"}}

@mock.patch.dict(os.environ, {"BASIC_AUTH_USERNAME": 'rojab', "BASIC_AUTH_PASSWORD": 'rojab'})
def test_authentication_env_var_configs(mocked_client):
    response = mocked_client.post(
        "/jobs/",
        auth=('rojab', 'rojab'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 201
    assert response.json() == {
        'manifest_path': 'test_manifest_file_path.csv', 'id': submitted_job_id, 'status': {"status": "started", "message": "Job submitted successfully"}}


def test_batch_scheduling_code_is_called(mocked_client):
    response = mocked_client.post(
        "/jobs/",
        auth=('bajor', 'bajor'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 201
    assert response.json() == {
        'manifest_path': 'test_manifest_file_path.csv', 'id': submitted_job_id, 'status': {"status": "started", "message": "Job submitted successfully"}}
