from fastapi.testclient import TestClient
import uuid, os
from unittest import mock
from bajor.main import app

import pdb

client = TestClient(app)
fake_revision = str(uuid.uuid4())

@mock.patch.dict(os.environ, {"REVISION": fake_revision})
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"revision": fake_revision}


# how to setup authorization code for fast API (JWT)
# https://auth0.com/blog/build-and-secure-fastapi-server-with-auth0/
# how to skip auth for testing
# https: // stackoverflow.com/questions/71457448/how-can-i-skip-authentication-in-a-single-test-with-a-fixture-in-fast-api-togeth
# how to setup a create route
# https://fastapi.tiangolo.com/tutorial/testing/#extended-testing-file
# post the manifest path
# pass some basic auth creds
# test the response object
# and test the bajor interface is called
def test_manifest_job_creation_without_auth_creds():
    response = client.post(
        "/jobs/",
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 401
    assert response.json() == { "detail": "Not authenticated"}


@mock.patch.dict(os.environ, {"BASIC_AUTH_USERNAME": 'bajor', "BASIC_AUTH_PASSWORD": 'bajor'})
def test_manifest_job_creation_incorrect_auth_creds():
    response = client.post(
        "/jobs/",
        auth=('test', 'test'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Incorrect username or password"}

def test_manifest_job_creation_correct_auth_creds():
    response = client.post(
        "/jobs/",
        auth=('bajor', 'bajor'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 201
    assert response.json() == {
        'manifest_path': 'test_manifest_file_path.csv', 'job_id': None, 'scheduled': None}

@mock.patch.dict(os.environ, {"BASIC_AUTH_USERNAME": 'rojab', "BASIC_AUTH_PASSWORD": 'rojab'})
def test_authentication_env_var_configs():
    response = client.post(
        "/jobs/",
        auth=('rojab', 'rojab'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 201
    assert response.json() == {
        'manifest_path': 'test_manifest_file_path.csv', 'job_id': None, 'scheduled': None}


@mock.patch.dict(os.environ, {"BASIC_AUTH_USERNAME": 'rojab', "BASIC_AUTH_PASSWORD": 'rojab'})
def test_batch_code_is_called():
    response = client.post(
        "/jobs/",
        auth=('rojab', 'rojab'),
        json={"manifest_path": "test_manifest_file_path.csv"},
    )
    assert response.status_code == 201
    assert response.json() == {
        'manifest_path': 'test_manifest_file_path.csv', 'job_id': None, 'scheduled': None}
