from fastapi.testclient import TestClient
import uuid, os
from unittest import mock
from bajor.main import app

client = TestClient(app)
fake_revision = str(uuid.uuid4())

@mock.patch.dict(os.environ, {"REVISION": fake_revision})
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"revision": fake_revision}

