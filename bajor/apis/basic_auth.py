import secrets
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from bajor.env_helpers import api_basic_username, api_basic_password

# add the http basic authorization mode
# https://fastapi.tiangolo.com/advanced/security/http-basic-auth/#simple-http-basic-auth
security = HTTPBasic()

def validate_basic_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username, api_basic_username())
    correct_password = secrets.compare_digest(
        credentials.password, api_basic_password())
    if not (correct_username and correct_password):
        return False
    else:
      return True
