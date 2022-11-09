import uvicorn
import os
from fastapi import FastAPI, Request
from honeybadger import contrib

from bajor.env_helpers import revision, host, port
from bajor.apis.training import router as training_router
from bajor.apis.predictions import router as predictions_router


if os.getenv('DEBUG'):
  import pdb

class ApiInfo(object):
    def __init__(
        self,
        app: FastAPI,
        request: Request):

        super(ApiInfo, self).__init__()

        self.autodocs = {
            'swagger': str(request.url) + app.docs_url.strip("/"),
            'redoc': str(request.url) + app.redoc_url.strip("/"),
        }

        self.revision = revision()

app = FastAPI()
app.add_middleware(contrib.ASGIHoneybadger)

# Main app end points (also a health check)
@app.get("/")
async def root(request: Request):
    apiInfo = ApiInfo(app, request)

    return apiInfo.__dict__

# mount the modules at their router path prefixes
app.include_router(training_router)
app.include_router(predictions_router)


def start_app(reload=False):
    uvicorn.run(
        "bajor.api:app",
        host=host(),
        port=int(port()),
        reload=reload
    )


def start_dev_app():
    start_app(reload=True)
