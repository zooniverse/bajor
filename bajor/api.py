import uvicorn
import os
from fastapi import FastAPI
from honeybadger import contrib
from bajor.env_helpers import revision, host, port
from bajor.apis.training import training_app
from bajor.apis.predictions import predictions_app

if os.getenv('DEBUG'):
  import pdb

app = FastAPI()
# configure HB through the env vars https://docs.honeybadger.io/lib/python/#configuration
app.add_middleware(contrib.ASGIHoneybadger)

# Main app end points (also a health check)
@app.get("/")
async def root():
    return {"revision": revision()}

# SubAPIs
# mount the subapi's at path prefixes
app.mount("/training", training_app)
app.mount("/prediction", predictions_app)

def start_app(reload=False):
    uvicorn.run(
        "bajor.api:app",
        host=host(),
        port=int(port()),
        reload=reload
    )


def start_dev_app():
    start_app(reload=True)
