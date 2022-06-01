import uvicorn, os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return { "revision": os.environ.get('REVISION') }

def start_app():
    uvicorn.run(
        "bajor.main:app",
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '8000'))
    )

def start_dev_app():
    uvicorn.run(
        "bajor.main:app",
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '8000')),
        reload=True
    )
