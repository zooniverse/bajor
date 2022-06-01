import uvicorn, os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return { "revision": os.environ.get('REVISION') }

def start_app():
    uvicorn.run("bajor.main:app", host="0.0.0.0",
                port=int(os.environ.get('PORT', '8000')))
