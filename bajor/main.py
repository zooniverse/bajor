from tracemalloc import start
import uvicorn, os
from fastapi import FastAPI
from honeybadger import contrib

app = FastAPI()
# configure HB through the env vars https://docs.honeybadger.io/lib/python/#configuration
app.add_middleware(contrib.ASGIHoneybadger)

@app.get("/")
def root():
    return { "revision": os.environ.get('REVISION') }

def start_app(reload=False):
    uvicorn.run(
        "bajor.main:app",
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '8000')),
        reload=reload
    )

def start_dev_app():
    start_app(reload=True)
