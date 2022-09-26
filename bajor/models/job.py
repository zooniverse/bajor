from pydantic import BaseModel

class Job(BaseModel):
    manifest_path: str
    id: str | None
    status: str | None
    run_opts: str = ''
