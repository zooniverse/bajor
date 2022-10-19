from pydantic import BaseModel, HttpUrl

class TrainingJob(BaseModel):
    manifest_path: str
    id: str | None
    status: str | None
    run_opts: str = ''


class PredictionJob(BaseModel):
    manifest_url: HttpUrl
    id: str | None
    status: str | None
    run_opts: str = ''
