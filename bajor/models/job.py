from pydantic import BaseModel, HttpUrl

class TrainingJob(BaseModel):
    manifest_path: str
    id: str | None
    status: str | None
    run_opts: str = ''

    # remove the leading / from the manifest url
    # as it's added via the blob storage paths in schedule_job
    def stripped_manifest_path(self):
        return self.manifest_path.lstrip('/')


class PredictionJob(BaseModel):
    manifest_url: HttpUrl
    id: str | None
    status: str | None
    run_opts: str = ''
