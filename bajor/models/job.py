from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict

class Options(BaseModel):
    run_opts: str = ""
    workflow_name: str = 'cosmic_dawn'
    fixed_crop: dict | None = None


class TrainingJob(BaseModel):
    manifest_path: str
    id: Optional[str] = None
    status: Optional[str] =  None
    opts: Options = Options()

    # remove the leading / from the manifest url
    # as it's added via the blob storage paths in schedule_job
    def stripped_manifest_path(self):
        return self.manifest_path.lstrip('/')


class PredictionJob(BaseModel):
    manifest_url: HttpUrl
    id: Optional[str] = None
    status: Optional[str] =  None
    opts: Options = Options()
