# launching batch processing functions go here
import logging, uuid, os

from azure.batch import BatchServiceClient
from azure.batch.models import JobListOptions
from azure.common.credentials import ServicePrincipalCredentials

# Gunicorn logger handler will get attached if needed in server.py
log = logging.getLogger(os.getenv('FLASK_APP', default='BAJOR'))


def create_batch_job(job_id=None, manifest_container_path=None):
    log.info(
        f'server_job, create_batch_job, job_id {job_id}, using manifest')

    job_submission_timestamp = get_utc_time()

    # attempt to read the manifest file from the container path
    #
    # 1. ensure the manifest_container_path file has data in it to avoid training on the default existing cataglogue only
    # TODO: read and check the manifest_container_path has data
    #
    # 2. count the number of manifest entries for reporting
    number_of_manifest_entries = 0
    log.info('server_job, create_batch_job, using manifest from path: {} with number of image paths: {}'.format(manifest_container_path, number_of_manifest_entries))
    # TODO: add job completed / error reporting for job tracking data model (pgserver / sqlite?)
    #   if len(image_paths) == 0:
    #       job_status = get_job_status(
    #           'completed', '0 images found in provided list of images.')
    #       job_status_table.update_job_status(job_id, job_status)
    #       return
    # and then job status like it's good etc
    # job_status = get_job_status('created', f'{num_images} images listed; submitting the job...')
    # job_status_table.update_job_status(job_id, job_status)

# job_status = get_job_status('created', f'{num_images} images listed; submitting the job...')
# job_status_table.update_job_status(job_id, job_status)

# except Exception as e:
# job_status = get_job_status('failed', f'Error occurred while preparing the Batch job: {e}')
# job_status_table.update_job_status(job_id, job_status)
# log.error(f'server_job, create_batch_job, Error occurred while preparing the Batch job: {e}')
# return  # do not start monitoring


def get_batch_job_list(job_list_options=JobListOptions(
    filter='state eq \'active\'',
    select='id'
)):
    credentials = ServicePrincipalCredentials(
        client_id=os.getenv('APP_CLIENT_ID'),
        secret=os.getenv('APP_CLIENT_SECRET'),
        tenant=os.getenv('APP_TENANT_ID'),
        resource='https://batch.core.windows.net/'
    )
    batch_client = BatchServiceClient(
        credentials=credentials,
        batch_url=os.getenv('BATCH_ACCOUNT_URL',
                            'https://zoobot.eastus.batch.azure.com')
    )

    jobs_generator = batch_client.job.list(
        job_list_options=job_list_options)
    jobs_list = [j for j in jobs_generator]
    print('Active batch jobs list')
    print(jobs_list)

if __name__ == '__main__':
    job_id = str(uuid.uuid4())
    #create_batch_job(job_id=job_id)
    # get_batch_job_list()


