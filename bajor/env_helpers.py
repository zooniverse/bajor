import os

def api_basic_username():
    return os.environ.get('BASIC_AUTH_USERNAME', 'bajor')

def api_basic_password():
    return os.environ.get('BASIC_AUTH_PASSWORD', 'bajor')

def api_job_complete_url(job_id):
    bajor_url = os.environ.get('JOB_COMPLETE_URL', 'https://bajor-staging.zooniverse.org')
    return f'{bajor_url}/job/{job_id}'

def revision():
    os.environ.get('REVISION')

def host():
    os.environ.get('HOST', '0.0.0.0')

def port():
    os.environ.get('PORT', '8000')
