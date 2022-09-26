import os

from functools import cache
from azure.batch import BatchServiceClient
from azure.common.credentials import ServicePrincipalCredentials

@cache
def azure_batch_client():
    credentials = ServicePrincipalCredentials(
        client_id=os.getenv('APP_CLIENT_ID'),
        secret=os.getenv('APP_CLIENT_SECRET'),
        tenant=os.getenv('APP_TENANT_ID'),
        resource='https://batch.core.windows.net/'
    )
    return BatchServiceClient(
        credentials=credentials,
        batch_url=os.getenv('BATCH_ACCOUNT_URL',
                            'https://zoobot.eastus.batch.azure.com')
    )
