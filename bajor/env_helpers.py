import os

def api_basic_username():
    return os.environ.get('BASIC_AUTH_USERNAME', 'bajor')

def api_basic_password():
    return os.environ.get('BASIC_AUTH_PASSWORD', 'bajor')

def revision():
    return os.environ.get('REVISION')

def host():
    return os.environ.get('HOST', '0.0.0.0')

def port():
    return os.environ.get('PORT', '8000')

def max_num_pool_nodes(pool_id):
    if('training' in pool_id):
        return os.environ.get('MAX_NODES_TRAINING', 2)
    elif('predictions' in pool_id):
        return os.environ.get('MAX_NODES_PREDICTION', 2)
    else:
        return 0

def training_run_opts():
    return os.environ.get('TRAINING_RUN_OPTS', '')
