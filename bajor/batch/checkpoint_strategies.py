from typing import Callable, Dict

CheckpointStrategy = Callable[[], str]

def euclid_checkpoint() -> str:
    return 'EUCLID_ZOOBOT_CHECKPOINT_TARGET'

def cosmos_checkpoint() -> str:
    return 'COSMOS_ZOOBOT_CHECKPOINT_TARGET'

def default_checkpoint() -> str:
    return 'ZOOBOT_CHECKPOINT_TARGET'

_checkpoint_strategies: Dict[str, CheckpointStrategy] = {
    'euclid': euclid_checkpoint,
    'jwst_cosmos': cosmos_checkpoint,
}

def get_checkpoint_target(workflow_name: str) -> str:
    return _checkpoint_strategies.get(workflow_name, default_checkpoint)()