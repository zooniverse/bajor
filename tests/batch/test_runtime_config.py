from bajor.batch.runtime_config import (
    resolve_container_image_name,
    resolve_checkpoint_target,
    resolve_pretrained_checkpoint_path,
)
from bajor.models.job import JobOptions


def test_explicit_checkpoint_path_overrides_legacy_checkpoint_target():
    options = JobOptions(
        workflow_name='euclid',
        pretrained_checkpoint_url='jwst/custom.ckpt'
    )

    assert resolve_checkpoint_target(options) == 'jwst/custom.ckpt'
    assert resolve_pretrained_checkpoint_path(options) == '$AZ_BATCH_NODE_MOUNTS_DIR/$MODELS_CONTAINER_MOUNT_DIR/jwst/custom.ckpt'


def test_blob_url_checkpoint_ref_is_normalized_to_relative_models_path():
    options = JobOptions(
        pretrained_checkpoint_url='https://kadeactivelearning.blob.core.windows.net/models/staging-euclid-zoobot.ckpt'
    )

    assert resolve_checkpoint_target(options) == 'staging-euclid-zoobot.ckpt'
    assert resolve_pretrained_checkpoint_path(options) == '$AZ_BATCH_NODE_MOUNTS_DIR/$MODELS_CONTAINER_MOUNT_DIR/staging-euclid-zoobot.ckpt'


def test_explicit_container_image_name_overrides_env():
    options = JobOptions(container_image_name='zoobot.azurecr.io/pytorch:custom-jwst')

    assert resolve_container_image_name(options) == 'zoobot.azurecr.io/pytorch:custom-jwst'
