import os
import shlex
from urllib.parse import urlparse

from bajor.batch.checkpoint_strategies import get_checkpoint_target
from bajor.models.job import JobOptions


MODELS_CONTAINER_MOUNT_DIR_ENV = 'MODELS_CONTAINER_MOUNT_DIR'
MODELS_CONTAINER_NAME = 'models'
def _models_mount_root() -> str:
    return f'$AZ_BATCH_NODE_MOUNTS_DIR/${MODELS_CONTAINER_MOUNT_DIR_ENV}'


def _relative_checkpoint_path(checkpoint_ref: str) -> str:
    if checkpoint_ref.startswith('http://') or checkpoint_ref.startswith('https://'):
        checkpoint_path = urlparse(checkpoint_ref).path.lstrip('/')
        if checkpoint_path.startswith(f'{MODELS_CONTAINER_NAME}/'):
            return checkpoint_path[len(f'{MODELS_CONTAINER_NAME}/'):]
        return checkpoint_path

    return checkpoint_ref.lstrip('/')


def resolve_checkpoint_target(options: JobOptions) -> str:
    if options.pretrained_checkpoint_url:
        return _relative_checkpoint_path(options.pretrained_checkpoint_url)

    checkpoint_target = get_checkpoint_target(options.workflow_name)
    return os.getenv(checkpoint_target, 'zoobot.ckpt')


def resolve_pretrained_checkpoint_path(options: JobOptions) -> str:
    if options.pretrained_checkpoint_url:
        return f'{_models_mount_root()}/{_relative_checkpoint_path(options.pretrained_checkpoint_url)}'

    checkpoint_file = os.getenv('ZOOBOT_FINETUNE_CHECKPOINT_FILE', 'zoobot_pretrained_model.ckpt')
    return f'{_models_mount_root()}/{checkpoint_file}'


def resolve_training_script_path(options: JobOptions) -> str:
    return options.training_script_path or os.getenv('ZOOBOT_FINETUNE_TRAIN_CMD', 'train_model_finetune_on_catalog.py')


def resolve_prediction_script_path(options: JobOptions) -> str:
    return options.prediction_script_path or os.getenv('ZOOBOT_PREDICTION_CMD', 'predict_catalog_with_model.py')


def resolve_promote_script_path(options: JobOptions) -> str:
    return options.promote_script_path or os.getenv('ZOOBOT_PROMOTE_CMD', 'promote_best_checkpoint_to_model.sh')


def resolve_container_image_name(options: JobOptions) -> str:
    return options.container_image_name or os.getenv('CONTAINER_IMAGE_NAME')


def build_run_opts(options: JobOptions, include_schema: bool = True) -> str:
    run_opts = options.run_opts.strip()
    generated_opts = []

    if include_schema and options.workflow_name and not _has_schema_arg(run_opts):
        generated_opts.append(f'--schema {options.workflow_name.lower()}')

    if options.custom_schema_json and not _has_custom_schema_arg(run_opts):
        generated_opts.append(f'--custom-schema-json {shlex.quote(options.custom_schema_json)}')

    return ' '.join([*generated_opts, run_opts]).strip()


def _has_schema_arg(run_opts: str) -> bool:
    try:
        return '--schema' in shlex.split(run_opts)
    except ValueError:
        return '--schema' in run_opts.split()


def _has_custom_schema_arg(run_opts: str) -> bool:
    try:
        return '--custom-schema-json' in shlex.split(run_opts)
    except ValueError:
        return '--custom-schema-json' in run_opts.split()
