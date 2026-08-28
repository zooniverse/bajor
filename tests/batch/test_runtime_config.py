import json
import shlex

from bajor.batch.runtime_config import (
    build_run_opts,
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


def test_workflow_name_is_added_to_run_opts_as_schema():
    options = JobOptions(workflow_name='gz_evo_v2_public')

    assert build_run_opts(options) == '--schema gz_evo_v2_public'


def test_workflow_name_does_not_duplicate_existing_schema_run_opt():
    options = JobOptions(
        workflow_name='gz_evo_v2_public',
        run_opts='--schema cosmic_dawn --debug'
    )

    assert build_run_opts(options) == '--schema cosmic_dawn --debug'


def test_build_run_opts_can_skip_schema_option():
    options = JobOptions(workflow_name='gz_evo_v2_public')

    assert build_run_opts(options, include_schema=False) == ''


def test_build_run_opts_adds_custom_schema_json():
    schema_payload = {
        'question_answer_pairs': {
            'merger': ['_yes', '_no', '_artifact']
        },
        'dependencies': {
            'merger': None
        }
    }
    custom_schema_json = json.dumps(schema_payload)
    options = JobOptions(custom_schema_json=custom_schema_json)
    run_opts = shlex.split(build_run_opts(options))

    assert '--custom-schema-json' in run_opts
    assert run_opts[run_opts.index('--custom-schema-json') + 1] == custom_schema_json


def test_build_run_opts_does_not_duplicate_existing_custom_schema_json():
    run_opts = '--custom-schema-json \'{"question_answer_pairs":{"old":["_yes"]}}\' --debug'
    options = JobOptions(
        custom_schema_json='{"question_answer_pairs":{"new":["_no"]}}',
        run_opts=run_opts
    )

    assert build_run_opts(options) == f'--schema cosmic_dawn {run_opts}'
