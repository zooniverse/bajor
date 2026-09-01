import os
import logging
import argparse
import json

import pandas as pd
import pytorch_lightning as pl

from zoobot.pytorch.training import finetune
import predict_on_catalog
from schema_resolver import load_schema
from galaxy_datasets.transforms import default_view_config, GalaxyViewTransform


def huggingface_config():
    repo_id = os.getenv('HF_REPO_ID')
    token = os.getenv('HF_TOKEN')

    if repo_id and token:
        return repo_id, token

    return None, None


def checkpoint_filename(hf_filename, checkpoint_path):
    return hf_filename or os.path.basename(checkpoint_path)


def resolve_checkpoint_path(checkpoint_path, hf_filename):
    repo_id, token = huggingface_config()
    if not repo_id:
        return checkpoint_path

    from huggingface_hub import hf_hub_download

    filename = checkpoint_filename(hf_filename, checkpoint_path)
    resolved_checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token
    )
    logging.info(f'Loaded Hugging Face checkpoint from: {resolved_checkpoint_path}')
    return resolved_checkpoint_path


def load_model_from_checkpoint(checkpoint_path, hf_filename=None):
    resolved_checkpoint_path = resolve_checkpoint_path(checkpoint_path, hf_filename)
    logging.info('Returning model from checkpoint: {}'.format(resolved_checkpoint_path))
    return finetune.FinetuneableZoobotTree.load_from_checkpoint(resolved_checkpoint_path)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', dest='checkpoint_path', type=str, required=True)
    parser.add_argument('--hf-filename', dest='hf_filename', type=str, default=None)
    parser.add_argument('--save-path', dest='save_loc', type=str, required=True)
    parser.add_argument('--catalog-url', dest='catalog_url', type=str, required=True)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=1)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=int(os.cpu_count()))
    parser.add_argument('--prefetch-factor', dest='prefetch_factor', type=int, default=4)
    # V100 GPU can handle 128 - can look at --mixed-precision opt to decrease the ram use
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--erase-iterations', dest='erase_iterations', type=int, default=0)
    parser.add_argument('--fixed-crop', dest='fixed_crop', type=str, default=None)
    parser.add_argument('--custom-schema-json', dest='custom_schema_json', type=str, default=None)
    args = parser.parse_args()

    # setup the error reporting tool - https://app.honeybadger.io/projects/
    honeybadger_api_key = os.getenv('HONEYBADGER_API_KEY')
    if honeybadger_api_key:
        from honeybadger import honeybadger
        honeybadger.configure(api_key=honeybadger_api_key)

    logging.info(f'Begin predictions on catalog: {args.catalog_url}')

    # load the catalog from a remote JSON url
    # for really large remote files they could be parquet format?!

    # expected manifest format from hamlet has no column headers
    # it's json arrays of data with the following list of array object
    # catalog[0] = first column being the URL of the prediction image
    # catalog[1] = second colum being the image metadata (including subject_id)
    raw_json_catalog = pd.read_json(args.catalog_url)
    # extract and convert the json metadata column
    # https: // pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html
    catalog = pd.json_normalize(raw_json_catalog[1].apply(json.loads))
    # add in the image url as it's used in the
    catalog['image_url'] = raw_json_catalog[0]

    schema = load_schema(None, custom_schema_json=args.custom_schema_json)

    model = load_model_from_checkpoint(args.checkpoint_path, args.hf_filename)

    transform = None
    try:
        if args.fixed_crop:
            transform_config = default_view_config()
            transform_config.erase_iterations = args.erase_iterations
            transform_config.fixed_crop = json.loads(args.fixed_crop)
            transform = GalaxyViewTransform(transform_config)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid fixed_crop JSON: {args.fixed_crop}")

    # provide a way to augment the datamodule and trainer default configs
    # used for testing on local dev machine
    datamodule_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
        # gz evo checkpoint expects 224x224 input image - the following value must align to the encoded value in the model checkpoint!
        'resize_after_crop': int(os.environ.get('IMAGE_SIZE', '224')),
        'custom_torchvision_transform': transform
    }
    trainer_args = {
        'devices': args.devices,
        'accelerator': args.accelerator
    }

    predict_on_catalog.predict(
        model=model,
        catalog=catalog,
        save_loc=args.save_loc,
        n_samples=args.num_samples,
        label_cols=schema.label_cols,
        schema=schema,
        datamodule_kwargs=datamodule_args,
        trainer_kwargs=trainer_args
    )


