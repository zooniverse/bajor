import logging
import argparse
import json
import os

import pandas as pd
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from galaxy_datasets.transforms import default_view_config, GalaxyViewTransform

from zoobot.pytorch.training import finetune
from zoobot.shared.schemas import cosmic_dawn_ortho_schema, euclid_ortho_schema, gz_jwst_schema
from train_on_catalog import TrainingGalaxyDataModule
if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    logging.info('Begin training on catalog')

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', dest='save_dir', type=str, required=True)
    # expects path to csv
    parser.add_argument('--catalog', dest='catalog_loc', type=str, required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, required=True)
    parser.add_argument('--schema', dest='schema', type=str, default='cosmic_dawn')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=int(os.cpu_count()))  # benchmarks show 4 work on our VM types - was int((os.cpu_count())
    parser.add_argument('--prefetch-factor', dest='prefetch_factor', type=int, default=9) # benchmarks show 9 works on our VM types (lots of ram) - was 4 (default)
    parser.add_argument('--batch-size', dest='batch_size', default=10, type=int) # benchmarks show 10 works on our new VM type - was 128 (default)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--progress-bar', dest='progress_bar', action='store_true')
    parser.add_argument('--encoder-dim', dest='encoder_dim', default=1280, type=int)
    parser.add_argument('--n-layers', dest='n_layers', default=2, type=int)
    parser.add_argument('--num-epochs', dest='num_epochs', default=100, type=int)
    parser.add_argument('--save-top-k', dest='save_top_k', default=1, type=int)
    parser.add_argument('--patience', default=15, type=int)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--erase-iterations', dest='erase_iterations', type=int, default=0)
    parser.add_argument('--fixed-crop', dest='fixed_crop', type=str, default=None)
    parser.add_argument('--n-blocks', dest='n_blocks', type=int, default=0)
    args = parser.parse_args()

    schema_dict = {
        'cosmic_dawn': cosmic_dawn_ortho_schema,
        'euclid': euclid_ortho_schema,
        'jwst_cosmos': gz_jwst_schema
    }
    schema = schema_dict.get(args.schema, cosmic_dawn_ortho_schema)
    # setup the error reporting tool - https://app.honeybadger.io/projects/
    honeybadger_api_key = os.getenv('HONEYBADGER_API_KEY')
    if honeybadger_api_key:
        from honeybadger import honeybadger
        honeybadger.configure(api_key=honeybadger_api_key)

    # load csv file catalog location into a pandas data frame
    kade_catalog = pd.read_csv(args.catalog_loc)

    if args.debug:
        logging.debug('Using --debug mode')
        # add debugging steps here.. like data cleaning, reshaping etc
        # kade_catalog['id_str'] = kade_catalog['id_str'].astype(str)
        # kade_catalog = kade_catalog.rename(columns={'problem_non-star': 'problem_artifact'})
        # kade_catalog['file_loc'].str.replace('/local/paths/to/catalog/images', '/mnt/batch/tasks/fsmounts/training/catalogues/production/images')

    # print the first and last file loc of the loaded catalog
    logging.info('Catalog has {} rows'.format(len(kade_catalog.index)))
    logging.info('First file_loc {}'.format(kade_catalog['file_loc'].iloc[0]))
    logging.info('Last file_loc {}'.format(
        kade_catalog['file_loc'].iloc[len(kade_catalog.index) - 1]))

    transform = None
    try:
        if args.fixed_crop:
            transform_config = default_view_config()
            transform_config.erase_iterations = args.erase_iterations
            transform_config.fixed_crop = json.loads(args.fixed_crop)
            transform = GalaxyViewTransform(transform_config)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid fixed_crop JSON: {args.fixed_crop}")

    datamodule = TrainingGalaxyDataModule(
        label_cols=schema.label_cols,
        catalog=kade_catalog,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        custom_torchvision_transform=transform,
        # gz evo checkpoint expects 224x224 input image - the following value must align to the encoded value in the model checkpoint!
        resize_after_crop=int(os.environ.get('IMAGE_SIZE', '224'))
        # uses default_args
    )
    datamodule.setup()

    if args.wandb:
        try:
            import os
            # wandb needs API keys present as WANDB_API_KEY env var
            # https://docs.wandb.ai/guides/track/advanced/environment-variables
            os.environ['WANDB_API_KEY']
            job_id = os.environ.get('AZ_BATCH_JOB_ID', 'dev-env')
            # setup wandb to use the use shared writable dir for config and cache
            # https://learn.microsoft.com/en-gb/azure/batch/files-and-directories#root-directory-structure
            shared_dir = os.getenv('AZ_BATCH_NODE_SHARED_DIR')
            os.environ['WANDB_CONFIG_DIR'] = f'{shared_dir}/.config/wandb'
            os.environ['WANDB_CACHE_DIR'] = f'{shared_dir}/.cache/wandb'
        except KeyError as e:
            logging.error('WANDB_API_KEY not found in environment variables')
            # and make sure we reraise the error
            raise e

        from pytorch_lightning.loggers import WandbLogger
        wandb_project_name = os.environ.get('WANDB_PROJECT_NAME', 'finetune-bajor')
        logger = WandbLogger(project=wandb_project_name, name=f'zoobot-{job_id}')
    else:
        logger = None


    # load the model from checkpoint
    model = finetune.FinetuneableZoobotTree(
        # params specific to tree finetuning
        schema=schema,
        n_blocks=args.n_blocks,
        # zoobot_checkpoint_loc=args.checkpoint
        name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano'
    )

    trainer = finetune.get_trainer(
        save_dir=args.save_dir,
        logger=logger,
        max_epochs=args.num_epochs,
        devices= args.devices,
        accelerator=args.accelerator,
        patience=args.patience,
        save_top_k=args.save_top_k,
        file_template='epoch-{epoch}-val_loss-{finetuning/val_loss:.7f}'
    )

    # derive the checkpoint and model results here for possible use later
    # e.g. like in a prediction system etc
    trainer.fit(model, datamodule)

    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    logging.info(
        f'Finished training on catalog - checkpoint save to: {best_checkpoint_path}')
