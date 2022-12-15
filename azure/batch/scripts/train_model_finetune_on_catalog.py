import logging
import argparse

import pandas as pd
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

from zoobot.pytorch.training import finetune
from zoobot.pytorch.estimators import define_model
from zoobot.shared.schemas import cosmic_dawn_ortho_schema

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
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=11)  # benchmarks show 11 work on our VM types - was int((os.cpu_count())
    parser.add_argument('--prefetch-factor', dest='prefetch_factor', type=int, default=9) # benchmarks show 9 works on our VM types (lots of ram) - was 4 (default)
    # V100 GPU can handle 128 - can look at --mixed-precision opt to decrease the ram use
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
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
    args = parser.parse_args()

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

    datamodule = GalaxyDataModule(
        label_cols=cosmic_dawn_ortho_schema.label_cols,
        catalog=kade_catalog,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
        # uses default_args
    )
    datamodule.setup()

    # use a config object to setup the finetuning system
    config = {
        'checkpoint': {
            'file_template': 'epoch-{epoch}-val_loss-{finetuning/val_loss:.7f}',
            'save_top_k': args.save_top_k
        },
        'early_stopping': {
            'patience': args.patience
        },
        'trainer': {
            'devices': args.devices,
            'accelerator': args.accelerator
        },
        'finetune': {
            'encoder_dim': args.encoder_dim,
            'n_epochs': args.num_epochs,
            'n_layers': args.n_layers,
            'label_dim': len(cosmic_dawn_ortho_schema.label_cols),
            'label_mode': 'count',
            'schema': cosmic_dawn_ortho_schema,
            'prog_bar': args.progress_bar
        }
    }

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
        logger = WandbLogger(project='finetune', name=f'zoobot-bajor-{job_id}')
    else:
        logger = None

    # load the model from checkpoint
    model = define_model.ZoobotLightningModule.load_from_checkpoint(
        args.checkpoint)

    """
    Model:  ZoobotLightningModule(
    (train_accuracy): Accuracy()
    (val_accuracy): Accuracy()
    (model): Sequential(
      (0): EfficientNet(
    """
    encoder = model.get_submodule('model.0')  # includes avgpool and head

    # derive the checkpoint and model results here for possible use later
    # e.g. like in a prediction system etc
    # however our setup will save the
    _model, checkpoint_path = finetune.run_finetuning(
        config, encoder, datamodule, save_dir=args.save_dir, logger=logger
    )

    logging.info(
        f'Finished training on catalog - checkpoint save to: {checkpoint_path}')
