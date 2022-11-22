import logging
import os
import argparse

import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from zoobot.shared import label_metadata, schemas
from zoobot.pytorch.training import train_with_pytorch_lightning

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    logging.info('Begin training on catalog')

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    # expects catalog, not tfrecords
    parser.add_argument('--catalog', dest='catalog_loc',
                        type=str, action='append')
    parser.add_argument('--skip-mission-catalog', dest='skip_mission_catalog', default=False, action='store_true',
                        help='If true, skip training on the main catalog and train on a small subset and must be used with --debug flag')
    parser.add_argument('--mission-catalog', dest='mission_catalog_loc',
                        type=str, action='append')
    parser.add_argument('--num-workers',
                        dest='num_workers', type=int, default=11)  # benchmarks show 11 work on our VM types - was int((os.cpu_count())
    parser.add_argument('--prefetch-factor',
                        dest='prefetch_factor', type=int, default=9) # benchmarks show 9 works on our VM types (lots of ram) - was 4 (default)
    parser.add_argument('--architecture',
                        dest='model_architecture', type=str, default='efficientnet')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000)
    parser.add_argument('--resize-size', dest='resize_size',
                        type=int, default=224)
    # V100 GPU can handle 128 - can look at --mixed-precision opt to decrease the ram use
    parser.add_argument('--batch-size', dest='batch_size',
                        default=128, type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--patience', default=8, type=int)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
                        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate',
                        default=0.2, type=float)
    parser.add_argument('--mixed-precision', dest='mixed_precision', default=True, action='store_true',
                        help='If true, use automatic mixed precision (via PyTorch Lightning) to reduce GPU memory use (~x2). Else, use full (32 bit) precision')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true',
                        help='If true, cut each catalog down to 5k galaxies (for quick training). Should cause overfitting.')
    args = parser.parse_args()

    # short term train on dr5 & dr8 only till we build the custom dr12 catalogue
    question_answer_pairs = {}
    question_answer_pairs.update(label_metadata.decals_dr5_ortho_pairs)
    question_answer_pairs.update(label_metadata.decals_dr8_ortho_pairs)
    # long term train on all availbe decals data columns (dr12, dr5, dr8 etc)
    # question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs

    dependencies = label_metadata.decals_ortho_dependencies
    schema = schemas.Schema(question_answer_pairs, dependencies)
    logging.info('Schema: {}'.format(schema))

    # load each csv file catalog location into a combined pandas data frame
    # Note: this requires the same csv column format across csv files
    kade_catalog = pd.concat(
        map(pd.read_csv, args.catalog_loc),
        ignore_index=True)

    # load the mission catalog parquet files
    # Note: again - these must have the same column format if loading multiple files
    if args.skip_mission_catalog and not args.debug:
        raise Exception(
            'Skipping mission catalog without using --debug flag is not allowed. Fleeing to safety...')
    elif args.debug and args.skip_mission_catalog:
        # debugging mode only use the kade catalog
        # for dev / testing - note this will produce junk results!
        # but will test the code works :)
        catalog = kade_catalog
    else:
        mission_catalog = pd.concat(
            map(pd.read_parquet, args.mission_catalog_loc),
            ignore_index=True)

        # extract only the kade manifest column data from the larger mission catalog data
        # this allows us to modify the kade manifest and automatically include the mission column data
        # as we progress with adding new mission data (e.g. decals 1&2 etc) to kade exports
        subset_mission_catalog = mission_catalog[kade_catalog.columns]

        # combine the catalog files for use in the zooboe training system
        catalog = pd.concat([kade_catalog, subset_mission_catalog], ignore_index=True)

        # debug mode - only use a subset of the data
        if args.debug:
          logging.warning(
              'Using debug mode: cutting catalog down to 5k galaxies')
          catalog = catalog.sample(5000).reset_index(drop=True)


    # print the first and last file loc of the loaded catalog
    logging.info('Catalog has {} rows'.format(len(catalog.index)))
    logging.info('First file_loc {}'.format(catalog['file_loc'].iloc[0]))
    logging.info('Last file_loc {}'.format(
        catalog['file_loc'].iloc[len(catalog.index) - 1]))

    if args.wandb:
        wandb_logger = WandbLogger(
            project='zoobot-pytorch-catalog-example',
            name=os.path.basename(args.save_dir),
            log_model="all")
        # only rank 0 process gets access to the wandb.run object, and for non-zero rank processes: wandb.run = None
        # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-w-and-b
    else:
        wandb_logger = None

    train_with_pytorch_lightning.train_default_zoobot_from_scratch(
        save_dir=args.save_dir,
        schema=schema,
        catalog=catalog,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        # augmentation parameters
        color=args.color,
        resize_size=args.resize_size,
        # hardware parameters
        accelerator=args.accelerator,
        nodes=args.nodes,
        gpus=args.gpus,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        mixed_precision=args.mixed_precision,
        wandb_logger=wandb_logger,
        # checkpointing setup, e.g. supervised loss values can be
        # 0.0000978
        # 0.0000241458789803
        checkpoint_file_template='epoch-{epoch}-supervised_loss-{val/supervised_loss:.7f}',
        auto_insert_metric_name=False
    )
