import os
import logging
import argparse
import json

import pandas as pd
import pytorch_lightning as pl

from zoobot.pytorch.estimators import define_model
# from zoobot.pytorch.predictions import predict_on_catalog
import predict_on_catalog
from zoobot.shared import label_metadata

def load_model_from_checkpoint(checkpoint_path):
    logging.info('Returning model from checkpoint: {}'.format(checkpoint_path))

    return define_model.ZoobotLightningModule.load_from_checkpoint(checkpoint_path)

def label_cols_from_metadata():
    # setup decals 5 & 8 labels
    question_answer_pairs = {}
    question_answer_pairs.update(label_metadata.decals_dr5_ortho_pairs)
    question_answer_pairs.update(label_metadata.decals_dr8_ortho_pairs)
    # long term predict on all available decals data columns (dr12, dr5, dr8 etc)
    # question_answer_pairs = label_metadata.decals_all_campaigns_ortho_pairs
    _decals_questions, decals_label_cols = label_metadata.extract_questions_and_label_cols(
        question_answer_pairs)
    return decals_label_cols

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', dest='checkpoint_path', type=str)
    parser.add_argument('--save-path', dest='save_loc', type=str)
    parser.add_argument('--catalog-url', dest='catalog_url', type=str)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=1)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=11)  # benchmarks show 11 work on our VM types - was int((os.cpu_count())
    parser.add_argument('--prefetch-factor', dest='prefetch_factor', type=int, default=9)  # benchmarks show 9 works on our VM types (lots of ram) - was 4 (default)
    # V100 GPU can handle 128 - can look at --mixed-precision opt to decrease the ram use
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', default=1, type=int)
    args = parser.parse_args()

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

    model = load_model_from_checkpoint(args.checkpoint_path)

    # provide a way to augment the datamodule and trainer default configs
    # used for testing on local dev machine
    datamodule_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor
    }
    trainer_args = {
        'gpus': args.gpus,
        'accelerator': args.accelerator
    }

    predict_on_catalog.predict(
        model=model,
        catalog=catalog,
        save_loc=args.save_loc,
        n_samples=args.num_samples,
        label_cols=label_cols_from_metadata(),
        datamodule_kwargs=datamodule_args,
        trainer_kwargs=trainer_args
    )



