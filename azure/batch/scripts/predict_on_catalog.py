import logging
import time
import datetime
from typing import List, Optional
from PIL import Image

import pandas as pd
import requests
import torch
import pytorch_lightning as pl

from zoobot.shared import save_predictions
from pytorch_galaxy_datasets import galaxy_datamodule, galaxy_dataset

# # override the get_galaxy_label package function for use in the prediction data loaders
# # essentially we don't need the labels in the prediction catalogues
# def predict_get_galaxy_label(galaxy, label_cols):
#     return []

# galaxy_dataset.get_galaxy_label = predict_get_galaxy_label


class PredictionGalaxyDataset(galaxy_dataset.GalaxyDataset):
  def __getitem__(self, idx):
      #the index is id_str so can use that for quick search on 1M+ catalo
      # galaxy = self._catalog.loc[idx]
      galaxy = self.catalog.iloc[idx]
      # option A
      # img_path = galaxy['file_loc']
      # image = read_image(img_path) # PIL under the hood: Returns CHW Tensor.
      # option B - tiny bit faster when CPU-limited
      url = galaxy['file_loc']
      try:
          # downloading the file in full before use
          # from io import BytesIO
          # r = requests.get(url)
          # image = Image.open(BytesIO(response.content))
          # decoded_jpeg = galaxy_dataset.decode_jpeg(BytesIO(r.content).read())
          #
          # streaming the file as it is used (saves on memory)
          r = requests.get(url, stream=True)
          decoded_jpeg = galaxy_dataset.decode_jpeg(r.raw.read())
          # HWC PIL image via simplejpeg
          image = Image.fromarray(decoded_jpeg)
      except Exception as e:
          logging.critical('Cannot load {}'.format(galaxy['file_loc']))
          raise e
      # label = get_galaxy_label(galaxy, self.label_cols)
      # avoid the label lookups as they aren't used in the prediction
      label = []

      # logging.info((image.shape, torch.max(image), image.dtype, label))  # always 0-255 uint8

      if self.transform:
          # a CHW tensor, which torchvision wants. May change to PIL image.
          image = self.transform(image)

      if self.target_transform:
          label = self.target_transform(label)

      # logging.info((image.shape, torch.max(image), image.dtype, label))  #  should be 0-1 float
      return image, label


class PredictionGalaxyDataModule(galaxy_datamodule.GalaxyDataModule):
    def setup(self, stage: Optional[str] = None):
        super().setup(stage)

        self.predict_dataset = PredictionGalaxyDataset(
            catalog=self.predict_catalog, label_cols=self.label_cols, transform=self.transform
        )


def predict(catalog: pd.DataFrame, model: pl.LightningModule, n_samples: int, label_cols: List, save_loc: str, datamodule_kwargs, trainer_kwargs):

    image_id_strs = list(catalog['id_str'])
    # perhapse include the subject_id for use in downstream systems
    # image_id_strs = list(catalog['subject_id'])

    predict_datamodule = PredictionGalaxyDataModule(
        label_cols=label_cols,
        predict_catalog=catalog,  # no need to specify the other catalogs
        # will use the default transforms unless overridden with datamodule_kwargs
        #
        **datamodule_kwargs  # e.g. batch_size, resize_size, crop_scale_bounds, etc.
    )

    # predict_datamodule = galaxy_datamodule.GalaxyDataModule(
    #     label_cols=label_cols,
    #     predict_catalog=catalog,  # no need to specify the other catalogs
    #     # will use the default transforms unless overridden with datamodule_kwargs
    #     #
    #     **datamodule_kwargs  # e.g. batch_size, resize_size, crop_scale_bounds, etc.
    # )
    # with this stage arg, will only use predict_catalog
    # crucial to specify the stage, or will error (as missing other catalogs)
    predict_datamodule.setup(stage='predict')

    # set up trainer (again)
    trainer = pl.Trainer(
        max_epochs=-1,  # does nothing in this context, suppresses warning
        **trainer_kwargs  # e.g. gpus
    )

    # from here, very similar to tensorflow version - could potentially refactor

    logging.info('Beginning predictions')
    start = datetime.datetime.fromtimestamp(time.time())
    logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))

    logging.info(len(trainer.predict(model, predict_datamodule)))

    # trainer.predict gives list of tensors, each tensor being predictions for a batch. Concat on axis 0.
    # range(n_samples) list comprehension repeats this, for dropout-permuted predictions. Stack to create new last axis.
    # final shape (n_galaxies, n_answers, n_samples)
    predictions = torch.stack([torch.concat(trainer.predict(model, predict_datamodule), dim=0) for n in range(n_samples)], dim=2).numpy()
    logging.info('Predictions complete - {}'.format(predictions.shape))

    logging.info(f'Saving predictions to {save_loc}')

    if save_loc.endswith('.csv'):      # save as pandas df
        save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)
    elif save_loc.endswith('.hdf5'):
        save_predictions.predictions_to_hdf5(predictions, image_id_strs, label_cols, save_loc)
    else:
        logging.warning('Save format of {} not recognised - assuming csv'.format(save_loc))
        save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)

    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))
