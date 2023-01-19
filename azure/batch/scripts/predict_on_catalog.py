import logging
import time
import datetime
from typing import List, Optional
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image

import numpy as np
import pandas as pd
from scipy.stats import beta
import pytorch_lightning as pl
import torch
# See https://github.com/mwalmsley/galaxy-datasets/blob/main/galaxy_datasets/pytorch/galaxy_dataset.py
from galaxy_datasets.pytorch import galaxy_dataset, galaxy_datamodule
# See https://github.com/mwalmsley/zoobot/blob/main/zoobot/shared/save_predictions.py
from zoobot.shared import save_predictions


# add retries on requests if we have flaky networks
# https://www.peterbe.com/plog/best-practice-with-retries-with-requests
def requests_retry_session(retries=3, backoff_factor=0.3):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        # use the following to force a retry on the following HTTP response status codes
        # https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html
        # server error or connection problems to the origin server
        status_forcelist=(104, 429, 500, 502, 503, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    # should only have https scheme but let's be safe here
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class PredictionGalaxyDataset(galaxy_dataset.GalaxyDataset):
  # override the default class implementation for predictions that download from URL
  def __init__(self, catalog: pd.DataFrame, label_cols=None, transform=None, target_transform=None):
      self.catalog = catalog

      self.label_cols = label_cols
      self.transform = transform
      self.target_transform = target_transform


  def __getitem__(self, idx):
      galaxy = self.catalog.iloc[idx]
      # load the data from the remote image URL
      url = galaxy['image_url']
      try:
          # Currently only supporting JPEG images
          # TODO: support other subject image formats (PNG etc)
          # Note: the model must be trained on similar image formats
          #
          # streaming the file as it is used (saves on memory)
          logging.debug('Downloading url: {}'.format(url))
          # use retries on requests if we have flaky networks
          response = requests_retry_session().get(url, stream=True)
          # ensure we raise other response errors like 404 and 500 etc
          # Note: we don't retry on errors that aren't in the `status_forcelist`, instead we fast fail!
          response.raise_for_status()
          url_mime_type = response.headers['content-type']
          # handle PNG images
          if url_mime_type == 'image/png':
              # use PIL image to read the png file buffer
              image = Image.open(response.raw)
          else: # but assume all other images are JPEG
              # HWC PIL image
              image = Image.fromarray(
                  galaxy_dataset.decode_jpeg(response.raw.read()))
      except Exception as e:
          # add some logging on the failed url
          logging.critical('Cannot load {}'.format(url))
          # and make sure we reraise the error
          raise e

      # avoid the label lookups as they aren't used in the prediction
      label = []

      # logging.info((image.shape, torch.max(image), image.dtype, label))  # always 0-255 uint8

      if self.transform:
          # a CHW tensor, which torchvision wants. May change to PIL image.
          image = self.transform(image)

      if self.target_transform:
          label = self.target_transform(label)

      # logging.info((image.shape, torch.max(image), image.dtype, label))  #  should be 0-1 float
      return image


class PredictionGalaxyDataModule(galaxy_datamodule.GalaxyDataModule):
    # override the setup method to setup our prediction dataset on the prediction catalog
    def setup(self, stage: Optional[str] = None):
        self.predict_dataset = PredictionGalaxyDataset(catalog=self.predict_catalog, transform=self.transform)


def save_predictions_to_json(predictions, id_str, label_cols, save_loc):
    # JSON output format is used for services like the zooniverse subject assistant
    # Could add any other decision rules into this function
    assert save_loc.endswith('.json')
    # setup the output data structure with a schema describing the data
    output_data = {
      'schema': {
        'version': 1,
        'type': 'zooniverse/subject_assistant',
        'data': {
            'predictions': {'subject_id': ['smooth-or-featured-cd_smooth', 'smooth-or-featured-cd_featured-or-disk', 'smooth-or-featured-cd_problem']},
            'probabilities': {'subject_id': 'probability_at_least_20pc_featured'}
        }
      }
      # will add the actual data under 'data' key, below
    }

    # check that the predictions mean what we think they mean
    assert label_cols[0] == 'smooth-or-featured-cd_smooth'
    assert label_cols[1] =='smooth-or-featured-cd_featured-or-disk'
    assert label_cols[2] == 'smooth-or-featured-cd_problem'
    # okay, now it's safe to hardcode the values below
    
    # only derive each galaxies smooth or features question right now for simplicity of metric
    # i.e. we're trying to figure out if this galaxy is interesting or not for human volunteers
    # if it's not featured it's not interesting so we can use this metric to decide to show it to volunteers
    smooth_or_featured_start_and_end_indices = [0, 2]

    # the featured answer label index
    smooth_or_featured_featured_index = 1

    # upper bound of volunteers answering for a feature, i.e. no more than e.g. 20% of volunteers select the answer (here, featured)
    featured_upper_bound = 0.2

    # currently, probability volunteers would give featured vote fraction below 20%
    probability_volunteers_say_featured_below_bound = odds_answer_below_bounds(predictions,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_featured_index,
        featured_upper_bound
    )

    # just for convention and for the existing kade code, it seems more natural to have high probability galaxies be put in the active set
    # https://github.com/zooniverse/kade/blob/9413312ba9629bd256426eb400e06c47e8d0968f/app/services/prediction_results/process.rb#L33
    # so let's record the odds that volunteers say featured will be *above* the bound (i.e. at least 20%)
    probability_volunteers_say_featured_above_bound = 1 - probability_volunteers_say_featured_below_bound

    # output the probability data as subject_id: probability volunteers say featured above bound (rounded to 4dp)
    # note - no longer a percentage probability
    prediction_data = {id_str[n]: round(probability_volunteers_say_featured_above_bound[n], 4) for n in range(len(predictions))}

    # also record the predictions themselves, for debugging and subject tracking
    # any probabilities can be derived from the predictions post-hoc if needed
    # predictions[n, :3] slices out predictions for the nth galaxy and the 0 to 2nd questions i.e. smooth/featured/problem
    # (could generalise to e.g. smooth_or_featured_start_and_end_indices[0]:smooth_or_featured_start_and_end_indices[0]+1], but overcomplicated I think)
    probability_data = {id_str[n]: np.round(predictions[n, :3], decimals=3).tolist() for n in range(len(predictions))}

    # add the prediction data to the output data dict
    output_data['data'] = {'predictions': prediction_data, 'probabilities': probability_data}
    with open(save_loc, 'w') as out_file:
        json.dump(output_data, out_file)


def test_save_predictions_to_json():
    predictions = np.random.rand(20, 3) * 100 + 1
    id_strs = [str(x) for x in range(20)]
    label_cols = ['smooth-or-featured-cd_smooth', 'smooth-or-featured-cd_featured-or-disk', 'smooth-or-featured-cd_problem']
    save_loc = 'temp.json'

    save_predictions_to_json(predictions, id_strs, label_cols, save_loc)

    with open('temp.json', 'r') as f:
        saved_preds = json.load(f)
    print(saved_preds)


# note - this is pretty much a copy of zoobot code, it might be possible to just import it
def predict(catalog: pd.DataFrame, model: pl.LightningModule, n_samples: int, label_cols: List, save_loc: str, datamodule_kwargs, trainer_kwargs):
    # extract the uniq image identifiers
    image_id_strs = list(catalog['subject_id'])

    predict_datamodule = PredictionGalaxyDataModule(
        label_cols=None, # we don't need the labels for predictions
        predict_catalog=catalog,  # no need to specify the other catalogs
        # will use the default transforms unless overridden with datamodule_kwargs
        #
        **datamodule_kwargs  # e.g. batch_size, resize_size, crop_scale_bounds, etc.
    )

    # setup the preduction catalog - specify the stage, or will error (as missing other catalogs)
    predict_datamodule.setup(stage='predict')

    # set up trainer (again)
    trainer = pl.Trainer(
        max_epochs=-1,  # does nothing in this context, suppresses warning
        **trainer_kwargs  # e.g. gpus
    )

    logging.info('Beginning predictions')
    start = datetime.datetime.fromtimestamp(time.time())
    logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))

    # derive the predictions
    predictions = trainer.predict(model, predict_datamodule)
    logging.info(len(predictions))

    # trainer.predict gives list of tensors, each tensor being predictions for a batch. Concat on axis 0.
    # range(n_samples) list comprehension repeats this, for dropout-permuted predictions. Stack to create new last axis.
    # final shape (n_galaxies, n_answers, n_samples)
    predictions = torch.stack([torch.concat(predictions, dim=0) for n in range(n_samples)], dim=2).numpy()

    logging.info('Predictions complete - {}'.format(predictions.shape))
    logging.info(f'Saving predictions to {save_loc}')

    if save_loc.endswith('.csv'):      # save as pandas df
        save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)
    elif save_loc.endswith('.hdf5'):
        save_predictions.predictions_to_hdf5(predictions, image_id_strs, label_cols, save_loc)
    elif save_loc.endswith('.json'):
        save_predictions_to_json(predictions, image_id_strs, label_cols, save_loc)
    else:
        logging.warning('Save format of {} not recognised - assuming csv'.format(save_loc))
        save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)

    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))


def predictions_to_expectation_of_answer(predictions: np.ndarray, question_indices: List[int], answer_index: int) -> np.ndarray:
    """
    Calculate expected vote fraction for some answer, from Zoobot dirichlet predictions

    https://en.wikipedia.org/wiki/Dirichlet_distribution
    See properties, moments, E[Xi]

    Args:
        predictions (np.ndarray): Dirichlet concentrations of shape (n_galaxies, n_answers)
        question_indices (List[int]): Start and end column index of the question's answers (e.g. [0, 2] for smooth or featured)
        answer_index (int): Column index of the answer (e.g. 0 for smooth)

    Returns:
        np.ndarray: expected value of Dirichlet variable for that answer (i.e. the fraction of volunteers giving that answer), shape (n_galaxies)
    """

    # could do in one line but might as well be explicit/clear
    alpha_all = predictions[:, question_indices[0]:question_indices[1]+1].sum(axis=1)
    alpha_i = predictions[:, answer_index]
    return alpha_i / alpha_all


def predictions_to_variance_of_answer(predictions: np.ndarray,  question_indices: List[int], answer_index: int) -> np.ndarray:
    """
    Calculate variance (uncertainty) on vote fraction for some answer, from Zoobot dirichlet predictions

    https://en.wikipedia.org/wiki/Dirichlet_distribution
    See properties, moments, Var[Xi]

    Args:
        predictions (np.ndarray): Dirichlet concentrations of shape (n_galaxies, n_answers)
        question_indices (List[int]): Start and end column index of the question's answers (e.g. [0, 2] for smooth or featured)
        answer_index (int): Column index of the answer (e.g. 0 for smooth)

    Returns:
        np.ndarray: Variance (uncertainty) of Dirichlet variable for that answer (i.e. the variance on the fraction of volunteers giving that answer), shape (n_galaxies)
    """
    alpha_all = predictions[:, question_indices[0]:question_indices[1]+1].sum(axis=1)
    alpha_i = predictions[:, answer_index]
    return alpha_i * (alpha_all - alpha_i) / (alpha_all**2 * (alpha_all + 1))


def test_predictions_to_expectation_of_answer():

    predictions = np.array([[8., 2., 1.5], [4., 5., 1.5]])

    smooth_or_featured_start_and_end_indices = [0, 2]
    smooth_or_featured_smooth_index = 0

    expectations = predictions_to_expectation_of_answer(
        predictions,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_smooth_index
    )

    # first row should have higher expectation than second
    assert np.allclose(expectations, [0.69565217, 0.38095238])
    # print('Expectations: ', expectations)


def test_predictions_to_variance_of_answer():

    predictions = np.array([[8., 2., 1.5], [4., 5., 1.5]])

    smooth_or_featured_start_and_end_indices = [0, 2]
    smooth_or_featured_smooth_index = 0

    variances = predictions_to_variance_of_answer(
        predictions,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_smooth_index
    )
    # first row should have smaller variance than second
    assert np.allclose(variances, [0.01693762, 0.02050675])
    # print('Variances: ', variances)


def odds_answer_below_bounds(predictions: np.ndarray,  question_indices: List[int], answer_index: int, bound) -> np.ndarray:
    """
    Calculate the predicted odds that the galaxy would have an infinite-volunteer vote fraction no higher than `bound'
    (for a given question and answer)

    e.g. the predicted odds that an infinite number of volunteers would answer `smooth' to `smooth or featured' less than 20% of the time

    (If you want the odds above bounds, just do 1 - this)

    ("predicted infinite-volunteer vote fraction" is the intuitive way to say "the value drawn from the dirichlet distribution")

    Args:
        predictions (np.ndarray): Dirichlet concentrations of shape (n_galaxies, n_answers)
        question_indices (List[int]): Start and end column index of the question's answers (e.g. [0, 2] for smooth or featured)
        answer_index (int): Column index of the answer (e.g. 0 for smooth)
        bound (float, optional): highest allowed infinite-volunter vote fraction. Defaults to 0.2.

    Returns:
        np.ndarray: predicted odds that the galaxy would have an infinite-volunteer vote fraction no higher than `bound', shape (batch)
    """
    concentrations_q = predictions[:, question_indices[0]:question_indices[1]+1]
    concentrations_a = predictions[:, answer_index]
    # dirichlet of this or not this is equivalent to beta distribution with concentrations (this, sum_of_not_this)
    concentrations_not_a = concentrations_q.sum(axis=1) - concentrations_a
    # concentrations_a and concentrations_not_a have shape (batch)
    return beta(a=concentrations_a, b=concentrations_not_a).cdf(bound)  # will broadcast

    # NB: we can actually test this
    # samples_of_a = beta(a=concentrations_a, b=concentrations_not_a).rvs((1000, len(predictions)))  # will broadcast
    # print(np.mean(samples_of_a < bound, axis=0))  # should be similar to .cdf(bound) above


def test_predictions_to_bounds():

    predictions = np.array([[8., 2., 1.5], [4., 5., 1.5]])

    smooth_or_featured_start_and_end_indices = [0, 2]
    smooth_or_featured_smooth_index = 0

    odds_below_bound = odds_answer_below_bounds(
        predictions,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_smooth_index,
        bound=0.2
    )
    # print(odds_below_bound)
    # first row should be very likely high ([8., 3.5] concentrations, 2:1 ratio) so odds below 0.2 should be very low
    # second row should be somewhat likely low ([4, 6.5] concentrations, 2:3 ratio) so odds below 0.2 should be somewhat low
    assert np.allclose(odds_below_bound, [0.00013951, 0.10257097])
    # print('CDF: ', variances)


if __name__ == '__main__':
    # run the tests for the prediction metric functions
    test_predictions_to_expectation_of_answer()
    test_predictions_to_variance_of_answer()
    test_predictions_to_bounds()
    test_save_predictions_to_json()
