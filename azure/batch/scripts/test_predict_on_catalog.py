import pytest

import json
import os

import numpy as np

import predict_on_catalog

@pytest.fixture
def predictions_with_sample_dim(n_samples=5):
    predictions = np.array([[8., 2., 1.5], [4., 5., 1.5]])
    predictions = np.stack([predictions] * n_samples, axis=2)
    return predictions


@pytest.fixture
def expectation_with_sample_dim(n_samples=5):
    return np.stack([[0.69565217, 0.38095238]] * n_samples, axis=1)


@pytest.fixture
def variance_with_sample_dim(n_samples=5):
    return np.stack([[0.01693762, 0.02050675]] * n_samples, axis=1)


@pytest.fixture
def odds_below_bound_with_sample_dim(n_samples=5):
    return np.stack([[0.00013951, 0.10257097]] * n_samples, axis=1)


@pytest.fixture
def real_predictions():
    # some real predictions for Cosmic Dawn
    predictions = np.array([
       [92.65231323,  3.26797128, 25.24700928],
       [ 93.7562027,   3.7324903, 33.72304916],
       [82.39868164,  6.23918152, 20.55649376],
       [73.68450928,  8.03439522, 20.93917465],
       [76.07131958,  6.40088654,  29.8066597],
       [54.40034485, 12.83099937, 13.50455379],
       [90.39558411,  5.74498463, 40.09345245],
       [44.36257935, 21.92322922, 14.49137306],
       [57.88036728, 10.58429527,  14.5579319],
       [15.32801437, 23.56198311,  7.74940348],
       [76.99712372,  5.80586195, 47.61122131],
       [80.41983795,  4.59404898, 42.60891342],
       [91.29488373,  5.62464571, 37.56932831],
       [17.39572906, 34.65762711,  8.72911072],  # this is index 14, likely to be featured
       [54.37077332,  20.0857563, 18.13856125],
       [33.16508484,  7.55197144, 15.04645443],
       [  5.2865777,  2.25175548, 26.42889023],
       [ 5.95480394,  2.10367179, 38.06949234],
       [77.01819611,  5.05003738, 25.69354248],
       [81.80924988,  5.31926441, 21.41218758]])
    predictions = np.expand_dims(predictions, axis=2)  # (galaxies, answers, 1)
    return predictions


@pytest.fixture
def predictions_for_two_samples(real_predictions):
    return np.dstack([real_predictions]*2)


def test_predictions_to_expectation_of_answer(predictions_with_sample_dim, expectation_with_sample_dim):

    smooth_or_featured_start_and_end_indices = [0, 2]
    smooth_or_featured_smooth_index = 0

    expectations = predict_on_catalog.predictions_to_expectation_of_answer(
        predictions_with_sample_dim,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_smooth_index
    )

    # first row should have higher expectation than second
    assert np.allclose(expectations, expectation_with_sample_dim)
    # print('Expectations: ', expectations)


def test_predictions_to_variance_of_answer(predictions_with_sample_dim, variance_with_sample_dim):

    smooth_or_featured_start_and_end_indices = [0, 2]
    smooth_or_featured_smooth_index = 0

    variances = predict_on_catalog.predictions_to_variance_of_answer(
        predictions_with_sample_dim,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_smooth_index
    )
    # first row should have smaller variance than second
    assert np.allclose(variances, variance_with_sample_dim)
    # print('Variances: ', variances)


def test_save_predictions_to_json(real_predictions):

    id_strs = [str(x) for x in range(len(real_predictions))]
    label_cols = ['smooth-or-featured-cd_smooth', 'smooth-or-featured-cd_featured-or-disk', 'smooth-or-featured-cd_problem']
    save_loc = 'temp.json'
    predict_on_catalog.save_predictions_to_json(real_predictions, id_strs, label_cols, save_loc)
    # process the saved results file for testing
    with open(save_loc, 'r') as f:
        saved_preds = json.load(f)
        # print(saved_preds)
    try:
        # 'data': { 
        #     'subject_id': {
        #         'sample_num': ['probability_at_least_20pc_featured', [...predictions] ] 
        #     }
        # }
        subject_id = '14'
        sample_num = '0' # only 1 sample in the real_predictions fixtures
        assert saved_preds['data'][subject_id][sample_num][0] > 0.5
        assert saved_preds['data'][subject_id][sample_num][1] == [54.371, 20.086, 18.139]
    finally:
        # cleanup the test file artefact
        os.unlink(save_loc)


def test_multi_samples_save_predictions_to_json(predictions_for_two_samples):

    id_strs = [str(x) for x in range(len(predictions_for_two_samples))]
    label_cols = ['smooth-or-featured-cd_smooth', 'smooth-or-featured-cd_featured-or-disk', 'smooth-or-featured-cd_problem']
    save_loc = 'temp.json'
    predict_on_catalog.save_predictions_to_json(predictions_for_two_samples, id_strs, label_cols, save_loc)
    # process the saved results file for testing
    with open(save_loc, 'r') as f:
        saved_preds = json.load(f)
        # print(saved_preds)
    try:
        # 'data': { 
        #     'subject_id': {
        #         'sample_num': ['probability_at_least_20pc_featured', [...predictions] ] 
        #     }
        # }
        subject_id = '14'
        first_sample = '0' # inspect the first sample
        second_sample = '1' # inspect the first sample
        assert saved_preds['data'][subject_id][first_sample][0] > 0.5
        assert saved_preds['data'][subject_id][first_sample][1] == [54.371, 20.086, 18.139]
        assert saved_preds['data'][subject_id][second_sample][0] > 0.5
        assert saved_preds['data'][subject_id][second_sample][1] == [54.371, 20.086, 18.139]
    finally:
        # cleanup the test file artefact
        os.unlink(save_loc)


def test_predictions_to_bounds(predictions_with_sample_dim, odds_below_bound_with_sample_dim):

    smooth_or_featured_start_and_end_indices = [0, 2]
    smooth_or_featured_smooth_index = 0

    odds_below_bound = predict_on_catalog.odds_answer_below_bounds(
        predictions_with_sample_dim,
        smooth_or_featured_start_and_end_indices,
        smooth_or_featured_smooth_index,
        bound=0.2
    )
    # print(odds_below_bound)
    # first row should be very likely high ([8., 3.5] concentrations, 2:1 ratio) so odds below 0.2 should be very low
    # second row should be somewhat likely low ([4, 6.5] concentrations, 2:3 ratio) so odds below 0.2 should be somewhat low
    assert np.allclose(odds_below_bound, odds_below_bound_with_sample_dim)
    # print('CDF: ', variances)