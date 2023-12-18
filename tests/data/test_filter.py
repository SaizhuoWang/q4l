import numpy as np
import pytest

from q4l.data.filter import TSNaNFilter


# Test the TSNaNFilter class
def test_TSNaNFilter():
    # Create an instance of TSNaNFilter with a nan_ratio of 0.2
    filter_obj = TSNaNFilter(nan_ratio=0.2)

    # Create a sample data array of shape (T, N, W, F)
    sample_data = np.array(
        [
            [
                [[1.0, 2.0], [np.nan, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            [
                [[1.0, np.nan], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        ]
    )

    # Test the filter method of TSNaNFilter
    valid_mask = filter_obj.filter(sample_data)

    # The expected valid mask after filtering the sample_data with nan_ratio 0.2
    expected_valid_mask = np.array(
        [
            [False, True],
            [False, True],
        ]
    )

    # Assert that the valid_mask is equal to the expected_valid_mask
    np.testing.assert_array_equal(valid_mask, expected_valid_mask)


@pytest.fixture
def data_package():
    return np.random.rand(1000, 10, 50, 5)


def test_TSNaNFilter_large(data_package):
    # set the nan ratio to 0.5
    nan_ratio = 0.5
    filter = TSNaNFilter(nan_ratio=nan_ratio)

    # create a test dataset with 10% of NaN values
    nan_mask = np.random.choice([False, True], size=(1000, 10, 50, 5), p=[0.9, 0.1])
    data_package[np.nonzero(nan_mask)] = np.nan

    # apply the filter
    filtered_mask = filter.filter(data_package)

    # assert that the filtered mask has the correct shape
    assert filtered_mask.shape == (1000, 10)

    # assert that the filtered mask is correct
    expected_mask = np.mean(nan_mask, axis=(-1, -2)) < nan_ratio
    np.testing.assert_array_equal(filtered_mask, expected_mask)
