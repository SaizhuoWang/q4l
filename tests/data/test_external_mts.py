import numpy as np
import pandas as pd
import pytest

from q4l import (
    DFOnlyDataset,
    METRLADataset,
    PEMSBAYDataset,
    PEMSDxDataset,
    SingleDFSampler,
    df_to_adj_matrix,
)


def test_df_to_adj_matrix():
    data = {"i": [0, 1, 2], "j": [1, 2, 0], "w": [1.0, 2.0, 3.0]}
    df = pd.DataFrame(data)
    n_nodes = 3
    adj_matrix = df_to_adj_matrix(df, n_nodes)
    expected_matrix = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    assert np.array_equal(adj_matrix, expected_matrix), "Adjacency matrix conversion is incorrect"


@pytest.fixture
def dummy_data_dir(tmpdir):
    data_dir = tmpdir.mkdir("data")
    pems_bay_meta = data_dir.join("PEMS-BAY-META.csv")
    pems_bay_meta.write("i,j,w\n0,1,1.0\n1,2,2.0\n2,0,3.0")
    pems_bay = data_dir.join("PEMS-BAY.csv")
    pems_bay.write("0,1,2\n1,2,3")
    return data_dir


def test_pemsbaydataset(dummy_data_dir):
    dataset = PEMSBAYDataset(dummy_data_dir)
    assert isinstance(dataset.metadata, pd.DataFrame), "Metadata should be a pandas DataFrame"
    assert isinstance(dataset.ts_df, pd.DataFrame), "Time series data should be a pandas DataFrame"


def test_pemsdxdataset(dummy_data_dir):
    dataset = PEMSDxDataset(dummy_data_dir, x=1)
    assert isinstance(dataset.adj_matrix, np.ndarray), "Adjacency matrix should be a numpy array"
    assert isinstance(dataset.ts_df, pd.Series), "Time series data should be a pandas Series"


def test_metrladataset(dummy_data_dir):
    dataset = METRLADataset(dummy_data_dir)
    assert isinstance(dataset.adj_matrix, np.ndarray), "Adjacency matrix should be a numpy array"
    assert isinstance(dataset.ts_df, pd.DataFrame), "Time series data should be a pandas DataFrame"


def test_dfonydataset(dummy_data_dir):
    dataset = DFOnlyDataset(dummy_data_dir, "PEMS-BAY.csv")
    assert isinstance(dataset.ts_df, pd.DataFrame), "Time series data should be a pandas DataFrame"


def test_singledfsampler(dummy_data_dir):
    dataset = PEMSBAYDataset(dummy_data_dir)
    sampler = SingleDFSampler(df=dataset.ts_df, x_window=12, y_window=3, sample_mode="C")
    assert sampler.sample_mode in ("C", "T"), "Sample mode should be 'C' or 'T'"
    assert len(sampler) == 2 * 3 - 12 - 3 + 1, "Length of sampler is incorrect"

    # Test the __getitem__ method
    item = sampler.__getitem__(0)
    assert "index" in item, "Item should contain the 'index' key"
    assert "x" in item, "Item should contain the 'x' key"
    assert "y" in item, "Item should contain the 'y' key"

    # Test the collate_fn method
    batch = [item, item]
    collated = sampler.collate_fn(batch)
    assert "index" in collated, "Collated batch should contain the 'index' key"
    assert "x" in collated, "Collated batch should contain the 'x' key"
    assert "y" in collated, "Collated batch should contain the 'y' key"
    assert len(collated["index"]) == len(batch), "Length of index in collated batch is incorrect"
    assert collated["x"].shape[0] == len(batch), "Length of x in collated batch is incorrect"
    assert collated["y"].shape[0] == len(batch), "Length of y in collated batch is incorrect"
