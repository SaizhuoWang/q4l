import numpy as np

from ....data.dataset import Q4LDataModule


def get_data(
    data: Q4LDataModule,
    partition: str,
    data_key: str,
    return_labels: bool = False,
):
    # train_sampler = data.prepare(
    #     partition="train", data_key=DK_L, return_sampler=True
    # )
    # valid_sampler = data.prepare(
    #     partition="valid", data_key=DK_L, return_sampler=True
    # )
    # test_sampler = data.prepare(
    #     partition="test", data_key=DK_L, return_sampler=True
    # )

    # # Training data numpy
    # x_train, y_train = train_sampler.get_all_valid_data()
    # x_train = np.reshape(x_train, (x_train.shape[0], -1))
    # y_train = np.squeeze(y_train)
    # # Validation data numpy
    # x_valid, y_valid = valid_sampler.get_all_valid_data()
    # x_valid = np.reshape(x_valid, (x_valid.shape[0], -1))
    # y_valid = np.squeeze(y_valid)

    # # Test data numpy
    # x_test, y_test = test_sampler.get_all_valid_data()
    # x_test = np.reshape(x_test, (x_test.shape[0], -1))
    # y_test = np.squeeze(y_test)

    # return x_train, y_train, x_valid, y_valid, x_test, y_test
    sampler = data.prepare(
        partition=partition, data_key=data_key, return_sampler=True
    )
    x, y, label = sampler.get_all_valid_data()
    x = np.reshape(x, (x.shape[0], -1))
    y = np.squeeze(y)
    if not return_labels:
        return x, y
    else:
        return x, y, label
