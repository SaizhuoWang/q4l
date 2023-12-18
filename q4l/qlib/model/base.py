# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import abc
from abc import abstractmethod
from typing import Text, Union

from ..data.dataset import Dataset
from ..data.dataset.weight import Reweighter
from ..utils.serial import Serializable


class BaseModel(Serializable, metaclass=abc.ABCMeta):
    """Modeling things."""

    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> object:
        """Make predictions after modeling things."""

    def __call__(self, *args, **kwargs) -> object:
        """Leverage Python syntactic sugar to make the models' behaviors like
        functions."""
        return self.predict(*args, **kwargs)


class Model(BaseModel):
    """Learnable Models."""

    @property
    def model(self):
        raise NotImplementedError(
            "Please implement this method in your model class"
        )

    def load_checkpoint(self):
        self._load_checkpoint()

    @abstractmethod
    def _load_checkpoint(self):
        raise NotImplementedError(
            "Please implement this method in your model class"
        )

    def fit(self, dataset: Dataset, reweighter: Reweighter):
        """Learn model from the base model.

        .. note::

            The attribute names of learned model should `not` start with '_'. So that the model could be
            dumped to disk.

        The following code example shows how to retrieve `x_train`, `y_train` and `w_train` from the `dataset`:

            .. code-block:: Python

                # get features and labels
                df_train, df_valid = dataset.prepare(
                    ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
                )
                x_train, y_train = df_train["feature"], df_train["label"]
                x_valid, y_valid = df_valid["feature"], df_valid["label"]

                # get weights
                try:
                    wdf_train, wdf_valid = dataset.prepare(["train", "valid"], col_set=["weight"],
                                                           data_key=DataHandlerLP.DK_L)
                    w_train, w_valid = wdf_train["weight"], wdf_valid["weight"]
                except KeyError as e:
                    w_train = pd.DataFrame(np.ones_like(y_train.values), index=y_train.index)
                    w_valid = pd.DataFrame(np.ones_like(y_valid.values), index=y_valid.index)

        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed data from model training.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(
        self, dataset: Dataset, segment: Union[Text, slice] = "test"
    ) -> object:
        """Give prediction given Dataset.

        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed dataset from model training.

        segment : Text or slice
            dataset will use this segment to prepare data. (default=test)

        Returns
        -------
        Prediction results with certain type such as `pandas.Series`.

        """
        raise NotImplementedError()


class ModelFT(Model):
    """Model (F)ine(t)unable."""

    @abc.abstractmethod
    def finetune(self, dataset: Dataset):
        """Finetune model based given dataset.

        A typical use case of finetuning model with q4l.qlib.workflow.R

        .. code-block:: python

            # start exp to train init model
            with R.start(experiment_name="init models"):
                model.fit(dataset)
                R.save_objects(init_model=model)
                rid = R.get_recorder().id

            # Finetune model based on previous trained model
            with R.start(experiment_name="finetune model"):
                recorder = R.get_recorder(recorder_id=rid, experiment_name="init models")
                model = recorder.load_object("init_model")
                model.finetune(dataset, num_boost_round=10)


        Parameters
        ----------
        dataset : Dataset
            dataset will generate the processed dataset from model training.

        """
        raise NotImplementedError()
