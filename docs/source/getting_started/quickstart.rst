Quickstart
**********************

1. Install q4l-customized version of qlib

Since current version of QuantBench still largely relies on qlib, we need to install a customized version of qlib first. This version of qlib is modified to support more features, and is still under development. You can install it via:

.. code-block:: bash

    pip install git+https://github.com/SaizhuoWang/qlib.git@dev_saizhuo

We are working on removing this dependency by integrating all related qlib features into q4l, so future version will be more coherent and will only require q4l installation.

2. Install q4l

.. code-block:: bash
    
    wget --http-user quantbench --http-password q4l_tEst_prV http://saizhuo.wang/dist/q4l-0.0.1.tar.gz
    tar -xzvf q4l-0.1.tar.gz
    cd q4l-0.0.1
    pip install -e .

3. Getting example data

Download the example data from this `link(OneDrive) <https://hkustconnect-my.sharepoint.com/:u:/g/personal/swangeh_connect_ust_hk/EVDul8YcwVhEsseMSYucWoEBIaV9VDEgP7riiu2qS2YHvQ?e=Rvo9CN>`_ or this `link(Baidu) <https://pan.baidu.com/s/1Zh5Jml8eRuogZPYONcczMw?pwd=89ww>`_. And put it under :code:`examples/benchmark/data`

.. code-block:: bash

    cd examples/benchmark/data
    tar -xzvf market_data.tar.gz

You may also refer to :code:`q4l/config/q4l_builtin/job/machine/local.yaml` and check :code:`data_root` if you want it to be put in another place.

4. Running LSTM example

Since we use mongodb to track experiment results, you need to start mongodb first. Let's suppose that you run a MongoDB server on localhost:27017, since this url is the default config in q4l. Then you can run the following command to start an experiment.

.. code-block:: bash

    cd examples/benchmark
    python src/main.py experiment/model=lstm

Experiment results are stored in the `mlruns` directory and you can use `mlflow ui` to view them. Mlflow is the experiment tracking system that is deeply integrated into q4l. By default we use local file system to track changes, and you can also specify mlflow tracking server url if you have deployed services.
 
5. Running more models

So far provides no more feature than qlib. Now let's try another model in q4l:

.. code-block:: bash

    python src/main.py experiment/model=patchtst

This will run a `PatchTST
<https://arxiv.org/abs/2211.14730>`_ model, with only 1 config change.

6. Running more experiments

Suppose you want to run multiple models, you can do this with one command:

.. code-block:: bash

    python src/main.py -m experiment/model=lstm,patchtst,fedformer,mixer

This command will run 4 experiments with models being LSTM/PatchTST/FEDFormer/Mixer respectively.

7. Changing dataset

You may also want to change the dataset to run experiments. This can also be achived via command-line changes. For example, if you want to change region and pool, just run:

.. code-block:: bash

    python src/main.py ++experiment.data.region=us ++experiment.data.pool=sp500

This will change the dataset to US stock market with SP500 stocks. If you want to change your alpha factors, just run:

.. code-block:: bash

    python src/main.py +experiment/data/loader/alpha@experiment.data.loader.alpha.alpha101=alpha101 \
        ++experiment.data.sampler.x_group=[alpha101]

This will change the alpha factors to `Alpha101
<http://arxiv.org/abs/1601.00991>`_, and change the x_group (i.e. the feature group) to alpha101.

Under the hood there are some other config changes, but they are handled by config groups in q4l.

There are a lot more components in q4l that can be tweaked, providing you with a lot of flexibility. Feel free to explore them in the following docs!