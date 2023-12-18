Install QuantBench (q4l)
****

1. Install q4l-customized version of qlib

.. code-block:: bash

    pip install git+git@github.com:SaizhuoWang/qlib.git@dev_saizhuo


2. Install q4l

.. code-block:: bash
    
    wget http://saizhuo.wang/dist/q4l-0.1.tar.gz
    tar -xzvf q4l-0.1.tar.gz
    cd q4l
    pip install -e .



3. Running examples

.. code-block:: bash

    cd examples/benchmark
    python src/main.py experiment/model=lstm


